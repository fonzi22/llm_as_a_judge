"""Offline tests: ID alignment, authoritative GT, API failures and resumability."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch
from types import SimpleNamespace
import urllib.error

from llm_eval import eval_matching as evaluator


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.annotation_path = self.root/'annotations.json'
        self.annotation_path.write_text(json.dumps({'images': [{
            'image_id': '000', 'density_group': 'same_object', 'level': 'L1',
            'qa_annotations': [
                {'qa_id': '000_qa_0001', 'question_id': '000_q0001',
                 'question': 'What color?', 'answer': 'Yellow.', 'question_type': 'Attribute'},
                {'qa_id': '000_qa_0002', 'question_id': '000_q0002',
                 'question': 'What shape?', 'answer': 'Rectangular.', 'question_type': 'Spatial'},
            ]}]}))
        self.annotations, self.aliases = evaluator.load_annotations(self.annotation_path)
        self.config = dict(model='test-judge', base_url='http://localhost:8000/v1',
                           temperature=0, max_tokens=64, timeout_sec=1, max_retries=1)
        self.rows = [{'image_id': '000', 'question_id': '000_q0001', 'answer': 'Yellow'},
                     {'image_id': '000', 'question_id': '000_q0002', 'answer': 'A rectangle'}]

    def write_predictions(self, rows=None, name='predictions.jsonl', metadata=None):
        path = self.root/name
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = self.rows if rows is None else rows
        if path.suffix == '.jsonl':
            path.write_text('\n'.join(json.dumps(r) for r in rows)+'\n')
        else:
            path.write_text(json.dumps(dict(metadata, results=rows) if metadata is not None else rows))
        return path

    def load(self, path, partial=False):
        return evaluator.load_tasks(path, self.annotations, partial, self.aliases)

    @staticmethod
    def api_reply(content='1', **kwargs):
        data = dict(choices=[dict(message={'content': content}, finish_reason='stop')], **kwargs)
        return io.BytesIO(json.dumps(data).encode())

    def test_all_three_source_schemas_and_aliases(self):
        minimal = self.write_predictions()
        tasks, info = self.load(minimal)
        self.assertEqual([t['qa_id'] for t in tasks], list(self.annotations))
        self.assertEqual(info['available_predictions'], 2)
        self.assertEqual(tasks[0]['level'], 'L1')
        for metadata in (None, {'model': 'google/gemini-example'}):
            rows = [dict(qa_id=t['qa_id'], question_id=t['question_id'], prediction=t['prediction'],
                         ground_truth='Stale GT', question=t['question'],
                         model=('gemini-example' if i else 'google/gemini-example'))
                    for i, t in enumerate(tasks)]
            if metadata is None:
                for row in rows:
                    row.pop('model')
            path = self.write_predictions(rows, 'gemini.json', metadata)
            loaded, info = self.load(path)
            self.assertEqual(loaded[0]['ground_truth'], 'Yellow.')
            self.assertEqual(info['stale_ground_truth_samples'], 2)

    def test_algo1_schema_complete_status_and_empty_predictions(self):
        rows = [dict(question_id=r['question_id'], image_id=r['image_id'],
                     pred_answer=r['answer'], status='complete') for r in self.rows]
        path = self.write_predictions(rows, 'algo1.json')
        tasks, info = self.load(path)
        self.assertEqual([t['prediction'] for t in tasks], ['Yellow', 'A rectangle'])
        self.assertEqual(info['source_error_samples'], 0)
        rows[1]['pred_answer'] = ''
        path = self.write_predictions(rows, 'algo1.json')
        with self.assertRaises(ValueError):
            self.load(path)
        tasks, info = self.load(path, partial=True)
        self.assertEqual(len(tasks), 1)
        self.assertEqual(info['source_error_samples'], 1)
        self.assertEqual(info['source_error_ids'], ['000_qa_0002'])

    def test_invalid_identity_never_falls_back_to_row_order(self):
        mutations = (
            lambda r: r.update(question_id='unknown'),
            lambda r: r.update(qa_id='000_qa_0002'),
            lambda r: r.update(image_id='999'),
            lambda r: r.update(question='Different question?'),
        )
        for mutate in mutations:
            rows = copy.deepcopy(self.rows)
            mutate(rows[0])
            with self.subTest(row=rows[0]), self.assertRaises(ValueError):
                self.load(self.write_predictions(rows))
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            self.load(self.write_predictions([self.rows[0], self.rows[0]]))

    def test_partial_inputs_and_error_denominators(self):
        rows = copy.deepcopy(self.rows)
        rows[1]['answer'] = ''
        path = self.write_predictions(rows)
        with self.assertRaisesRegex(ValueError, 'allow-partial'):
            self.load(path)
        tasks, info = self.load(path, partial=True)
        self.assertEqual(info['source_error_ids'], ['000_qa_0002'])
        result = dict(tasks[0], judge_status='success', match=1)
        summary = evaluator.summarize(tasks, {result['sample_id']: result}, info)
        self.assertEqual(summary['accuracy'], 0.5)
        self.assertEqual(summary['accuracy_on_judged'], 1)
        result.update(judge_status='error', match=None)
        summary = evaluator.summarize(tasks, {result['sample_id']: result}, info)
        self.assertIsNone(summary['accuracy'])
        self.assertEqual(summary['judge_error_samples'], 1)
        _, info = self.load(self.write_predictions(self.rows[:1]), partial=True)
        self.assertEqual(info['missing_ids'], ['000_qa_0002'])

    def test_discovery_excludes_tests_configs_and_identical_copies(self):
        folder = self.root/'predictions'
        full = self.write_predictions(name='predictions/model/full/model.jsonl')
        self.write_predictions(name='predictions/copy/model.jsonl')
        self.write_predictions(self.rows[:1], name='predictions/model/test/model.limit1.jsonl')
        (folder/'model.config.json').write_text('{}')
        files, skipped = evaluator.discover_predictions([folder])
        self.assertEqual(len(files), 1)
        self.assertEqual(len(skipped), 1)
        self.assertEqual(len(evaluator.discover_predictions([folder], include_tests=True)[0]), 2)
        # Explicit files can be used for a smoke test.
        explicit = folder/'model/test/model.limit1.jsonl'
        self.assertEqual(evaluator.discover_predictions([explicit])[0], [explicit])
        # Two differently named baselines can legitimately produce identical answers.
        self.write_predictions(name='predictions/another-model.jsonl')
        self.assertEqual(len(evaluator.discover_predictions([folder])[0]), 2)
        out = self.root/'out'
        self.assertNotEqual(evaluator.output_path_for(full, out),
                            evaluator.output_path_for(folder/'copy/model.jsonl', out))

    def test_checkpoint_resume_and_gt_change(self):
        path = self.write_predictions()
        tasks, info = self.load(path)
        output = self.root/'out/judgments.json'
        with patch.object(evaluator.urllib.request, 'urlopen',
                          side_effect=[self.api_reply('1'), self.api_reply('0')]) as api:
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertTrue(evaluator.evaluate(tasks, info, self.config, output, None))
                self.assertTrue(evaluator.evaluate(tasks, info, self.config, output, None))
            self.assertEqual(api.call_count, 2)
            request = api.call_args_list[0].args[0]
            payload = json.loads(request.data)
            self.assertIn('Yellow.', payload['messages'][1]['content'])
            self.assertNotIn('Authorization', request.headers)
        report = json.loads(output.read_text())
        self.assertEqual(report['summary']['accuracy'], 0.5)
        self.assertEqual(report['summary']['by_question_type']['Attribute']['matched'], 1)
        self.assertTrue(output.with_suffix('.csv').exists())
        self.annotations['000_qa_0001']['answer'] = 'Blue.'
        tasks, info = self.load(path)
        with patch.object(evaluator.urllib.request, 'urlopen', return_value=self.api_reply('0')) as api:
            with contextlib.redirect_stdout(io.StringIO()):
                evaluator.evaluate(tasks, info, self.config, output, None)
            self.assertEqual(api.call_count, 1)
        changed_config = dict(self.config, model='another-judge')
        with self.assertRaisesRegex(ValueError, 'config/prompt changed'):
            evaluator.evaluate(tasks, info, changed_config, output, None)

    def test_bad_judge_response_is_an_error_and_can_be_retried(self):
        tasks, info = self.load(self.write_predictions(self.rows[:1]), partial=True)
        output = self.root/'judgments.json'
        for response in (self.api_reply('probably 1'), io.BytesIO(b'{"choices": []}')):
            with patch.object(evaluator.urllib.request, 'urlopen', return_value=response):
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertFalse(evaluator.evaluate(tasks, info, self.config, output, None))
            report = json.loads(output.read_text())
            self.assertIsNone(report['results'][0]['match'])
            self.assertIsNone(report['summary']['accuracy'])
        with patch.object(evaluator.urllib.request, 'urlopen', return_value=self.api_reply('1')):
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertTrue(evaluator.evaluate(tasks, info, self.config, output, None))

    def test_fatal_http_saves_checkpoint_and_stops(self):
        tasks, info = self.load(self.write_predictions())
        output = self.root/'judgments.json'
        error = urllib.error.HTTPError('http://localhost', 401, 'Unauthorized', {}, None)
        with patch.object(evaluator.urllib.request, 'urlopen', side_effect=error) as api:
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, 'checkpoint saved'):
                evaluator.evaluate(tasks, info, self.config, output, 'test-key')
            self.assertEqual(api.call_count, 1)
        report = json.loads(output.read_text())
        self.assertEqual(report['summary']['pending_samples'], 1)
        self.assertNotIn('test-key', output.read_text())

    def test_dry_run_makes_no_requests_or_output(self):
        path = self.write_predictions()
        config = self.root/'config.json'
        config.write_text(json.dumps(self.config))
        output_dir = self.root/'out'
        with patch.object(evaluator.urllib.request, 'urlopen') as api:
            with contextlib.redirect_stdout(io.StringIO()):
                evaluator.main(['--config', str(config), '--annotations', str(self.annotation_path),
                                '--predictions', str(path), '--output-dir', str(output_dir), '--dry-run'])
            api.assert_not_called()
        self.assertFalse(output_dir.exists())

    def test_cli_writes_comparison_and_resumes_without_api_calls(self):
        source = self.write_predictions()
        config = self.root/'config.json'
        config.write_text(json.dumps(self.config))
        output_dir = self.root/'out'
        arguments = ['--config', str(config), '--annotations', str(self.annotation_path),
                     '--predictions', str(source), '--output-dir', str(output_dir)]
        with patch.object(evaluator.urllib.request, 'urlopen',
                          side_effect=[self.api_reply('1'), self.api_reply('0')]) as api:
            with contextlib.redirect_stdout(io.StringIO()):
                evaluator.main(arguments)
                evaluator.main(arguments)
            self.assertEqual(api.call_count, 2)
        summaries = json.loads((output_dir/'summary.json').read_text())['models']
        self.assertEqual(len(summaries), 1)
        self.assertEqual(summaries[0]['accuracy'], 0.5)
        self.assertEqual(summaries[0]['judge_model'], 'test-judge')
        self.assertTrue((output_dir/'summary.csv').exists())

    def test_strict_judgment_parsing(self):
        for value, expected in [('1', 1), (' 0\n', 0), ('<answer>1</answer>', 1)]:
            self.assertEqual(evaluator.parse_binary_judgment(value), expected)
        for value in ('yes', '10', '1 or 0', '<think>reason</think>1', ''):
            with self.assertRaises(ValueError):
                evaluator.parse_binary_judgment(value)

    def fake_vllm(self, text='1', finish_reason='stop'):
        result = SimpleNamespace(prompt_token_ids=[1, 2, 3], outputs=[
            SimpleNamespace(text=text, token_ids=[4], finish_reason=finish_reason)])
        module = MagicMock()
        module.LLM.return_value.chat.return_value = [result]
        return module

    def test_local_backend_runs_in_process_and_reuses_engine_across_files(self):
        first = self.write_predictions(name='one.jsonl')
        second = self.write_predictions(name='two.jsonl')
        weights = self.root/'weights'
        weights.mkdir()
        config = dict(self.config, backend='local', model=str(weights),
                      vllm={'dtype': 'bfloat16'}, chat_template_kwargs={'enable_thinking': False})
        config.pop('base_url')
        config.pop('timeout_sec')
        config_path = self.root/'local.json'
        config_path.write_text(json.dumps(config))
        out = self.root/'out'
        arguments = ['--config', str(config_path), '--annotations', str(self.annotation_path),
                     '--predictions', str(first), str(second), '--output-dir', str(out)]
        fake = self.fake_vllm()
        with patch.dict('sys.modules', {'vllm': fake}), patch.dict('os.environ', {}), \
                patch.object(evaluator.urllib.request, 'urlopen') as api:
            with contextlib.redirect_stdout(io.StringIO()):
                evaluator.main(arguments)
                evaluator.main(arguments)
            api.assert_not_called()
            fake.LLM.assert_called_once_with(model=str(weights), tokenizer=str(weights),
                                             dtype='bfloat16', generation_config='vllm')
            self.assertEqual(fake.LLM.return_value.chat.call_count, 4)
        summary = json.loads((out/'summary.json').read_text())
        self.assertEqual(len(summary['models']), 2)
        self.assertTrue(all(s['accuracy'] == 1 for s in summary['models']))

    def test_local_cache_resolution_is_offline(self):
        hub = MagicMock()
        hub.snapshot_download.return_value = str(self.root/'cached-snapshot')
        fake = self.fake_vllm()
        config = dict(self.config, backend='local', model='org/cached-judge')
        with patch.dict('sys.modules', {'vllm': fake, 'huggingface_hub': hub}), \
                patch.dict('os.environ', {}):
            judge = evaluator.create_judge(config)
            self.assertEqual(judge.generate('prompt'), '1')
            hub.snapshot_download.assert_called_once_with('org/cached-judge', local_files_only=True,
                                                          cache_dir=None)
            kwargs = fake.LLM.call_args.kwargs
            self.assertEqual(kwargs['model'], str(self.root/'cached-snapshot'))
            self.assertEqual(judge.last_response['usage']['prompt_tokens'], 3)

    def test_local_missing_weights_stop_before_judging_samples(self):
        hub = MagicMock()
        hub.snapshot_download.side_effect = FileNotFoundError('not cached')
        config = dict(self.config, backend='local', model='org/missing-judge')
        tasks, info = self.load(self.write_predictions())
        output = self.root/'out/judgments.json'
        with patch.dict('sys.modules', {'huggingface_hub': hub}), \
                self.assertRaisesRegex(RuntimeError, 'No weights were downloaded'):
            evaluator.evaluate(tasks, info, config, output)
        report = json.loads(output.read_text())
        self.assertEqual(report['summary']['pending_samples'], 2)
        self.assertEqual(report['summary']['judge_error_samples'], 0)

    def test_empty_checkpoint_can_restart_after_model_path_fix(self):
        tasks, info = self.load(self.write_predictions())
        output = self.root/'out/judgments.json'
        evaluator.save_outputs(output, tasks, {}, self.config, 'old-signature', info)
        fixed = dict(self.config, model='fixed-model-path')
        with patch.object(evaluator.urllib.request, 'urlopen',
                          side_effect=[self.api_reply('1'), self.api_reply('0')]):
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertTrue(evaluator.evaluate(tasks, info, fixed, output))
        report = json.loads(output.read_text())
        self.assertEqual(report['judge_config']['model'], 'fixed-model-path')
        self.assertEqual(report['summary']['judged_samples'], 2)

    def test_local_truncation_is_not_a_binary_judgment(self):
        config = dict(self.config, backend='local', model=str(self.root))
        with patch.dict('sys.modules', {'vllm': self.fake_vllm(finish_reason='length')}), \
                patch.dict('os.environ', {}):
            judge = evaluator.create_judge(config)
            with self.assertRaisesRegex(ValueError, 'truncated'):
                judge.generate('prompt')

    def test_local_dry_run_needs_no_weights_or_runtime(self):
        path = self.write_predictions()
        config = dict(self.config, backend='local', model='/missing/model')
        config.pop('base_url')
        config.pop('timeout_sec')
        config_path = self.root/'config.json'
        config_path.write_text(json.dumps(config))
        with patch.object(evaluator.LocalJudge, 'prepare') as prepare, \
                patch.object(evaluator.urllib.request, 'urlopen') as api:
            with contextlib.redirect_stdout(io.StringIO()):
                evaluator.main(['--config', str(config_path), '--annotations', str(self.annotation_path),
                                '--predictions', str(path), '--dry-run'])
            prepare.assert_not_called()
            api.assert_not_called()


if __name__ == '__main__':
    unittest.main()
