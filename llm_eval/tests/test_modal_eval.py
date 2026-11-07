"""Modal integration checks without an account, network requests or GPU."""
import importlib.util
import json
from pathlib import Path, PurePosixPath
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

if importlib.util.find_spec('modal'):
    from llm_eval import modal_eval as runner
else:
    runner = None


@unittest.skipIf(runner is None, 'Install llm_eval/requirements-modal.txt for Modal checks')
class ModalEvaluationTests(unittest.TestCase):
    def test_windows_client_keeps_remote_paths_posix(self):
        self.assertIsInstance(runner.REMOTE_ROOT, PurePosixPath)
        args = runner.evaluation_args(runner.REMOTE_ROOT, 'nam/full_8b.jsonl',
                                      PurePosixPath('/outputs/test'))
        self.assertEqual(args[1], '/opt/llm_eval/configs/qwen_matching_modal.json')
        self.assertEqual(args[5], '/opt/llm_eval/predictions/nam/full_8b.jsonl')
        with self.assertRaises(ValueError):
            runner.prediction_relative_path(str(runner.ROOT / 'annotations.json'))
        with self.assertRaises(ValueError):
            runner.validate_run_name('../other-run')

    def test_download_preserves_nested_reports_and_replaces_files(self):
        volume = MagicMock()
        volume.iterdir.return_value = [
            SimpleNamespace(path='test/summary.csv', type=runner.modal.volume.FileEntryType.FILE),
            SimpleNamespace(path='/test/nam/judgments.json', type=runner.modal.volume.FileEntryType.FILE),
        ]
        volume.read_file.return_value = [b'first', b'second']
        with tempfile.TemporaryDirectory() as temp, patch.object(runner, 'output_volume', volume):
            root = Path(temp)
            (root / 'summary.csv').write_text('old report')
            runner.download_results('test', root)
            self.assertEqual((root / 'summary.csv').read_bytes(), b'firstsecond')
            self.assertEqual((root / 'nam/judgments.json').read_bytes(), b'firstsecond')
            self.assertEqual(list(root.rglob('*.download')), [])

    def test_remote_validates_before_download_and_commits_failed_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'configs').mkdir()
            (root / 'configs' / runner.CONFIG_NAME).write_text(json.dumps({
                'model': 'test/model', 'cache_dir': '/hf-cache/hub'}))
            events = []
            def evaluate(args):
                events.append('validate' if '--dry-run' in args else 'judge')
                if '--dry-run' not in args:
                    raise SystemExit(1)
            evaluator = SimpleNamespace(main=evaluate)
            hub = SimpleNamespace(snapshot_download=lambda *a, **kw: events.append('download'))
            outputs, cache = MagicMock(), MagicMock()
            with patch.object(runner, 'REMOTE_ROOT', root), \
                 patch.object(runner, 'output_volume', outputs), \
                 patch.object(runner, 'cache_volume', cache), \
                 patch.dict('sys.modules', {'eval_matching': evaluator, 'huggingface_hub': hub}), \
                 patch('sys.path', list(__import__('sys').path)):
                status = runner.run_evaluation.local('.', 'test', False, False)
            self.assertEqual(status, 1)
            self.assertEqual(events, ['validate', 'download', 'judge'])
            cache.commit.assert_called_once()
            outputs.reload.assert_called_once()
            outputs.commit.assert_called_once()


if __name__ == '__main__':
    unittest.main()
