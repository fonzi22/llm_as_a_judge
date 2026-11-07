"""Run the existing text judge on Modal: modal run llm_eval/modal_eval.py.

Only the Modal client is needed locally; vLLM and model weights stay on Modal.
"""
import json
from pathlib import Path, PurePosixPath
import re
import sys

import modal

ROOT = Path(__file__).resolve().parent
# Remote paths must remain POSIX even when the client runs on Windows.
REMOTE_ROOT = PurePosixPath('/opt/llm_eval')
CONFIG_NAME = 'qwen_matching_modal.json'
OUTPUT_VOLUME = 'hrvs-llm-eval-results'
DEFAULT_RUN = 'qwen3_8b_matching'


def ignore_prediction_file(path):
    # Do not upload scripts, images, bytecode or unrelated files in predictions/.
    return Path(path).suffix.lower() not in ('.json', '.jsonl')


image = (
    modal.Image.debian_slim(python_version='3.11')
    .pip_install('vllm==0.11.0', 'transformers==4.57.1', 'huggingface_hub>=0.34,<1')
    .env({
        'HF_HOME': '/hf-cache',
        'VLLM_WORKER_MULTIPROC_METHOD': 'spawn',
        'HF_HUB_DISABLE_TELEMETRY': '1',
        'VLLM_NO_USAGE_STATS': '1',
    })
    .add_local_file(ROOT / 'eval_matching.py', str(REMOTE_ROOT / 'eval_matching.py'))
    .add_local_file(ROOT / 'annotations.json', str(REMOTE_ROOT / 'annotations.json'))
    .add_local_file(ROOT / 'configs' / CONFIG_NAME, str(REMOTE_ROOT / 'configs' / CONFIG_NAME))
    .add_local_dir(ROOT / 'predictions', str(REMOTE_ROOT / 'predictions'),
                   ignore=ignore_prediction_file)
)
cache_volume = modal.Volume.from_name('hrvs-llm-eval-hf-cache', create_if_missing=True)
output_volume = modal.Volume.from_name(OUTPUT_VOLUME, create_if_missing=True)
app = modal.App('hrvs-llm-eval')


def validate_run_name(run_name):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,99}', run_name):
        raise ValueError('run-name must be 1-100 letters, digits, dots, dashes or underscores; '
                         'start with a letter or digit')
    return run_name


def prediction_relative_path(value):
    """Accept cwd paths or paths relative to llm_eval; keep remote paths portable."""
    path = Path(value).expanduser()
    if not path.exists():
        path = ROOT / path
    path = path.resolve()
    if not path.exists():
        raise ValueError(f'Prediction path not found: {value}')
    try:
        return path.relative_to(ROOT / 'predictions').as_posix()
    except ValueError as exc:
        raise ValueError('predictions must be inside llm_eval/predictions/') from exc


def evaluation_args(root, prediction, output, allow_partial=False, include_tests=False):
    args = ['--config', str(root / 'configs' / CONFIG_NAME),
            '--annotations', str(root / 'annotations.json'),
            '--predictions', str(root / 'predictions' / prediction),
            '--output-dir', str(output)]
    if allow_partial:
        args.append('--allow-partial')
    if include_tests:
        args.append('--include-tests')
    return args


@app.function(image=image, gpu=['L4', 'A10'], cpu=4, memory=32768, timeout=24 * 60 * 60,
              volumes={'/hf-cache': cache_volume, '/outputs': output_volume},
              max_containers=1)
def run_evaluation(prediction: str, run_name: str, allow_partial: bool, include_tests: bool):
    from huggingface_hub import snapshot_download

    sys.path.insert(0, str(REMOTE_ROOT))
    import eval_matching

    validate_run_name(run_name)
    output_volume.reload()
    cache_volume.reload()
    args = evaluation_args(REMOTE_ROOT, prediction, Path('/outputs') / run_name,
                           allow_partial, include_tests)
    # Reject invalid input before downloading weights or starting the engine.
    eval_matching.main(args + ['--dry-run'])
    config = json.loads(Path(REMOTE_ROOT / 'configs' / CONFIG_NAME).read_text(encoding='utf-8'))
    # LocalJudge intentionally never downloads. Populate its persistent cache first.
    snapshot_download(config['model'], cache_dir=config['cache_dir'])
    cache_volume.commit()
    try:
        eval_matching.main(args)
    except SystemExit as exc:
        # The evaluator's CLI exits for validation/model/judge errors. Return the
        # status so the local entrypoint can download the saved diagnostics too.
        return int(exc.code or 0)
    finally:
        output_volume.commit()
    return 0


def download_results(run_name, destination):
    destination = Path(destination).expanduser().resolve()
    count = 0
    for entry in output_volume.iterdir('/' + run_name, recursive=True):
        if entry.type != modal.volume.FileEntryType.FILE:
            continue
        relative = PurePosixPath(entry.path.lstrip('/')).relative_to(run_name)
        target = destination.joinpath(*relative.parts)
        if not target.resolve().is_relative_to(destination):
            raise ValueError(f'Invalid result path: {entry.path}')
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + '.download')
        with temporary.open('wb') as stream:
            for chunk in output_volume.read_file(entry.path):
                stream.write(chunk)
        temporary.replace(target)
        count += 1
    print(f'Downloaded {count} report files to {destination}', flush=True)


@app.local_entrypoint()
def main(predictions: str = 'predictions', run_name: str = DEFAULT_RUN,
         output_dir: str = '', allow_partial: bool = False, include_tests: bool = False,
         dry_run: bool = False, download_only: bool = False):
    """Validate locally, run on one L4 or A10, then download reports. Reuse run-name to resume."""
    validate_run_name(run_name)
    destination = Path(output_dir) if output_dir else ROOT / 'results' / 'modal' / run_name
    if download_only:
        download_results(run_name, destination)
        return

    sys.path.insert(0, str(ROOT))
    import eval_matching

    prediction = prediction_relative_path(predictions)
    args = evaluation_args(ROOT, prediction, destination, allow_partial, include_tests)
    eval_matching.main(args + ['--dry-run'])
    if dry_run:
        return
    status = run_evaluation.remote(prediction, run_name, allow_partial, include_tests)
    download_results(run_name, destination)
    if status:
        raise RuntimeError(f'Judge failed (exit {status}); downloaded available checkpoints. '
                           'See Modal logs; rerun the same command to resume.')
