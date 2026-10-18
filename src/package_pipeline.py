"""Create a portable ZIP including the explicitly selected fitted artifact."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent.parent


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def package(kind, output, checkpoint_dir=None):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f'{output}: use a fresh archive name')
    selected = (ROOT / (checkpoint_dir or ('model_checkpoints/trial' if kind == 'trial'
                                         else 'model_checkpoints/final'))).resolve()
    if not selected.is_relative_to(ROOT / 'model_checkpoints') or selected == ROOT / 'model_checkpoints':
        raise ValueError('Checkpoint directory must be a subdirectory of model_checkpoints/')
    model = selected / 'model.pt'
    if not model.is_file():
        raise FileNotFoundError(f'{model}: run the documented training command first')
    training = selected / 'training_report.json'
    metadata = json.loads(training.read_text()) if training.is_file() else {}
    frozen = json.loads((ROOT / 'provenance/original_freeze.json').read_text())
    original_model = sha256(model) == frozen['model_sha256']
    if kind == 'final' and not original_model and metadata.get('artifact_kind') != 'jury_full_retrain':
        raise ValueError('A final archive requires the original champion or a full-budget final retrain')
    if kind == 'trial' and metadata.get('artifact_kind') != 'isolated_trial':
        raise ValueError('A trial archive requires an explicitly labeled trial checkpoint')
    members = []
    for path in sorted(ROOT.rglob('*')):
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT)
        if any(part in {'.venv', '__pycache__', '.git', 'wheelhouse'} for part in relative.parts):
            continue
        if path.suffix in {'.pyc', '.zip'} or path.name in {'.DS_Store', 'installed-environment.txt'}:
            continue
        if relative.parts[0] == 'model_checkpoints' and not path.is_relative_to(selected):
            continue
        if path.resolve() != output:
            members.append(path)
    hashes = {str(path.relative_to(ROOT)): sha256(path) for path in members}
    manifest = dict(kind=kind, original_frozen_model=original_model,
                    checkpoint_directory=str(selected.relative_to(ROOT)),
                    model_sha256=sha256(model), files=hashes)
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in members:
            archive.write(path, 'jury_pipeline/' + str(path.relative_to(ROOT)))
        archive.writestr('jury_pipeline/PACKAGE_MANIFEST.json', json.dumps(manifest, indent=2) + '\n')
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() is not None:
            raise ValueError('Archive CRC check failed')
        for relative, digest in hashes.items():
            if hashlib.sha256(archive.read('jury_pipeline/' + relative)).hexdigest() != digest:
                raise ValueError(f'Archive hash mismatch: {relative}')
    checksum = output.with_name(output.name + '.sha256')
    checksum.write_text(sha256(output) + '  ' + output.name + '\n')
    print(json.dumps(dict(archive=str(output), kind=kind, files=len(hashes), bytes=output.stat().st_size,
                          sha256=sha256(output), original_frozen_model=original_model)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind', choices=['trial', 'final'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoint-dir', help='Relative checkpoint subdirectory, if different from the default')
    args = parser.parse_args()
    package(args.kind, args.output, args.checkpoint_dir)
