"""Audit submission files and declared preparation status without running models."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def local_file(value):
    if not isinstance(value, str) or not value:
        return None
    path = (ROOT / value).resolve()
    return path if path.is_relative_to(ROOT) and path.is_file() else None


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def audit():
    status = json.loads((ROOT / 'delivery/submission_status.json').read_text())
    manifest_path = ROOT / 'submission/FREEZE_MANIFEST.json'
    manifest = json.loads(manifest_path.read_text())
    frozen = []
    for name, expected in manifest['sha256'].items():
        path = local_file(f'submission/{name}')
        frozen.append({'path': f'submission/{name}',
                       'passed': path is not None and sha256(path) == expected})
    failures = [item['path'] for item in frozen if not item['passed']]
    required = [
        'README.md', 'delivery/README.md', 'delivery/CHECKLIST.md',
        'delivery/RUN_FOR_JURY.md', 'delivery/CREDITS.md',
        'delivery/report/report.md', 'delivery/report/report.pdf',
        'delivery/video/SCRIPT.md', 'delivery/presentation/OUTLINE.md',
        'submission/requirements-catboost.txt', 'submission/requirements.txt',
    ]
    missing = [name for name in required if local_file(name) is None]
    pending = []
    members = status.get('team_members')
    if not (isinstance(members, list) and members
            and all(isinstance(name, str) and name.strip() for name in members)):
        pending.append('Team full names and optional roles: to be completed later at user request')
    for key, label in [
        ('final_bundle_manifest', 'Audited all-history final-model manifest'),
        ('clean_environment_audit', 'Clean-environment offline inference audit'),
        ('license_path', 'Team-selected repository license'),
        ('video_path', 'Recorded summary video'),
    ]:
        if local_file(status.get(key)) is None:
            pending.append(label)
    if not status.get('final_artifact_delivery'):
        pending.append('Verified jury access to the fitted final artifact')
    if status.get('report_reviewed') is not True:
        pending.append('Team review of final report and consistency with delivered model')
    duration = status.get('video_duration_seconds')
    if isinstance(duration, bool) or not isinstance(duration, (int, float)) or not 0 < duration <= 120:
        pending.append('Verified video duration between 0 and 120 seconds')
    url = status.get('swisstransfer_url')
    if not isinstance(url, str) or not url.startswith('https://www.swisstransfer.com/'):
        pending.append('Working SwissTransfer video link')
    for key, label in [
        ('final_publication_verified', 'Final authorized repository publication'),
        ('handoff_completed', 'Single team handoff to organizers'),
    ]:
        if status.get(key) is not True:
            pending.append(label)
    report = local_file(status.get('report_path'))
    receipt_path = local_file(status.get('report_build_receipt'))
    report_verified = False
    if receipt_path and report:
        receipt = json.loads(receipt_path.read_text())
        source = local_file(receipt.get('source'))
        report_verified = (
            receipt.get('pages') == 3 and receipt.get('visual_review') == 'passed'
            and receipt.get('output_sha256') == sha256(report)
            and source is not None and receipt.get('source_sha256') == sha256(source)
        )
    if not report_verified:
        pending.append('Current report page-count, source identity and visual-review receipt')
    return {
        'scope': 'Local file identity and declared checklist status; no installation, model execution, upload or URL-access test',
        'frozen_bundle': {'checked': len(frozen), 'passed': not failures, 'failures': failures},
        'missing_preparation_files': missing,
        'report_build_verified': report_verified,
        'pending_handoff_items': pending,
        'declared_checklist_complete': not (failures or missing or pending),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='Save the JSON packaging/status audit')
    args = parser.parse_args()
    result = audit()
    rendered = json.dumps(result, indent=2) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end='')
    return 0 if result['declared_checklist_complete'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
