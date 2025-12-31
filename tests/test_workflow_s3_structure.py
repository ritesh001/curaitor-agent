import os
from pathlib import Path

import yaml


WORKFLOW_PATH = Path('.github/workflows/curaitor-scheduled.yml')


def test_workflow_file_exists():
    assert WORKFLOW_PATH.exists(), "Scheduled workflow file is missing"


def test_permissions_include_oidc():
    data = yaml.safe_load(WORKFLOW_PATH.read_text())
    perms = data.get('permissions', {})
    assert perms.get('id-token') == 'write', "permissions.id-token must be 'write' for OIDC"
    assert perms.get('contents') == 'read', "permissions.contents should be 'read' for checkout"


def test_has_configure_aws_credentials_step():
    data = yaml.safe_load(WORKFLOW_PATH.read_text())
    steps = data.get('jobs', {}).get('langgraph', {}).get('steps', [])
    found = False
    for step in steps:
        if step.get('uses') == 'aws-actions/configure-aws-credentials@v4':
            found = True
            with_ = step.get('with', {})
            # Region should be us-east-2 per user input
            assert with_.get('aws-region') == 'us-east-2', "aws-region must be us-east-2"
            # role-to-assume is provided via secret
            assert 'role-to-assume' in with_, "role-to-assume must be set"
            break
    assert found, "aws-actions/configure-aws-credentials@v4 step not found"


def test_has_s3_upload_step_and_prefix():
    text = WORKFLOW_PATH.read_text()

    # Bucket correctness
    assert 'S3_BUCKET: curaitor-agent-dec2025' in text, "S3 bucket env must be curaitor-agent-dec2025"

    # Unique run prefix based on run_id
    assert 'RUN_PREFIX="curaitor/${{ github.run_id }}"' in text or \
           "RUN_PREFIX='curaitor/${{ github.run_id }}'" in text, \
           'RUN_PREFIX must include ${{ github.run_id }} for uniqueness'

    # Basic commands present
    for cmd in [
        'aws s3 cp data/curaitor.sqlite',
        'aws s3 cp arxiv_out_hits.npz',
        'aws s3 cp search_results.csv',
        'aws s3 cp logs/langgraph_run.txt',
        'aws s3 sync papers/',
    ]:
        assert cmd in text, f"Expected upload command missing: {cmd}"

