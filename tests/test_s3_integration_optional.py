import os
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path

import pytest


def _have_aws_cli() -> bool:
    return shutil.which("aws") is not None


def _env_true(name: str) -> bool:
    return os.environ.get(name, "").lower() in {"1", "true", "yes", "on"}


@pytest.mark.integration
def test_optional_s3_upload_once():
    """
    Optional integration test. Skips unless all conditions are met:
    - RUN_S3_INTEGRATION_TEST is true
    - AWS CLI is available and credentials are configured (OIDC or keys)
    - S3_BUCKET is set (defaults to curaitor-agent-dec2025)

    This performs a small put/list/delete round-trip under the curaitor/ prefix
    to mirror the workflow behavior.
    """
    if not _env_true("RUN_S3_INTEGRATION_TEST"):
        pytest.skip("Set RUN_S3_INTEGRATION_TEST=1 to enable this test")

    if not _have_aws_cli():
        pytest.skip("AWS CLI not found in PATH; install or set up to run this test")

    bucket = os.environ.get("S3_BUCKET", "curaitor-agent-dec2025")
    region = os.environ.get("AWS_REGION", "us-east-2")

    # Check that caller identity works to fail fast on creds issues
    try:
        subprocess.run(["aws", "sts", "get-caller-identity", "--region", region], check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        pytest.skip(f"AWS credentials not configured for region {region}: {e}")

    # Prepare a small temp file
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "hello.txt"
        p.write_text("hello from ci test\n")

        run_prefix = f"curaitor/test-ci-{uuid.uuid4()}"
        key = f"{run_prefix}/hello.txt"
        uri = f"s3://{bucket}/{key}"

        # Upload
        subprocess.run(["aws", "s3", "cp", str(p), uri, "--region", region], check=True)

        # List/Head to confirm presence
        ls = subprocess.run(["aws", "s3", "ls", uri, "--region", region], check=True, capture_output=True, text=True)
        assert "hello.txt" in ls.stdout

        # Cleanup
        subprocess.run(["aws", "s3", "rm", uri, "--region", region], check=True)

