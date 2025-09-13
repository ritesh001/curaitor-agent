from crawler.arxiv_downloader import download_arxiv_pdfs
import os
import json
import pytest

@pytest.fixture
def setup_test_environment(tmp_path):
    # Create a temporary directory for testing
    test_dir = tmp_path / "arxiv_pdfs"
    test_dir.mkdir()
    return test_dir

def test_download_arxiv_pdfs(setup_test_environment):
    # Define the test parameters
    arxiv_ids = ["2101.00001", "2101.00002"]  # Example arXiv IDs
    output_dir = setup_test_environment

    # Call the function to download PDFs
    download_arxiv_pdfs(arxiv_ids, output_dir)

    # Check if the PDFs are downloaded
    for arxiv_id in arxiv_ids:
        pdf_path = output_dir / f"{arxiv_id}.pdf"
        assert pdf_path.exists(), f"PDF for {arxiv_id} was not downloaded."

    # Clean up the downloaded files
    for arxiv_id in arxiv_ids:
        pdf_path = output_dir / f"{arxiv_id}.pdf"
        if pdf_path.exists():
            os.remove(pdf_path)