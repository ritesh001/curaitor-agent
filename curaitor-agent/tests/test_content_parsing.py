import unittest
from rag.content_parsing import extract_pdf_components
from pathlib import Path

class TestContentParsing(unittest.TestCase):

    def setUp(self):
        # Setup any necessary paths or variables
        self.test_pdf_path = Path("data/raw/test_document.pdf")  # Adjust the path as needed

    def test_extract_pdf_components(self):
        # Test the extraction of components from a PDF
        result = extract_pdf_components(self.test_pdf_path)
        
        # Check if the result contains the expected keys
        self.assertIn('texts', result)
        self.assertIn('tables', result)
        self.assertIn('images', result)
        self.assertIn('doc', result)

        # Check if texts is a list
        self.assertIsInstance(result['texts'], list)

        # Check if tables is a list
        self.assertIsInstance(result['tables'], list)

        # Check if images is a list
        self.assertIsInstance(result['images'], list)

    def test_invalid_pdf_path(self):
        # Test handling of an invalid PDF path
        with self.assertRaises(FileNotFoundError):
            extract_pdf_components("invalid/path/to/pdf.pdf")

if __name__ == '__main__':
    unittest.main()