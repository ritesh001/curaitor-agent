# Curaitor Agent

Curaitor Agent is a project designed to automate the process of downloading research papers from arXiv, extracting relevant information from them, and allowing users to query this information using a language model.

## Project Structure

The project is organized into several directories and files:

- **main.py**: Entry point of the application that orchestrates crawling, extraction, and querying processes.
- **config.yaml**: Configuration settings for the application, including API keys and model parameters.
- **requirements.txt**: Lists the required Python packages for the project.
- **pyproject.toml**: Used for packaging and dependency management.
- **.env.example**: Template for environment variables needed by the application.
- **.gitignore**: Specifies files and directories to be ignored by Git.

### Directories

- **agent/**: Contains the logic for interacting with the language model.
  - **llm_agent.py**: Defines the `ExtractionAgent` class for extracting information based on user queries.
  - **planner.py**: Contains the `AgentPipeline` class for managing the workflow of loading crawled items and extracting text.

- **rag/**: Responsible for extracting content from PDF documents.
  - **content_parsing.py**: Functions for extracting text, tables, and images from PDFs.

- **crawler/**: Manages the crawling process.
  - **pipeline.py**: Defines the `run_crawler` function.
  - **arxiv_downloader.py**: Functions for downloading PDFs from arXiv.

- **extractor/**: Handles the processing of extracted content.
  - **web_parser.py**: Functions for parsing web content.
  - **pdf_processor.py**: Processes PDF files.
  - **index_builder.py**: Builds an index for the extracted content.

- **scripts/**: Contains scripts for various tasks.
  - **download_arxiv.py**: Downloads PDFs from arXiv.
  - **extract_and_index.py**: Extracts text from PDFs and indexes the content.
  - **ask.py**: Allows users to query the AI agent.

- **data/**: Stores data files.
  - **raw/**: Stores raw data files, such as downloaded PDFs.
  - **processed/**: Stores processed data files, such as extracted text.
  - **index/**: Stores index files for quick access to extracted content.
  - **output/**: Stores output files, such as results from the AI agent.

- **tests/**: Contains unit tests for the project.
  - **test_arxiv_downloader.py**: Unit tests for the arxiv_downloader module.
  - **test_content_parsing.py**: Unit tests for the content_parsing module.
  - **test_agent_pipeline.py**: Unit tests for the agent pipeline functionality.

## Installation

To set up the project, clone the repository and install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

1. **Download PDFs**: Run the `download_arxiv.py` script to download PDFs from arXiv.
2. **Extract and Index**: Use the `extract_and_index.py` script to extract text from the downloaded PDFs and index the content.
3. **Query the AI Agent**: Use the `ask.py` script to interact with the AI agent and query the extracted information.

## Contribution

Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.