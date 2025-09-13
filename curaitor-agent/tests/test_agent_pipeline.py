from agent.planner import AgentPipeline
import json
import os

def test_agent_pipeline():
    # Setup
    config_path = 'config.yaml'
    crawled_items_path = 'tests/test_crawled.json'
    output_path = 'tests/test_output.json'
    
    # Create a sample crawled items JSON for testing
    sample_items = [
        {
            "file_path": "data/raw/sample1.pdf",
            "title": "Sample Paper 1",
            "authors": ["Author A", "Author B"]
        },
        {
            "file_path": "data/raw/sample2.pdf",
            "title": "Sample Paper 2",
            "authors": ["Author C"]
        }
    ]
    
    with open(crawled_items_path, 'w', encoding='utf-8') as f:
        json.dump(sample_items, f)

    # Initialize the AgentPipeline
    pipeline = AgentPipeline(config_path=config_path)

    # Run the pipeline
    pipeline.run(crawled_items_path=crawled_items_path, output_path=output_path)

    # Verify output
    assert os.path.exists(output_path), "Output file was not created."
    
    with open(output_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    assert isinstance(results, list), "Output should be a list."
    assert len(results) == len(sample_items), "Number of results should match number of input items."

    # Clean up
    os.remove(crawled_items_path)
    os.remove(output_path)

if __name__ == "__main__":
    test_agent_pipeline()