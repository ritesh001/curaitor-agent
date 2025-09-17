# from crawler.pipeline import run_crawler
# from agent.planner import AgentPipeline

if __name__ == '__main__':
    # Step 1: Crawl
    run_crawler()
    # Step 2: Agent pipeline
    pipeline = AgentPipeline()
    pipeline.run()
    print("Extraction complete. Results in data/output.json")