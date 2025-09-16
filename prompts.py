@mcp.prompt()
def generate_search_prompt(topic: str, num_papers: int = 5) -> str:
    """Prompt for the Retrieval Agent: find and extract structured information on academic papers."""
    return f"""Search for {num_papers} academic papers about '{topic}' using the search_papers tool. Follow these instructions:
    1. Search for papers using search_papers(topic='{topic}', keywords='{keywords}', max_results={num_papers})
    2. For each paper found, extract and organize the following:
       - Title
       - Authors
       - Publication date
       - Brief summary of key findings
       - Main contributions or innovations
       - Methodologies used
       - Relevance score to '{topic}'
    3. Return results in structured JSON for downstream analysis.
    """


@mcp.prompt()
def generate_analysis_prompt(topic: str, retrieved_docs: str) -> str:
    """Prompt for the Analysis Agent: synthesize insights across retrieved literature."""
    return f"""You are the Analysis Agent. Analyze the following retrieved documents (JSON format):  
{retrieved_docs}  

Please provide:
1. An **executive summary** in bullet points.
2. A **thematic synthesis** (group findings into categories/themes).
3. Identified **gaps and future research opportunities**.
4. A **short conclusion** linking results back to the topic '{topic}'.
Always cite papers inline (author, year). Keep the writing clear and well-structured.
"""

@mcp.prompt()
def generate_report_prompt(topic: str, analysis_output: str, recipient: str) -> str:
    """Prompt for the Report Agent: format analysis into a professional email report."""
    return f"""You are the Report Agent. Convert the following analysis into a polished report email.  

Details:
- Recipient: {recipient}
- Subject: "Literature Review on {topic}"
- Body content: {analysis_output}

Formatting instructions:
- Use clear sections: Introduction, Key Findings, Thematic Synthesis, Gaps/Future Directions, Conclusion.
- Maintain academic but accessible tone.
- Include inline citations (author, year).
- End with a properly formatted reference list.
"""


