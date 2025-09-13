from langchain import OpenAI, LLMChain, PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter

from dotenv import load_dotenv 
load_dotenv() 
import json
import os

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError:
    ChatGoogleGenerativeAI = None

try:
    from langchain_ollama import ChatOllama  
except ImportError:
    ChatOllama = None

try:
    from langchain_openai import ChatOpenAI
except ImportError:
    ChatOpenAI = None

class ExtractionAgent:
    def __init__(self, config):
        llm_cfg = config['llm']
        provider = llm_cfg.get('provider', 'gemini').lower()
        model = llm_cfg['model']
        temperature = llm_cfg.get('temperature', 0)

        if provider == 'ollama':
            if ChatOllama is None:
                raise ImportError("Install with: pip install langchain-ollama")
            self.llm = ChatOllama(
                model=model,
                temperature=temperature,
                base_url=llm_cfg.get('base_url', 'http://localhost:11434'),
            )
        elif provider == 'openrouter':
            if ChatOpenAI is None:
                raise ImportError("Install with: pip install langchain-openai")
            self.llm = ChatOpenAI(
                model=model,
                temperature=temperature,
                openai_api_base="https://openrouter.ai/api/v1",
                openai_api_key=llm_cfg.get('api_key') or os.getenv('OPENROUTER_API_KEY'),
            )
        else:
            if ChatGoogleGenerativeAI is None:
                raise ImportError("pip install langchain-google-genai")
            self.llm = ChatGoogleGenerativeAI(
                model=model,
                temperature=temperature
            )
            
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=config['llm']['chunk_size'])
        self.prompt = PromptTemplate(
            input_variables=['chunk','schema'],
            template="""
Extract the following schema fields from the text chunk:
{schema}

Text:
{chunk}

Return JSON only.
"""
        )
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)

    def extract(self, text, schema):
        chunks = self.splitter.split_text(text)
        results = []
        for chunk in chunks:
            raw = self.chain.run(chunk=chunk, schema=schema)
            raw = raw.strip()
            try:
                results.append(json.loads(raw))
            except Exception:
                results.append({"raw": raw})
        return results

    def query(self, user_query, extracted_text):
        response = self.llm(f"Answer the following question based on the provided text: {user_query}\n\nText: {extracted_text}")
        return response.strip()