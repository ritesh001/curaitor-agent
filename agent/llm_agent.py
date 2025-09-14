from langchain import OpenAI, LLMChain, PromptTemplate
# from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter

from dotenv import load_dotenv
load_dotenv() ## to read contents in .env file
import json
import os

# Conditional imports
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError:
    ChatGoogleGenerativeAI = None

try:
    from langchain_ollama import ChatOllama  # new integration
except ImportError:
    ChatOllama = None

try:
    from langchain_openai import ChatOpenAI
except ImportError:
    ChatOpenAI = None

class ExtractionAgent:
    def __init__(self, config):
        # self.llm = OpenAI(model_name=config['llm']['model'], 
        #                   temperature=config['llm']['temperature'])
        # self.llm = ChatGoogleGenerativeAI(model=config['llm']['model'], 
        #                                   temperature=config['llm']['temperature'])
        llm_cfg = config['llm']
        provider = llm_cfg.get('provider', 'gemini').lower()
        model = llm_cfg['model']
        temperature = llm_cfg.get('temperature', 0)

        if provider == 'ollama':
            if ChatOllama is None:
                raise ImportError("Install with: pip install langchain-ollama")
            # base_url default http://localhost:11434; override via llm.base_url if needed
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
                # headers={
                #     "HTTP-Referer": llm_cfg.get('site_url', 'https://yoursite.com'),
                #     "X-Title": llm_cfg.get('site_name', 'Your App Name')
                # }
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

    # def extract(self, text, schema):
    #     chunks = self.splitter.split_text(text)
    #     results = []
    #     for chunk in chunks:
    #         res = self.chain.run(chunk=chunk, schema=schema)
    #         results.append(res)
    #     # merge partial JSONs
    #     return results

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
