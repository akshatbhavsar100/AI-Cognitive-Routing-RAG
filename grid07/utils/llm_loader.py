"""
utils/llm_loader.py
-------------------
Reads LLM_PROVIDER and LLM_MODEL from the environment and returns a
LangChain chat model object.  All three phases import from here so
swapping the provider only requires changing .env.
"""

import os
from dotenv import load_dotenv

load_dotenv()

def get_llm(temperature: float = 0.7):
    """
    Returns a LangChain ChatModel based on LLM_PROVIDER env variable.
    Supported providers: groq | openai | ollama
    """
    provider = os.getenv("LLM_PROVIDER", "groq").lower()
    model    = os.getenv("LLM_MODEL", "llama3-8b-8192")

    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model_name=model,
            temperature=temperature,
        )

    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            model=model,
            temperature=temperature,
        )

    elif provider == "ollama":
        from langchain_community.chat_models import ChatOllama
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        return ChatOllama(model=model, base_url=base_url, temperature=temperature)

    else:
        raise ValueError(
            f"Unknown LLM_PROVIDER='{provider}'. "
            "Choose from: groq | openai | ollama"
        )
