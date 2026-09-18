"""Centralized LLM configuration for Gemini and OpenAI-compatible gateways."""
import os
from functools import lru_cache

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()


def text_from_response(response) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                value = item.get("text") or item.get("content") or item.get("value")
                if value:
                    parts.append(str(value))
            else:
                value = getattr(item, "text", None) or getattr(item, "content", None)
                if value:
                    parts.append(str(value))
        return "\n".join(parts)
    return str(content)


@lru_cache(maxsize=8)
def get_llm(provider: str | None = None, model: str | None = None):
    provider = (provider or os.getenv("LLM_PROVIDER", "gemini")).lower().strip()
    if provider in {"proxyllm", "openai", "open_ai"}:
        from langchain_openai import ChatOpenAI
        api_key = os.getenv("PROXYLLM_API_KEY")
        base_url = os.getenv("PROXYLLM_BASE_URL")
        if not api_key or not base_url:
            raise ValueError("Thiếu PROXYLLM_API_KEY hoặc PROXYLLM_BASE_URL trong .env")
        return ChatOpenAI(model=model or os.getenv("PROXYLLM_MODEL", "gpt-4o-mini"),
                          api_key=api_key, base_url=base_url, temperature=0.2,
                          timeout=120, max_retries=2)
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Thiếu GEMINI_API_KEY trong .env")
    return ChatGoogleGenerativeAI(model=model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
                                  google_api_key=api_key, temperature=0.2,
                                  timeout=120, max_retries=2)
