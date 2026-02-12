"""
LLM Provider Abstraction Layer

Provides a unified interface for multiple LLM providers:
- Vertex AI (GCP authentication)
- Google AI (API key)
- OpenAI (API key)
- Anthropic (API key)
"""

import os
import json
from abc import ABC, abstractmethod
from typing import Dict, Optional, List
from dotenv import load_dotenv

load_dotenv()


# ============================================================================
# Provider Configuration
# ============================================================================

PROVIDER_CONFIG = {
    "vertex_ai": {
        "name": "Vertex AI (GCP)",
        "requires_api_key": False,
        "models": ["gemini-2.5-pro", "gemini-2.0-flash-001"],
        "default_model": "gemini-2.5-pro",
    },
    "google_ai": {
        "name": "Google AI (API Key)",
        "requires_api_key": True,
        "models": ["gemini-2.0-flash", "gemini-1.5-pro", "gemini-1.5-flash"],
        "default_model": "gemini-2.0-flash",
    },
    "openai": {
        "name": "OpenAI",
        "requires_api_key": True,
        "models": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"],
        "default_model": "gpt-4o",
    },
    "anthropic": {
        "name": "Anthropic",
        "requires_api_key": True,
        "models": ["claude-3-5-sonnet-latest", "claude-3-5-haiku-latest"],
        "default_model": "claude-3-5-sonnet-latest",
    },
}


def get_provider_choices() -> List[str]:
    """Return list of provider display names for UI."""
    return [config["name"] for config in PROVIDER_CONFIG.values()]


def get_provider_key(display_name: str) -> str:
    """Convert display name back to provider key."""
    for key, config in PROVIDER_CONFIG.items():
        if config["name"] == display_name:
            return key
    return "vertex_ai"


def get_models_for_provider(provider_key: str) -> List[str]:
    """Return available models for a provider."""
    return PROVIDER_CONFIG.get(provider_key, {}).get("models", [])


def get_default_model(provider_key: str) -> str:
    """Return default model for a provider."""
    return PROVIDER_CONFIG.get(provider_key, {}).get("default_model", "")


# ============================================================================
# Abstract Base Class
# ============================================================================

class LLMProvider(ABC):
    """Abstract base class for all LLM provider implementations.

    Subclasses must implement :meth:`generate_json` and
    :meth:`generate_text`.  Shared helpers for cleaning and repairing
    JSON responses are provided by the base class.

    Args:
        model: Model identifier string (e.g. ``"gemini-2.5-pro"``).
    """

    def __init__(self, model: str):
        self.model_name = model
    
    @abstractmethod
    def generate_json(self, prompt: str, max_tokens: int = 8192) -> Dict:
        """
        Generate a JSON response from the LLM.
        
        Args:
            prompt: The prompt to send to the LLM
            max_tokens: Maximum output tokens
            
        Returns:
            Parsed JSON dict from the model response
        """
        pass
    
    @abstractmethod
    def generate_text(self, prompt: str, max_tokens: int = 8192) -> str:
        """
        Generate a text response from the LLM.
        
        Args:
            prompt: The prompt to send to the LLM
            max_tokens: Maximum output tokens
            
        Returns:
            Text response from the model
        """
        pass
    
    def _clean_json_response(self, text: str) -> str:
        """Remove markdown code blocks from JSON response."""
        text = text.strip()
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        return text.strip()
    
    def _repair_json(self, json_str: str) -> str:
        """Attempt to repair truncated JSON by closing open brackets."""
        json_str = json_str.replace(':- "', ': "')
        
        stack = []
        escaped = False
        in_string = False
        
        for char in json_str:
            if char == '\\':
                escaped = not escaped
                continue
            
            if char == '"' and not escaped:
                in_string = not in_string
            
            if not in_string:
                if char == '{':
                    stack.append('}')
                elif char == '[':
                    stack.append(']')
                elif char in '}]' and stack and stack[-1] == char:
                    stack.pop()
            
            escaped = False
        
        if in_string:
            json_str += '"'
        
        while stack:
            json_str += stack.pop()
        
        return json_str


# ============================================================================
# Vertex AI Provider (GCP Authentication)
# ============================================================================

class VertexAIProvider(LLMProvider):
    """Google Vertex AI provider using GCP application-default credentials.

    Requires ``GOOGLE_CLOUD_PROJECT`` (and optionally
    ``GOOGLE_CLOUD_LOCATION``) environment variables.

    Args:
        model: Vertex AI model name.
    """

    def __init__(self, model: str = "gemini-2.5-pro"):
        super().__init__(model)
        
        import vertexai
        from vertexai.generative_models import GenerativeModel
        
        project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
        location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
        
        if not project_id:
            raise ValueError("GOOGLE_CLOUD_PROJECT not found in environment variables")
        
        vertexai.init(project=project_id, location=location)
        
        # Suppress deprecation warnings
        import warnings
        warnings.filterwarnings("ignore", category=UserWarning, module="vertexai.generative_models")
        
        self.model = GenerativeModel(model)
    
    def generate_json(self, prompt: str, max_tokens: int = 8192) -> Dict:
        response = self.model.generate_content(
            prompt,
            generation_config={
                "response_mime_type": "application/json",
                "max_output_tokens": max_tokens
            }
        )
        
        text = self._clean_json_response(response.text)
        
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            repaired = self._repair_json(text)
            try:
                return json.loads(repaired)
            except json.JSONDecodeError:
                print(f"Failed to parse JSON response")
                return {}
    
    def generate_text(self, prompt: str, max_tokens: int = 8192) -> str:
        response = self.model.generate_content(
            prompt,
            generation_config={"max_output_tokens": max_tokens}
        )
        return response.text


# ============================================================================
# Google AI Provider (API Key)
# ============================================================================

class GoogleAIProvider(LLMProvider):
    """Google AI provider using an API key via the ``google-generativeai`` SDK.

    Args:
        api_key: Google AI API key.
        model: Model name.
    """

    def __init__(self, api_key: str, model: str = "gemini-2.0-flash"):
        super().__init__(model)
        
        import google.generativeai as genai
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model)
    
    def generate_json(self, prompt: str, max_tokens: int = 8192) -> Dict:
        import google.generativeai as genai
        
        response = self.model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                max_output_tokens=max_tokens
            )
        )
        
        text = self._clean_json_response(response.text)
        
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            repaired = self._repair_json(text)
            try:
                return json.loads(repaired)
            except json.JSONDecodeError:
                print(f"Failed to parse JSON response")
                return {}
    
    def generate_text(self, prompt: str, max_tokens: int = 8192) -> str:
        import google.generativeai as genai
        
        response = self.model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(max_output_tokens=max_tokens)
        )
        return response.text


# ============================================================================
# OpenAI Provider
# ============================================================================

class OpenAIProvider(LLMProvider):
    """OpenAI provider (GPT-4o and related models).

    Args:
        api_key: OpenAI API key.
        model: Model name.
    """

    def __init__(self, api_key: str, model: str = "gpt-4o"):
        super().__init__(model)
        
        from openai import OpenAI
        
        self.client = OpenAI(api_key=api_key)
    
    def generate_json(self, prompt: str, max_tokens: int = 8192) -> Dict:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            max_tokens=max_tokens
        )
        
        text = response.choices[0].message.content
        text = self._clean_json_response(text)
        
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            repaired = self._repair_json(text)
            try:
                return json.loads(repaired)
            except json.JSONDecodeError:
                print(f"Failed to parse JSON response")
                return {}
    
    def generate_text(self, prompt: str, max_tokens: int = 8192) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens
        )
        return response.choices[0].message.content


# ============================================================================
# Anthropic Provider
# ============================================================================

class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider.

    JSON mode is enforced via prompt instruction since the Anthropic API
    does not offer a native JSON response format.

    Args:
        api_key: Anthropic API key.
        model: Model name.
    """

    def __init__(self, api_key: str, model: str = "claude-3-5-sonnet-latest"):
        super().__init__(model)
        
        from anthropic import Anthropic
        
        self.client = Anthropic(api_key=api_key)
    
    def generate_json(self, prompt: str, max_tokens: int = 8192) -> Dict:
        # Anthropic doesn't have native JSON mode, so we enforce it via prompt
        json_prompt = f"""{prompt}

IMPORTANT: You must respond with ONLY valid JSON. No markdown, no explanation, just the JSON object."""
        
        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": json_prompt}]
        )
        
        text = response.content[0].text
        text = self._clean_json_response(text)
        
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            repaired = self._repair_json(text)
            try:
                return json.loads(repaired)
            except json.JSONDecodeError:
                print(f"Failed to parse JSON response")
                return {}
    
    def generate_text(self, prompt: str, max_tokens: int = 8192) -> str:
        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text


# ============================================================================
# Factory Function
# ============================================================================

def create_provider(
    provider_key: str,
    api_key: Optional[str] = None,
    model: Optional[str] = None
) -> LLMProvider:
    """
    Factory function to create an LLM provider instance.
    
    Args:
        provider_key: One of 'vertex_ai', 'google_ai', 'openai', 'anthropic'
        api_key: API key (required for all except vertex_ai)
        model: Model name (uses default if not specified)
    
    Returns:
        LLMProvider instance
    """
    config = PROVIDER_CONFIG.get(provider_key)
    if not config:
        raise ValueError(f"Unknown provider: {provider_key}")
    
    if config["requires_api_key"] and not api_key:
        raise ValueError(f"{config['name']} requires an API key")
    
    model = model or config["default_model"]
    
    if provider_key == "vertex_ai":
        return VertexAIProvider(model=model)
    elif provider_key == "google_ai":
        return GoogleAIProvider(api_key=api_key, model=model)
    elif provider_key == "openai":
        return OpenAIProvider(api_key=api_key, model=model)
    elif provider_key == "anthropic":
        return AnthropicProvider(api_key=api_key, model=model)
    else:
        raise ValueError(f"Unknown provider: {provider_key}")
