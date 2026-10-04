"""LLM Service - Abstraction layer for GPT/Gemini/Local LLM."""
import json
import logging
from typing import Optional, Dict, Any
from app.core.config import settings

logger = logging.getLogger(__name__)


class LLMService:
    """Configurable LLM abstraction supporting OpenAI, Gemini, and local LLMs."""

    def __init__(self):
        self.provider = settings.LLM_PROVIDER

    async def generate(self, prompt: str, system_prompt: Optional[str] = None,
                       temperature: float = 0.3, max_tokens: int = 2000) -> str:
        """Generate text using configured LLM provider."""
        try:
            if self.provider == "openai" and settings.OPENAI_API_KEY and not settings.OPENAI_API_KEY.startswith("your-"):
                return await self._generate_openai(prompt, system_prompt, temperature, max_tokens)
            elif self.provider == "gemini" and settings.GEMINI_API_KEY and not settings.GEMINI_API_KEY.startswith("your-"):
                return await self._generate_gemini(prompt, system_prompt, temperature, max_tokens)
            elif self.provider == "local" and settings.LOCAL_LLM_URL:
                return await self._generate_local(prompt, system_prompt, temperature, max_tokens)
            else:
                # Fallback: intelligent rule-based response
                return self._fallback_generate(prompt)
        except Exception as e:
            logger.warning(f"LLM generation failed ({self.provider}): {e}. Using fallback.")
            return self._fallback_generate(prompt)

    async def generate_json(self, prompt: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """Generate JSON response from LLM."""
        full_prompt = f"{prompt}\n\nRespond ONLY with valid JSON, no markdown formatting."
        response = await self.generate(full_prompt, system_prompt)
        # Clean response
        response = response.strip()
        if response.startswith("```"):
            lines = response.split("\n")
            response = "\n".join(lines[1:-1])
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse LLM JSON response: {response[:200]}")
            return {}

    async def _generate_openai(self, prompt: str, system_prompt: Optional[str],
                                temperature: float, max_tokens: int) -> str:
        import openai
        client = openai.AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = await client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content

    async def _generate_gemini(self, prompt: str, system_prompt: Optional[str],
                                temperature: float, max_tokens: int) -> str:
        import google.generativeai as genai
        genai.configure(api_key=settings.GEMINI_API_KEY)
        model = genai.GenerativeModel(settings.GEMINI_MODEL)
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        response = await model.generate_content_async(
            full_prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            )
        )
        return response.text

    async def _generate_local(self, prompt: str, system_prompt: Optional[str],
                               temperature: float, max_tokens: int) -> str:
        import httpx
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.post(
                f"{settings.LOCAL_LLM_URL}/api/chat",
                json={
                    "model": settings.LOCAL_LLM_MODEL,
                    "messages": messages,
                    "stream": False,
                    "options": {"temperature": temperature, "num_predict": max_tokens}
                }
            )
            return response.json()["message"]["content"]

    def _fallback_generate(self, prompt: str) -> str:
        """Rule-based fallback when no LLM is available."""
        prompt_lower = prompt.lower()

        if "business goal" in prompt_lower or "understand" in prompt_lower:
            return json.dumps({
                "problem_type": "classification",
                "key_objectives": ["Predict target variable accurately"],
                "success_metrics": ["accuracy", "f1_score"],
                "constraints": [],
                "recommendations": ["Start with baseline models and iterate"]
            })

        if "recommend" in prompt_lower or "algorithm" in prompt_lower:
            return json.dumps({
                "recommendations": [
                    {"algorithm": "random_forest", "reason": "Good general-purpose model"},
                    {"algorithm": "xgboost", "reason": "Strong for tabular data"},
                    {"algorithm": "logistic_regression", "reason": "Fast and interpretable"}
                ]
            })

        if "preprocessing" in prompt_lower:
            return json.dumps({
                "steps": [
                    {"step": "impute_numeric", "strategy": "median"},
                    {"step": "impute_categorical", "strategy": "mode"},
                    {"step": "scale_numeric", "strategy": "standard"},
                    {"step": "encode_categorical", "strategy": "onehot"}
                ]
            })

        return json.dumps({"response": "Proceeding with default configuration."})


llm_service = LLMService()
