"""
API abstraction layer for the LLMMatrix pilot.

Thin wrapper supporting three providers: Anthropic, OpenAI, DeepSeek.
Includes cost tracking, rate limit handling, and a hard spending cap.
"""

import logging
import os
import time

import httpx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Per-token pricing (USD per 1M tokens, as of mid-2026)
# Update these if pricing changes.
# ---------------------------------------------------------------------------

PRICING = {
    # Anthropic
    "claude-sonnet-4-6": {"input": 3.00, "output": 15.00},
    # OpenAI
    "gpt-5-mini": {"input": 0.40, "output": 1.60},
    # DeepSeek
    "deepseek-chat": {"input": 0.27, "output": 1.10},
}

# Fallback pricing for unknown models
DEFAULT_PRICING = {"input": 5.00, "output": 15.00}


class CostCapExceeded(Exception):
    """Raised when the running cost total exceeds the configured cap."""
    pass


class ModelRunner:
    """
    Manages API calls to multiple LLM providers with cost tracking.

    Attributes:
        total_cost_usd: Running total of API spend.
        cost_cap_usd: Hard cap on total spend.
        call_log: List of per-call cost/latency records.
    """

    def __init__(self, cost_cap_usd: float = 30.0) -> None:
        """
        Initialize the runner. API keys are read from environment variables.

        Args:
            cost_cap_usd: Abort if total cost exceeds this amount.
        """
        self.cost_cap_usd = cost_cap_usd
        self.total_cost_usd = 0.0
        self.call_log: list[dict] = []

        # Validate API keys exist (don't fail yet — only fail when needed)
        self._api_keys = {
            "anthropic": os.environ.get("ANTHROPIC_API_KEY"),
            "openai": os.environ.get("OPENAI_API_KEY"),
            "deepseek": os.environ.get("DEEPSEEK_API_KEY"),
        }

    def _check_api_key(self, provider: str) -> str:
        """Get API key for provider, raising if not set."""
        key = self._api_keys.get(provider)
        if not key:
            env_var = f"{provider.upper()}_API_KEY"
            raise ValueError(
                f"API key for {provider} not found. Set {env_var} environment variable."
            )
        return key

    def _compute_cost(self, model: str, tokens_in: int, tokens_out: int) -> float:
        """Compute cost in USD for a single API call."""
        pricing = PRICING.get(model, DEFAULT_PRICING)
        cost = (tokens_in * pricing["input"] + tokens_out * pricing["output"]) / 1_000_000
        return cost

    def _call_anthropic(
        self,
        model: str,
        prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> dict:
        """Call the Anthropic Messages API."""
        import anthropic

        api_key = self._check_api_key("anthropic")
        client = anthropic.Anthropic(api_key=api_key)

        start = time.time()
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        latency = time.time() - start

        response_text = response.content[0].text
        tokens_in = response.usage.input_tokens
        tokens_out = response.usage.output_tokens

        return {
            "response_text": response_text,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "latency_s": latency,
        }

    def _call_openai(
        self,
        model: str,
        prompt: str,
        temperature: float,
        max_tokens: int,
        seed: int | None,
    ) -> dict:
        """Call the OpenAI Chat Completions API."""
        import openai

        api_key = self._check_api_key("openai")
        client = openai.OpenAI(api_key=api_key)

        kwargs = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if seed is not None:
            kwargs["seed"] = seed

        start = time.time()
        response = client.chat.completions.create(**kwargs)
        latency = time.time() - start

        response_text = response.choices[0].message.content
        tokens_in = response.usage.prompt_tokens
        tokens_out = response.usage.completion_tokens

        return {
            "response_text": response_text,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "latency_s": latency,
        }

    def _call_deepseek(
        self,
        model: str,
        prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> dict:
        """Call the DeepSeek API (OpenAI-compatible endpoint)."""
        import openai

        api_key = self._check_api_key("deepseek")
        client = openai.OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com",
        )

        start = time.time()
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        latency = time.time() - start

        response_text = response.choices[0].message.content
        tokens_in = response.usage.prompt_tokens
        tokens_out = response.usage.completion_tokens

        return {
            "response_text": response_text,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "latency_s": latency,
        }

    def run_model(
        self,
        provider: str,
        model: str,
        prompt: str,
        temperature: float = 0.7,
        seed: int | None = None,
        max_tokens: int = 4000,
        max_retries: int = 3,
    ) -> dict:
        """
        Send a prompt to a model and return the response with metadata.

        Implements exponential backoff on rate limit (429) errors.
        Checks the cost cap before and after each call.

        Args:
            provider: "anthropic", "openai", or "deepseek".
            model: Model identifier (e.g., "claude-sonnet-4-6").
            prompt: The full prompt string.
            temperature: Sampling temperature.
            seed: Random seed (OpenAI only; ignored for other providers).
            max_tokens: Maximum response tokens.
            max_retries: Maximum retry attempts on rate limit errors.

        Returns:
            Dict with keys: response_text, tokens_in, tokens_out,
            cost_usd, latency_s.

        Raises:
            CostCapExceeded: If total cost would exceed the cap.
        """
        # Pre-check cost cap
        if self.total_cost_usd >= self.cost_cap_usd:
            raise CostCapExceeded(
                f"Cost cap of ${self.cost_cap_usd:.2f} already reached "
                f"(spent ${self.total_cost_usd:.2f}). "
                f"Increase cost_cap_usd or confirm manually to continue."
            )

        last_error = None
        for attempt in range(max_retries + 1):
            try:
                if provider == "anthropic":
                    result = self._call_anthropic(model, prompt, temperature, max_tokens)
                elif provider == "openai":
                    result = self._call_openai(model, prompt, temperature, max_tokens, seed)
                elif provider == "deepseek":
                    result = self._call_deepseek(model, prompt, temperature, max_tokens)
                else:
                    raise ValueError(f"Unknown provider: {provider}")

                # Compute cost
                cost = self._compute_cost(model, result["tokens_in"], result["tokens_out"])
                result["cost_usd"] = cost
                self.total_cost_usd += cost

                # Log the call
                self.call_log.append({
                    "provider": provider,
                    "model": model,
                    "tokens_in": result["tokens_in"],
                    "tokens_out": result["tokens_out"],
                    "cost_usd": cost,
                    "latency_s": result["latency_s"],
                })

                logger.info(
                    f"API call: {provider}/{model} — "
                    f"{result['tokens_in']}in/{result['tokens_out']}out — "
                    f"${cost:.4f} — {result['latency_s']:.1f}s — "
                    f"total: ${self.total_cost_usd:.2f}"
                )

                # Post-check cost cap
                if self.total_cost_usd >= self.cost_cap_usd:
                    logger.warning(
                        f"Cost cap reached: ${self.total_cost_usd:.2f} >= "
                        f"${self.cost_cap_usd:.2f}"
                    )

                return result

            except Exception as exc:
                error_str = str(exc)
                is_rate_limit = "429" in error_str or "rate" in error_str.lower()

                if is_rate_limit and attempt < max_retries:
                    wait_time = 2 ** (attempt + 1)
                    logger.info(
                        f"Rate limited on {provider}/{model}, "
                        f"retry {attempt + 1}/{max_retries} in {wait_time}s"
                    )
                    time.sleep(wait_time)
                    last_error = exc
                    continue
                else:
                    raise

        # Should not reach here, but just in case
        raise last_error  # type: ignore
