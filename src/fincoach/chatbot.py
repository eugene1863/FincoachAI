"""Provider-neutral conversational layer for calculated financial summaries.

This module never asks an LLM to calculate financial metrics. It first builds a
compact context using FinCoach's analytics and coaching functions, then allows a
language-model provider to explain those precomputed values.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import pandas as pd
from dotenv import dotenv_values, load_dotenv

from fincoach.analytics import calculate_financial_summary, monthly_summary, spending_by_category
from fincoach.coaching import (
    SavingsGoalAnalysis,
    analyze_budgets,
    analyze_spending_change,
    generate_recommendations,
)

SYSTEM_PROMPT = """You are FinCoach, an educational personal finance assistant.
You are not a financial advisor and must not claim to provide professional financial advice.

Rules:
- Use only the calculated financial context supplied by the application.
- Never calculate, estimate, infer, or invent transaction values yourself.
- Never claim that a value exists if it is absent from the context.
- If information is unavailable, say so clearly.
- Clearly distinguish factual observations from optional recommendations.
- Do not imply that you saw raw transactions; you receive summarized calculations only.
- Keep answers concise, specific, supportive, and grounded in the supplied currency values.
"""

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ENV_FILE = PROJECT_ROOT / ".env"
API_KEY_PLACEHOLDER = "your-api-key-here"


class ChatProviderError(RuntimeError):
    """Secret-safe, structured description of an LLM provider failure."""

    def __init__(
        self,
        *,
        category: str,
        exception_type: str,
        model: str,
        provider_message: str,
        user_message: str,
        status_code: int | None = None,
        error_code: str | None = None,
    ) -> None:
        super().__init__(user_message)
        self.category = category
        self.exception_type = exception_type
        self.model = model
        self.provider_message = provider_message
        self.user_message = user_message
        self.status_code = status_code
        self.error_code = error_code

    def development_message(self) -> str:
        """Return diagnostics that contain no API-key material."""

        status = str(self.status_code) if self.status_code is not None else "unavailable"
        code = self.error_code or "unavailable"
        return (
            "OpenAI request failed.\n\n"
            f"- Exception type: `{self.exception_type}`\n"
            f"- HTTP status: `{status}`\n"
            f"- Error code: `{code}`\n"
            f"- Model: `{self.model}`\n"
            f"- OpenAI message: {self.provider_message}"
        )


@dataclass(frozen=True, slots=True)
class FinancialContext:
    """Calculated, privacy-conscious information supplied to the chat provider."""

    covered_period: Mapping[str, str]
    overall_summary: Mapping[str, float | int | None]
    latest_month: Mapping[str, float | str | None]
    monthly_summaries: tuple[Mapping[str, float | str | None], ...]
    latest_spending_by_category: Mapping[str, float]
    budget_analysis: tuple[Mapping[str, float | str], ...]
    spending_change: Mapping[str, object] | None
    savings_goal: Mapping[str, float | int | bool] | None
    deterministic_recommendations: tuple[str, ...]

    def to_json(self) -> str:
        """Serialize the summary for a provider without transaction-level data."""

        return json.dumps(asdict(self), ensure_ascii=False, allow_nan=False, indent=2)


class ChatProvider(Protocol):
    """Replaceable interface implemented by an LLM provider adapter."""

    def generate(self, *, system_prompt: str, user_prompt: str) -> str:
        """Generate one natural-language answer."""


class OpenAIChatProvider:
    """OpenAI Responses API adapter behind the provider-neutral interface."""

    def __init__(self, api_key: str, *, model: str = "gpt-5.2") -> None:
        if not api_key.strip():
            raise ValueError("An OpenAI API key is required")

        # Imported lazily so context building and its tests do not require an API
        # connection or initialize an external client.
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key, timeout=30.0, max_retries=0)
        self._model = model

    def generate(self, *, system_prompt: str, user_prompt: str) -> str:
        """Request a concise explanation through the OpenAI Responses API."""

        try:
            response = self._client.responses.create(
                model=self._model,
                instructions=system_prompt,
                input=user_prompt,
                text={"verbosity": "low"},
            )
            return response.output_text.strip()
        except Exception as error:
            raise translate_openai_error(error, model=self._model) from error


class FinancialChatbot:
    """Coordinate calculated context and a replaceable language-model provider."""

    def __init__(self, provider: ChatProvider) -> None:
        self._provider = provider

    def answer(
        self,
        question: str,
        context: FinancialContext,
        history: Sequence[Mapping[str, str]] = (),
    ) -> str:
        """Explain calculated financial context in response to a user question."""

        clean_question = question.strip()
        if not clean_question:
            raise ValueError("Question cannot be blank")

        recent_history = [
            {"role": item.get("role", ""), "content": item.get("content", "")}
            for item in history[-6:]
            if item.get("role") in {"user", "assistant"}
        ]
        user_prompt = (
            "CALCULATED FINANCIAL CONTEXT (authoritative; do not recalculate):\n"
            f"{context.to_json()}\n\n"
            "RECENT CONVERSATION:\n"
            f"{json.dumps(recent_history, ensure_ascii=False)}\n\n"
            f"USER QUESTION:\n{clean_question}"
        )
        return self._provider.generate(system_prompt=SYSTEM_PROMPT, user_prompt=user_prompt)


def build_financial_context(
    transactions: pd.DataFrame,
    *,
    budgets: Mapping[str, float] | None = None,
    savings_goal: SavingsGoalAnalysis | None = None,
) -> FinancialContext:
    """Build a summary using existing analytics and coaching functions only."""

    if transactions.empty:
        raise ValueError("Financial context requires at least one transaction")

    dates = pd.to_datetime(transactions["date"], errors="raise")
    overall = calculate_financial_summary(transactions)
    monthly = monthly_summary(transactions)
    latest_month_name = str(monthly.iloc[-1]["month"])
    latest_mask = dates.dt.to_period("M").astype("string").eq(latest_month_name)
    latest_transactions = transactions.loc[latest_mask]
    latest_categories = spending_by_category(latest_transactions)
    latest_row = monthly.iloc[-1]

    budget_results = analyze_budgets(transactions, budgets or {})
    try:
        change = asdict(analyze_spending_change(transactions))
    except ValueError:
        change = None

    recommendations = generate_recommendations(
        transactions,
        budgets=budgets,
        savings_goal=savings_goal,
    )
    monthly_records = tuple(_json_safe_record(record) for record in monthly.to_dict("records"))

    return FinancialContext(
        covered_period={
            "start": dates.min().date().isoformat(),
            "end": dates.max().date().isoformat(),
        },
        overall_summary=asdict(overall),
        latest_month=_json_safe_record(latest_row.to_dict()),
        monthly_summaries=monthly_records,
        latest_spending_by_category={
            str(category): float(amount) for category, amount in latest_categories.items()
        },
        budget_analysis=tuple(asdict(result) for result in budget_results),
        spending_change=change,
        savings_goal=asdict(savings_goal) if savings_goal else None,
        deterministic_recommendations=tuple(recommendations),
    )


def chatbot_from_environment() -> FinancialChatbot | None:
    """Create the configured chatbot, or return ``None`` when no API key exists."""

    # Resolve the file explicitly instead of depending on Streamlit's working
    # directory. Reading the current file values also avoids stale configuration
    # when a long-running Streamlit process previously loaded the placeholder.
    load_dotenv(dotenv_path=PROJECT_ENV_FILE)
    file_config = dotenv_values(PROJECT_ENV_FILE)

    environment_key = os.getenv("OPENAI_API_KEY", "").strip()
    file_key = str(file_config.get("OPENAI_API_KEY") or "").strip()
    api_key = (
        environment_key
        if environment_key and environment_key != API_KEY_PLACEHOLDER
        else file_key
    )
    if not api_key or api_key == API_KEY_PLACEHOLDER:
        return None

    environment_model = os.getenv("OPENAI_MODEL", "").strip()
    file_model = str(file_config.get("OPENAI_MODEL") or "").strip()
    model = environment_model or file_model or "gpt-5.2"
    return FinancialChatbot(OpenAIChatProvider(api_key, model=model))


def is_development_mode() -> bool:
    """Return whether detailed, secret-safe diagnostics should be displayed."""

    load_dotenv(dotenv_path=PROJECT_ENV_FILE)
    file_config = dotenv_values(PROJECT_ENV_FILE)
    environment = os.getenv("FINCOACH_ENV", "").strip()
    configured = environment or str(file_config.get("FINCOACH_ENV") or "").strip()
    return configured.casefold() == "development"


def translate_openai_error(error: Exception, *, model: str) -> ChatProviderError:
    """Map OpenAI SDK exceptions to stable application-level diagnostics."""

    from openai import (
        APIConnectionError,
        APIStatusError,
        APITimeoutError,
        AuthenticationError,
        NotFoundError,
        RateLimitError,
    )

    status_code = getattr(error, "status_code", None)
    body = getattr(error, "body", None)
    details = body.get("error", body) if isinstance(body, dict) else {}
    provider_message = details.get("message") if isinstance(details, dict) else None
    error_code = details.get("code") if isinstance(details, dict) else None
    message = _redact_api_key_material(str(provider_message or error))
    normalized_code = str(error_code or "").casefold()
    normalized_message = message.casefold()

    if isinstance(error, AuthenticationError) or status_code == 401:
        category = "authentication"
        user_message = "OpenAI rejected the API key. Replace it with a valid project API key."
    elif isinstance(error, NotFoundError) or status_code == 404:
        category = "model_not_found" if "model" in normalized_message else "not_found"
        user_message = (
            f"The configured model `{model}` was not found or is unavailable to this project."
        )
    elif (
        "insufficient_quota" in normalized_code
        or "quota" in normalized_message
        or "billing" in normalized_message
    ):
        category = "insufficient_quota"
        user_message = "The OpenAI project has insufficient quota or requires billing setup."
    elif isinstance(error, RateLimitError) or status_code == 429:
        category = "rate_limit"
        user_message = "OpenAI rate-limited the request. Wait briefly and try again."
    elif isinstance(error, (APIConnectionError, APITimeoutError)):
        category = "connection"
        user_message = "FinCoach could not reach OpenAI. Check the network and try again."
    elif isinstance(error, APIStatusError):
        category = "api_error"
        user_message = "OpenAI returned an unexpected API error. Please try again later."
    else:
        category = "unexpected"
        user_message = "The conversational assistant encountered an unexpected error."

    return ChatProviderError(
        category=category,
        exception_type=type(error).__name__,
        model=model,
        provider_message=message,
        user_message=user_message,
        status_code=status_code,
        error_code=str(error_code) if error_code else None,
    )


def _redact_api_key_material(message: str) -> str:
    """Remove API-key-shaped strings from provider diagnostics."""

    return re.sub(r"sk-[A-Za-z0-9_-]+(?:\.{3}[A-Za-z0-9_-]+)?", "[REDACTED]", message)


def _json_safe_record(record: Mapping[str, object]) -> dict[str, float | str | None]:
    """Convert pandas/NumPy scalar values and NaN to strict JSON values."""

    safe: dict[str, float | str | None] = {}
    for key, value in record.items():
        if pd.isna(value):
            safe[str(key)] = None
        elif isinstance(value, str):
            safe[str(key)] = value
        else:
            safe[str(key)] = float(value)
    return safe
