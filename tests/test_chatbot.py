import json

import httpx
import pandas as pd
from openai import APIConnectionError, AuthenticationError, NotFoundError, RateLimitError

import fincoach.chatbot as chatbot_module
from fincoach.chatbot import (
    SYSTEM_PROMPT,
    FinancialChatbot,
    build_financial_context,
    translate_openai_error,
)


class FakeProvider:
    def __init__(self) -> None:
        self.system_prompt = ""
        self.user_prompt = ""

    def generate(self, *, system_prompt: str, user_prompt: str) -> str:
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        return "Grounded answer"


def sample_transactions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-02-01"]),
            "description": ["Salary ACME", "REWE receipt", "Restaurant receipt"],
            "amount": [2000.0, -400.0, -100.0],
            "category": ["Income", "Groceries", "Dining"],
        }
    )


def test_context_contains_calculated_summaries_not_raw_transactions() -> None:
    context = build_financial_context(sample_transactions(), budgets={"Dining": 75})
    serialized = context.to_json()
    payload = json.loads(serialized)

    assert payload["overall_summary"]["income"] == 2000.0
    assert payload["latest_spending_by_category"] == {"Dining": 100.0}
    assert payload["budget_analysis"][0]["status"] == "over_budget"
    assert "description" not in serialized
    assert "Salary ACME" not in serialized
    assert "REWE receipt" not in serialized


def test_chatbot_passes_context_and_safeguards_to_provider() -> None:
    provider = FakeProvider()
    chatbot = FinancialChatbot(provider)
    context = build_financial_context(sample_transactions())

    answer = chatbot.answer("What is my savings rate?", context)

    assert answer == "Grounded answer"
    assert "not a financial advisor" in provider.system_prompt
    assert "Never calculate" in provider.system_prompt
    assert "What is my savings rate?" in provider.user_prompt
    assert '"savings_rate": 75.0' in provider.user_prompt


def test_system_prompt_requires_unavailable_information_to_be_disclosed() -> None:
    assert "If information is unavailable" in SYSTEM_PROMPT
    assert "distinguish factual observations" in SYSTEM_PROMPT


def test_chatbot_loads_current_project_root_env_without_stale_placeholder(
    tmp_path, monkeypatch
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "OPENAI_API_KEY=test-key-from-root\nOPENAI_MODEL=test-model\n",
        encoding="utf-8",
    )
    captured: dict[str, str] = {}

    class FakeOpenAIProvider:
        def __init__(self, api_key: str, *, model: str) -> None:
            captured["api_key"] = api_key
            captured["model"] = model

        def generate(self, *, system_prompt: str, user_prompt: str) -> str:
            return "answer"

    monkeypatch.setattr(chatbot_module, "PROJECT_ENV_FILE", env_file)
    monkeypatch.setattr(chatbot_module, "OpenAIChatProvider", FakeOpenAIProvider)
    monkeypatch.setenv("OPENAI_API_KEY", "your-api-key-here")
    monkeypatch.delenv("OPENAI_MODEL", raising=False)

    chatbot = chatbot_module.chatbot_from_environment()

    assert chatbot is not None
    assert captured == {"api_key": "test-key-from-root", "model": "test-model"}


def _response(status_code: int) -> httpx.Response:
    request = httpx.Request("POST", "https://api.openai.com/v1/responses")
    return httpx.Response(status_code, request=request)


def test_authentication_error_has_secret_safe_diagnostics() -> None:
    error = AuthenticationError(
        "authentication failed",
        response=_response(401),
        body={"message": "Incorrect API key: sk-secret-value", "code": "invalid_api_key"},
    )

    result = translate_openai_error(error, model="test-model")

    assert result.category == "authentication"
    assert result.status_code == 401
    assert result.error_code == "invalid_api_key"
    assert "sk-secret-value" not in result.development_message()
    assert "[REDACTED]" in result.development_message()


def test_model_not_found_error_is_distinguished() -> None:
    error = NotFoundError(
        "not found",
        response=_response(404),
        body={"message": "The model does not exist", "code": "model_not_found"},
    )

    result = translate_openai_error(error, model="missing-model")

    assert result.category == "model_not_found"
    assert result.status_code == 404
    assert result.model == "missing-model"


def test_insufficient_quota_is_distinguished_from_rate_limit() -> None:
    error = RateLimitError(
        "quota exceeded",
        response=_response(429),
        body={"message": "You exceeded your quota", "code": "insufficient_quota"},
    )

    result = translate_openai_error(error, model="test-model")

    assert result.category == "insufficient_quota"
    assert result.status_code == 429


def test_connection_error_is_distinguished() -> None:
    request = httpx.Request("POST", "https://api.openai.com/v1/responses")
    result = translate_openai_error(APIConnectionError(request=request), model="test-model")

    assert result.category == "connection"
    assert result.status_code is None
