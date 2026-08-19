import pandas as pd
import pytest

from fincoach.analytics import (
    calculate_financial_summary,
    calculate_net_savings,
    calculate_savings_rate,
    calculate_total_expenses,
    calculate_total_income,
    category_summary,
    monthly_summary,
    spending_by_category,
)


def sample_transactions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-02-01"]),
            "amount": [2000.0, -500.0, -250.0],
            "transaction_type": ["income", "expense", "expense"],
            "category": ["Income", "Housing", "Food"],
        }
    )


def test_calculates_headline_metrics() -> None:
    summary = calculate_financial_summary(sample_transactions())

    assert summary.income == 2000.0
    assert summary.expenses == 750.0
    assert summary.net_cash_flow == 1250.0
    assert summary.savings_rate == pytest.approx(62.5)
    assert summary.transaction_count == 3


def test_phase_one_totals() -> None:
    transactions = sample_transactions()

    assert calculate_total_income(transactions) == 2000.0
    assert calculate_total_expenses(transactions) == 750.0
    assert calculate_net_savings(transactions) == 1250.0
    assert calculate_savings_rate(transactions) == pytest.approx(62.5)


def test_savings_rate_is_none_without_income() -> None:
    assert calculate_savings_rate(pd.DataFrame({"amount": [-100.0]})) is None


def test_spending_by_category_returns_positive_totals() -> None:
    result = spending_by_category(sample_transactions())

    assert result.to_dict() == {"Housing": 500.0, "Food": 250.0}
    assert result.name == "expenses"


def test_monthly_summary_handles_month_without_income() -> None:
    result = monthly_summary(sample_transactions())

    assert result.loc[0, "savings_rate"] == pytest.approx(75.0)
    assert pd.isna(result.loc[1, "savings_rate"])
    assert result["net_savings"].tolist() == [1500.0, -250.0]


def test_category_summary_uses_positive_expense_magnitudes() -> None:
    result = category_summary(sample_transactions())

    assert result["category"].tolist() == ["Housing", "Food"]
    assert result["share_percent"].sum() == pytest.approx(100.0)
