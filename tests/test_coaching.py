from datetime import date

import pandas as pd
import pytest

from fincoach.coaching import (
    analyze_budgets,
    analyze_savings_goal,
    analyze_spending_change,
    calculate_average_monthly_savings,
    generate_recommendations,
)


@pytest.fixture
def transactions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2026-01-02",
                    "2026-01-05",
                    "2026-01-06",
                    "2026-02-02",
                    "2026-02-05",
                    "2026-02-07",
                ]
            ),
            "description": ["Salary", "REWE", "Restaurant", "Salary", "REWE", "Restaurant"],
            "amount": [2000.0, -200.0, -50.0, 2000.0, -250.0, -100.0],
            "category": ["Income", "Groceries", "Dining", "Income", "Groceries", "Dining"],
        }
    )


def test_under_budget_spending(transactions: pd.DataFrame) -> None:
    result = analyze_budgets(transactions, {"Groceries": 400})[0]

    assert result.actual_spending == 250.0
    assert result.remaining_amount == 150.0
    assert result.percentage_used == pytest.approx(62.5)
    assert result.status == "under_budget"


def test_over_budget_spending(transactions: pd.DataFrame) -> None:
    result = analyze_budgets(transactions, {"Dining": 75})[0]

    assert result.actual_spending == 100.0
    assert result.remaining_amount == -25.0
    assert result.status == "over_budget"


def test_savings_goal_on_track() -> None:
    result = analyze_savings_goal(
        6000, 3000, date(2026, 7, 1), 600, as_of=date(2026, 1, 1)
    )

    assert result.months_remaining == 6
    assert result.required_monthly_savings == 500.0
    assert result.on_track is True


def test_savings_goal_behind_target() -> None:
    result = analyze_savings_goal(
        6000, 3000, date(2026, 7, 1), 300, as_of=date(2026, 1, 1)
    )

    assert result.remaining_amount == 3000.0
    assert result.required_monthly_savings == 500.0
    assert result.on_track is False


def test_average_monthly_savings(transactions: pd.DataFrame) -> None:
    assert calculate_average_monthly_savings(transactions) == 1700.0


def test_month_over_month_spending_increase(transactions: pd.DataFrame) -> None:
    result = analyze_spending_change(transactions)

    assert result.previous_spending == 250.0
    assert result.latest_spending == 350.0
    assert result.total_spending_change == 100.0
    assert result.percentage_change == 40.0
    assert result.biggest_spending_increase.category in {"Dining", "Groceries"}
    assert result.biggest_spending_increase.change == 50.0


def test_month_over_month_spending_decrease() -> None:
    transactions = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-01", "2026-02-01"]),
            "amount": [-500.0, -300.0],
            "category": ["Shopping", "Shopping"],
        }
    )

    result = analyze_spending_change(transactions)

    assert result.total_spending_change == -200.0
    assert result.percentage_change == -40.0
    assert result.biggest_spending_decrease.category == "Shopping"


def test_recommendation_generation(transactions: pd.DataFrame) -> None:
    goal = analyze_savings_goal(
        6000, 3000, date(2026, 7, 1), 300, as_of=date(2026, 1, 1)
    )

    recommendations = generate_recommendations(
        transactions, budgets={"Dining": 75}, savings_goal=goal
    )

    assert "You exceeded your Dining budget by €25.00." in recommendations
    assert "Your dining spending increased by 100% compared with last month." in recommendations
    assert any("above 20%" in item for item in recommendations)
    assert any("additional €200.00 per month" in item for item in recommendations)
