"""Deterministic budgeting, goal tracking, and financial coaching logic."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime

import pandas as pd

from fincoach.analytics import calculate_savings_rate, monthly_summary, spending_by_category

NEAR_LIMIT_PERCENT = 80.0
SIGNIFICANT_CHANGE_PERCENT = 20.0


@dataclass(frozen=True, slots=True)
class BudgetAnalysis:
    """Actual spending compared with one monthly category budget."""

    category: str
    budget: float
    actual_spending: float
    remaining_amount: float
    percentage_used: float
    status: str


@dataclass(frozen=True, slots=True)
class SavingsGoalAnalysis:
    """Progress and required pace for a savings goal."""

    goal_amount: float
    current_savings: float
    remaining_amount: float
    months_remaining: int
    required_monthly_savings: float
    average_monthly_savings: float
    on_track: bool


@dataclass(frozen=True, slots=True)
class CategorySpendingChange:
    """Month-over-month spending movement for one category."""

    category: str
    previous_spending: float
    latest_spending: float
    change: float
    percentage_change: float | None


@dataclass(frozen=True, slots=True)
class SpendingChangeAnalysis:
    """Overall and category-level comparison of two calendar months."""

    previous_month: str
    latest_month: str
    previous_spending: float
    latest_spending: float
    total_spending_change: float
    percentage_change: float | None
    category_changes: tuple[CategorySpendingChange, ...]
    biggest_spending_increase: CategorySpendingChange | None
    biggest_spending_decrease: CategorySpendingChange | None


def calculate_average_monthly_savings(transactions: pd.DataFrame) -> float:
    """Return mean monthly net savings across the available transaction history."""

    summary = monthly_summary(transactions)
    if summary.empty:
        return 0.0
    return float(summary["net_savings"].mean())


def analyze_budgets(
    transactions: pd.DataFrame,
    budgets: Mapping[str, float],
    *,
    month: str | pd.Period | None = None,
) -> tuple[BudgetAnalysis, ...]:
    """Compare category spending with budgets for one calendar month.

    When ``month`` is omitted, the latest month in the transaction data is used.
    A category is ``near_limit`` from 80% through 100% usage, and is
    ``over_budget`` above 100%. Budget amounts must be non-negative.
    """

    monthly_transactions = _transactions_for_month(transactions, month)
    actual_by_category = spending_by_category(monthly_transactions)
    results: list[BudgetAnalysis] = []

    for category, raw_budget in budgets.items():
        budget = float(raw_budget)
        if budget < 0:
            raise ValueError(f"Budget for {category!r} cannot be negative")

        actual = float(actual_by_category.get(category, 0.0))
        percentage = actual / budget * 100 if budget > 0 else (math.inf if actual > 0 else 0.0)
        if actual > budget:
            status = "over_budget"
        elif percentage >= NEAR_LIMIT_PERCENT:
            status = "near_limit"
        else:
            status = "under_budget"

        results.append(
            BudgetAnalysis(
                category=category,
                budget=budget,
                actual_spending=actual,
                remaining_amount=budget - actual,
                percentage_used=percentage,
                status=status,
            )
        )
    return tuple(results)


def analyze_savings_goal(
    goal_amount: float,
    current_savings: float,
    target_date: str | date | datetime | pd.Timestamp,
    average_monthly_savings: float,
    *,
    as_of: str | date | datetime | pd.Timestamp | None = None,
) -> SavingsGoalAnalysis:
    """Calculate the monthly saving pace required to reach a dated goal.

    ``as_of`` defaults to today and is injectable for repeatable reports and
    tests. Partial months count as a month in which saving can occur.
    """

    goal = float(goal_amount)
    current = float(current_savings)
    average = float(average_monthly_savings)
    if goal < 0 or current < 0 or average < 0:
        raise ValueError("Savings goal values cannot be negative")

    target = pd.Timestamp(target_date).normalize()
    reference = pd.Timestamp(as_of or date.today()).normalize()
    if pd.isna(target) or pd.isna(reference):
        raise ValueError("Target and reference dates must be valid")

    remaining = max(goal - current, 0.0)
    months = max(math.ceil((target - reference).days / 30.4375), 0)
    if remaining == 0:
        required = 0.0
        on_track = True
    elif months == 0:
        required = math.inf
        on_track = False
    else:
        required = remaining / months
        on_track = average >= required

    return SavingsGoalAnalysis(
        goal_amount=goal,
        current_savings=current,
        remaining_amount=remaining,
        months_remaining=months,
        required_monthly_savings=required,
        average_monthly_savings=average,
        on_track=on_track,
    )


def analyze_spending_change(transactions: pd.DataFrame) -> SpendingChangeAnalysis:
    """Compare the latest calendar month with the immediately previous month."""

    _require_columns(transactions, {"date", "amount", "category"})
    if transactions.empty:
        raise ValueError("At least one transaction is required")

    data = transactions.copy()
    data["date"] = pd.to_datetime(data["date"], errors="raise")
    latest_period = data["date"].dt.to_period("M").max()
    previous_period = latest_period - 1
    periods = data["date"].dt.to_period("M")
    latest = data.loc[periods == latest_period]
    previous = data.loc[periods == previous_period]

    latest_by_category = spending_by_category(latest)
    previous_by_category = spending_by_category(previous)
    categories = sorted(set(latest_by_category.index) | set(previous_by_category.index))

    changes: list[CategorySpendingChange] = []
    for category in categories:
        latest_amount = float(latest_by_category.get(category, 0.0))
        previous_amount = float(previous_by_category.get(category, 0.0))
        change = latest_amount - previous_amount
        changes.append(
            CategorySpendingChange(
                category=str(category),
                previous_spending=previous_amount,
                latest_spending=latest_amount,
                change=change,
                percentage_change=_percentage_change(previous_amount, change),
            )
        )

    previous_total = float(previous_by_category.sum())
    latest_total = float(latest_by_category.sum())
    total_change = latest_total - previous_total
    increases = [change for change in changes if change.change > 0]
    decreases = [change for change in changes if change.change < 0]

    return SpendingChangeAnalysis(
        previous_month=str(previous_period),
        latest_month=str(latest_period),
        previous_spending=previous_total,
        latest_spending=latest_total,
        total_spending_change=total_change,
        percentage_change=_percentage_change(previous_total, total_change),
        category_changes=tuple(changes),
        biggest_spending_increase=max(increases, key=lambda item: item.change, default=None),
        biggest_spending_decrease=min(decreases, key=lambda item: item.change, default=None),
    )


def generate_recommendations(
    transactions: pd.DataFrame,
    *,
    budgets: Mapping[str, float] | None = None,
    savings_goal: SavingsGoalAnalysis | None = None,
) -> list[str]:
    """Generate transparent recommendations from calculated financial metrics."""

    recommendations: list[str] = []
    latest = _transactions_for_month(transactions, None)

    if budgets:
        for result in analyze_budgets(transactions, budgets):
            if result.status == "over_budget":
                recommendations.append(
                    f"You exceeded your {result.category} budget by "
                    f"€{abs(result.remaining_amount):.2f}."
                )

    try:
        change_analysis = analyze_spending_change(transactions)
    except ValueError:
        change_analysis = None
    if change_analysis:
        dining = next(
            (item for item in change_analysis.category_changes if item.category == "Dining"), None
        )
        if (
            dining
            and dining.percentage_change is not None
            and dining.percentage_change >= SIGNIFICANT_CHANGE_PERCENT
        ):
            recommendations.append(
                "Your dining spending increased by "
                f"{dining.percentage_change:.0f}% compared with last month."
            )

    savings_rate = calculate_savings_rate(latest)
    if savings_rate is not None and savings_rate < 10:
        recommendations.append(
            "Your savings rate is below 10%. Consider reducing discretionary expenses."
        )
    elif savings_rate is not None and savings_rate > 20:
        recommendations.append(
            f"Great work—your savings rate is {savings_rate:.1f}%, which is above 20%."
        )

    if savings_goal and not savings_goal.on_track:
        additional = savings_goal.required_monthly_savings - savings_goal.average_monthly_savings
        if math.isinf(additional):
            recommendations.append(
                "Your savings goal deadline has arrived; review the target amount or date."
            )
        else:
            recommendations.append(
                f"To get back on track for your savings goal, save an additional "
                f"€{max(additional, 0.0):.2f} per month."
            )

    return recommendations


def _transactions_for_month(
    transactions: pd.DataFrame, month: str | pd.Period | None
) -> pd.DataFrame:
    """Return transactions for an explicit month, or the latest available month."""

    _require_columns(transactions, {"date", "amount", "category"})
    if transactions.empty:
        return transactions.copy()
    dates = pd.to_datetime(transactions["date"], errors="raise")
    periods = dates.dt.to_period("M")
    selected = pd.Period(month, freq="M") if month is not None else periods.max()
    return transactions.loc[periods == selected].copy()


def _percentage_change(previous: float, change: float) -> float | None:
    """Return percentage change, using ``None`` for a zero comparison base."""

    return change / previous * 100 if previous != 0 else None


def _require_columns(transactions: pd.DataFrame, required: set[str]) -> None:
    missing = sorted(required - set(transactions.columns))
    if missing:
        raise ValueError(f"Transactions are missing required columns: {missing}")
