"""Core, deterministic financial calculations over clean transactions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = {"amount"}


@dataclass(frozen=True, slots=True)
class FinancialSummary:
    """Headline cash-flow metrics for a collection of transactions."""

    income: float
    expenses: float
    net_cash_flow: float
    savings_rate: float | None
    transaction_count: int


def calculate_financial_summary(transactions: pd.DataFrame) -> FinancialSummary:
    """Calculate income, expense outflow, cash flow, and savings rate.

    Expenses are returned as a positive magnitude. Savings rate is ``None`` when
    there is no income, because dividing by zero would be misleading.
    """

    income = calculate_total_income(transactions)
    expenses = calculate_total_expenses(transactions)
    net_cash_flow = calculate_net_savings(transactions)
    savings_rate = calculate_savings_rate(transactions)
    return FinancialSummary(
        income=income,
        expenses=expenses,
        net_cash_flow=net_cash_flow,
        savings_rate=savings_rate,
        transaction_count=len(transactions),
    )


def calculate_total_income(transactions: pd.DataFrame) -> float:
    """Return the sum of all positive transaction amounts."""

    amounts = _amounts(transactions)
    return float(amounts[amounts > 0].sum())


def calculate_total_expenses(transactions: pd.DataFrame) -> float:
    """Return total expenses as a positive monetary value."""

    amounts = _amounts(transactions)
    return float(-amounts[amounts < 0].sum())


def calculate_net_savings(transactions: pd.DataFrame) -> float:
    """Return income remaining after expenses; negative means overspending."""

    return calculate_total_income(transactions) - calculate_total_expenses(transactions)


def calculate_savings_rate(transactions: pd.DataFrame) -> float | None:
    """Return net savings as a percentage of income, or ``None`` without income."""

    income = calculate_total_income(transactions)
    if income == 0:
        return None
    return calculate_net_savings(transactions) / income * 100


def spending_by_category(transactions: pd.DataFrame) -> pd.Series:
    """Return positive expense totals grouped by category, largest first."""

    _validate_transactions(transactions, extra_columns={"category"})
    expenses = transactions.loc[transactions["amount"] < 0, ["category", "amount"]].copy()
    if expenses.empty:
        return pd.Series(dtype="float64", name="expenses")

    expenses["category"] = expenses["category"].fillna("Uncategorized")
    result = -expenses.groupby("category")["amount"].sum()
    return result.sort_values(ascending=False).rename("expenses")


def monthly_summary(transactions: pd.DataFrame) -> pd.DataFrame:
    """Return cash-flow metrics grouped by calendar month."""

    _validate_transactions(transactions, extra_columns={"date"})
    if transactions.empty:
        return pd.DataFrame(
            columns=["month", "income", "expenses", "net_savings", "savings_rate"]
        )

    data = transactions.copy()
    dates = pd.to_datetime(data["date"], errors="raise")
    data["month"] = dates.dt.to_period("M").astype("string")
    data["income"] = data["amount"].clip(lower=0)
    data["expenses"] = -data["amount"].clip(upper=0)
    grouped = data.groupby("month", sort=True, as_index=False)[["income", "expenses"]].sum()
    grouped["net_savings"] = grouped["income"] - grouped["expenses"]
    grouped["savings_rate"] = np.where(
        grouped["income"] > 0,
        grouped["net_savings"] / grouped["income"] * 100,
        np.nan,
    )
    return grouped


def category_summary(transactions: pd.DataFrame) -> pd.DataFrame:
    """Aggregate expense magnitude and share by category."""

    spending = spending_by_category(transactions)
    if spending.empty:
        return pd.DataFrame(columns=["category", "expenses", "share_percent"])
    grouped = spending.rename_axis("category").reset_index()
    grouped["share_percent"] = grouped["expenses"] / grouped["expenses"].sum() * 100
    return grouped


def _amounts(transactions: pd.DataFrame) -> pd.Series:
    """Validate and return the numeric amount series."""

    _validate_transactions(transactions)
    return pd.to_numeric(transactions["amount"], errors="raise")


def _validate_transactions(
    transactions: pd.DataFrame, *, extra_columns: set[str] | None = None
) -> None:
    required = REQUIRED_COLUMNS | (extra_columns or set())
    missing = sorted(required - set(transactions.columns))
    if missing:
        raise ValueError(f"Transactions are missing canonical columns: {missing}")
