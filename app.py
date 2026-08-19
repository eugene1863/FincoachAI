"""Streamlit dashboard for the FinCoach AI analytics and coaching engine."""

from __future__ import annotations

from dataclasses import asdict
from datetime import date, timedelta
from pathlib import Path
import sys

import pandas as pd
import streamlit as st

# Streamlit executes this root-level file without automatically adding the
# repository's ``src`` directory to Python's import path. Keep local launches
# working even before the project has been installed in editable mode.
PROJECT_ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from fincoach import (
    ChatProviderError,
    SavingsGoalAnalysis,
    analyze_budgets,
    analyze_savings_goal,
    analyze_spending_change,
    build_financial_context,
    calculate_average_monthly_savings,
    calculate_financial_summary,
    categorize_transactions,
    chatbot_from_environment,
    generate_recommendations,
    load_transactions,
    monthly_summary,
    is_development_mode,
    spending_by_category,
)
from fincoach.exceptions import TransactionSchemaError

SAMPLE_FILE = PROJECT_ROOT / "examples" / "sample_transactions.csv"
DEFAULT_BUDGETS = {
    "Groceries": 300.0,
    "Dining": 150.0,
    "Shopping": 200.0,
    "Entertainment": 100.0,
}
STATUS_LABELS = {
    "under_budget": "Under budget",
    "near_limit": "Near limit",
    "over_budget": "Over budget",
}


def main() -> None:
    """Render the FinCoach dashboard."""

    st.set_page_config(page_title="FinCoach AI", page_icon="💶", layout="wide")
    _apply_styles()

    st.title("FinCoach AI")
    st.caption("A private, deterministic view of your spending, savings, and goals.")

    uploaded_file = st.file_uploader(
        "Upload bank transactions",
        type=["csv"],
        help="Required columns: date, description, amount, category.",
    )
    use_sample = st.checkbox("Use bundled sample data", value=False)

    source = SAMPLE_FILE if use_sample else uploaded_file
    if source is None:
        st.info("Upload a CSV file to begin, or use the bundled sample data.")
        _show_csv_requirements()
        return

    try:
        transactions = categorize_transactions(load_transactions(source))
    except (
        TransactionSchemaError,
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
        UnicodeDecodeError,
        ValueError,
    ) as error:
        st.error(f"We could not process this CSV: {error}")
        st.caption("Check the required columns and ensure dates and amounts use valid formats.")
        return

    if transactions.empty:
        st.warning("The CSV contains no valid transaction rows after cleaning.")
        return

    st.success(f"Loaded {len(transactions):,} clean transactions.")
    _show_transaction_preview(transactions)
    _show_summary_metrics(transactions)
    _show_visualizations(transactions)
    budgets = _show_budget_section(transactions)
    goal = _show_savings_goal(transactions)
    _show_recommendations(transactions, budgets, goal)
    _show_chatbot(transactions, budgets, goal, source, use_sample)


def _show_csv_requirements() -> None:
    with st.expander("CSV format"):
        st.code(
            "date,description,amount,category\n"
            "2026-08-01,Salary,3200,Income\n"
            "2026-08-02,REWE,-64.35,",
            language="text",
        )
        st.caption("Use positive amounts for income and negative amounts for expenses.")


def _show_transaction_preview(transactions: pd.DataFrame) -> None:
    st.subheader("Cleaned transactions")
    st.dataframe(
        transactions,
        width="stretch",
        hide_index=True,
        column_config={
            "date": st.column_config.DateColumn("Date", format="DD MMM YYYY"),
            "amount": st.column_config.NumberColumn("Amount", format="€ %.2f"),
        },
    )


def _show_summary_metrics(transactions: pd.DataFrame) -> None:
    summary = calculate_financial_summary(transactions)
    columns = st.columns(4)
    columns[0].metric("Total income", _currency(summary.income))
    columns[1].metric("Total expenses", _currency(summary.expenses))
    columns[2].metric("Net savings", _currency(summary.net_cash_flow))
    columns[3].metric(
        "Savings rate",
        f"{summary.savings_rate:.1f}%" if summary.savings_rate is not None else "N/A",
    )


def _show_visualizations(transactions: pd.DataFrame) -> None:
    st.subheader("Financial overview")
    spending = spending_by_category(transactions)
    monthly = monthly_summary(transactions).set_index("month")

    left, right = st.columns(2)
    with left:
        st.markdown("**Spending by category**")
        if spending.empty:
            st.info("No expenses are available to chart.")
        else:
            st.bar_chart(spending, color="#2563EB")
    with right:
        st.markdown("**Monthly income vs expenses**")
        st.bar_chart(monthly[["income", "expenses"]], color=["#16A34A", "#DC2626"])

    left, right = st.columns(2)
    with left:
        st.markdown("**Monthly savings**")
        st.line_chart(monthly[["net_savings"]], color="#2563EB")
    with right:
        st.markdown("**Month-over-month spending**")
        try:
            comparison = analyze_spending_change(transactions)
            comparison_data = pd.DataFrame(
                {
                    comparison.previous_month: {
                        item.category: item.previous_spending
                        for item in comparison.category_changes
                    },
                    comparison.latest_month: {
                        item.category: item.latest_spending for item in comparison.category_changes
                    },
                }
            )
            if comparison_data.empty:
                st.info("No expense data is available to compare.")
            else:
                st.bar_chart(comparison_data, color=["#94A3B8", "#7C3AED"])
        except ValueError:
            st.info("Add transactions from another month to see a comparison.")


def _show_budget_section(transactions: pd.DataFrame) -> dict[str, float]:
    st.subheader("Monthly budgets")
    st.caption("Set category limits for the latest month in your data.")

    expense_categories = list(spending_by_category(transactions).index)
    categories = list(dict.fromkeys([*DEFAULT_BUDGETS, *expense_categories]))
    budgets: dict[str, float] = {}
    columns = st.columns(2)
    for index, category in enumerate(categories):
        default = DEFAULT_BUDGETS.get(category, 0.0)
        with columns[index % 2]:
            value = st.number_input(
                f"{category} budget (€)",
                min_value=0.0,
                value=default,
                step=25.0,
                key=f"budget_{category}",
            )
        if value > 0:
            budgets[category] = float(value)

    if not budgets:
        st.info("Enter at least one budget amount to see budget progress.")
        return budgets

    results = analyze_budgets(transactions, budgets)
    table = pd.DataFrame(asdict(result) for result in results)
    table["status"] = table["status"].map(STATUS_LABELS)
    st.dataframe(
        table,
        width="stretch",
        hide_index=True,
        column_config={
            "category": "Category",
            "budget": st.column_config.NumberColumn("Budget", format="€ %.2f"),
            "actual_spending": st.column_config.NumberColumn("Actual", format="€ %.2f"),
            "remaining_amount": st.column_config.NumberColumn("Remaining", format="€ %.2f"),
            "percentage_used": st.column_config.NumberColumn("Used", format="%.1f%%"),
            "status": "Status",
        },
    )
    return budgets


def _show_savings_goal(transactions: pd.DataFrame) -> SavingsGoalAnalysis:
    st.subheader("Savings goal")
    left, middle, right = st.columns(3)
    with left:
        target_amount = st.number_input(
            "Target amount (€)", min_value=0.0, value=5000.0, step=100.0
        )
    with middle:
        current_savings = st.number_input(
            "Current savings (€)", min_value=0.0, value=0.0, step=100.0
        )
    with right:
        target_date = st.date_input(
            "Target date", value=date.today() + timedelta(days=365), min_value=date.today()
        )

    average_monthly_savings = calculate_average_monthly_savings(transactions)
    goal = analyze_savings_goal(
        target_amount,
        current_savings,
        target_date,
        max(average_monthly_savings, 0.0),
    )

    status = "On track" if goal.on_track else "Behind target"
    columns = st.columns(4)
    columns[0].metric("Goal status", status)
    columns[1].metric("Remaining", _currency(goal.remaining_amount))
    columns[2].metric("Months remaining", str(goal.months_remaining))
    columns[3].metric("Required monthly", _currency(goal.required_monthly_savings))
    st.caption(
        f"Based on average monthly net savings of {_currency(average_monthly_savings)} "
        "from the uploaded history."
    )
    return goal


def _show_recommendations(
    transactions: pd.DataFrame,
    budgets: dict[str, float],
    goal: SavingsGoalAnalysis,
) -> None:
    st.subheader("Your coaching insights")
    recommendations = generate_recommendations(
        transactions, budgets=budgets, savings_goal=goal
    )
    if not recommendations:
        st.info("No recommendations are available for the current data.")
        return
    for recommendation in recommendations:
        st.info(recommendation)


def _show_chatbot(
    transactions: pd.DataFrame,
    budgets: dict[str, float],
    goal: SavingsGoalAnalysis,
    source: object,
    use_sample: bool,
) -> None:
    """Render chat history and send calculated context to the configured provider."""

    st.subheader("Ask FinCoach")
    st.caption(
        "Ask about spending, budgets, savings, or month-over-month changes. "
        "Only calculated summaries—not raw transaction rows—are sent to the LLM provider."
    )

    source_id = "bundled-sample" if use_sample else _uploaded_source_id(source)
    if st.session_state.get("chat_source_id") != source_id:
        st.session_state.chat_source_id = source_id
        st.session_state.chat_messages = []
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("Ask a question about your finances")
    if not question:
        return

    prior_history = list(st.session_state.chat_messages)
    st.session_state.chat_messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    chatbot = chatbot_from_environment()
    if chatbot is None:
        answer = (
            "The conversational assistant is not configured yet. Add `OPENAI_API_KEY` "
            "to your environment or `.env` file, then restart FinCoach."
        )
    else:
        try:
            context = build_financial_context(
                transactions,
                budgets=budgets,
                savings_goal=goal,
            )
            with st.spinner("Reviewing your calculated financial summary..."):
                answer = chatbot.answer(question, context, prior_history)
        except ChatProviderError as error:
            answer = (
                error.development_message() if is_development_mode() else error.user_message
            )
        except Exception as error:
            if is_development_mode():
                answer = (
                    "FinCoach encountered a local error before contacting OpenAI.\n\n"
                    f"- Exception type: `{type(error).__name__}`\n"
                    f"- Message: {error}"
                )
            else:
                answer = "FinCoach could not prepare the assistant request. Please try again."

    st.session_state.chat_messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)


def _uploaded_source_id(source: object) -> str:
    """Create a non-financial identity used only to reset chat on file changes."""

    name = getattr(source, "name", "uploaded.csv")
    size = getattr(source, "size", "unknown")
    return f"{name}:{size}"


def _currency(value: float) -> str:
    """Format a monetary value for display without performing calculations."""

    if value == float("inf"):
        return "N/A"
    return f"€{value:,.2f}"


def _apply_styles() -> None:
    st.markdown(
        """
        <style>
        .block-container {max-width: 1200px; padding-top: 2rem; padding-bottom: 4rem;}
        [data-testid="stMetric"] {background: #f8fafc; border: 1px solid #e2e8f0;
                                  border-radius: 0.75rem; padding: 1rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
