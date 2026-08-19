"""Public interface for the FinCoach AI analytics foundation."""

from fincoach.analytics import (
    FinancialSummary,
    calculate_financial_summary,
    calculate_net_savings,
    calculate_savings_rate,
    calculate_total_expenses,
    calculate_total_income,
    category_summary,
    monthly_summary,
    spending_by_category,
)
from fincoach.categorization import categorize_transaction, categorize_transactions
from fincoach.chatbot import (
    ChatProviderError,
    FinancialChatbot,
    FinancialContext,
    build_financial_context,
    chatbot_from_environment,
    is_development_mode,
)
from fincoach.coaching import (
    BudgetAnalysis,
    CategorySpendingChange,
    SavingsGoalAnalysis,
    SpendingChangeAnalysis,
    analyze_budgets,
    analyze_savings_goal,
    analyze_spending_change,
    calculate_average_monthly_savings,
    generate_recommendations,
)
from fincoach.transactions import (
    PreprocessingResult,
    TransactionColumns,
    TransactionPreprocessor,
    load_transactions,
)

__all__ = [
    "FinancialSummary",
    "FinancialChatbot",
    "FinancialContext",
    "BudgetAnalysis",
    "CategorySpendingChange",
    "ChatProviderError",
    "PreprocessingResult",
    "SavingsGoalAnalysis",
    "SpendingChangeAnalysis",
    "TransactionColumns",
    "TransactionPreprocessor",
    "calculate_financial_summary",
    "calculate_average_monthly_savings",
    "calculate_net_savings",
    "calculate_savings_rate",
    "calculate_total_expenses",
    "calculate_total_income",
    "build_financial_context",
    "analyze_budgets",
    "analyze_savings_goal",
    "analyze_spending_change",
    "categorize_transaction",
    "categorize_transactions",
    "category_summary",
    "chatbot_from_environment",
    "is_development_mode",
    "generate_recommendations",
    "load_transactions",
    "monthly_summary",
    "spending_by_category",
]
