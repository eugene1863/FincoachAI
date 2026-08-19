"""Rule-based transaction categorization.

Phase 2 intentionally uses transparent merchant rules. The rules are easy to
review and extend, and do not require training data or a machine-learning model.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pandas as pd

CATEGORY_RULES: Mapping[str, Sequence[str]] = {
    "Groceries": ("rewe", "edeka", "lidl", "aldi"),
    "Transport": ("bvg", "deutsche bahn", "uber"),
    "Entertainment": ("netflix", "spotify", "disney"),
    "Shopping": ("amazon", "zalando"),
    "Dining": ("lieferando", "mcdonalds", "restaurant"),
    "Housing": ("rent", "vonovia"),
    "Income": ("salary", "payroll", "freelance payment"),
}

UNCLASSIFIED_CATEGORIES = {"", "uncategorized"}


def categorize_transaction(description: str | None) -> str:
    """Return the category matching a transaction description.

    Matching is case-insensitive and checks whether a configured merchant term
    occurs anywhere in the description. Missing, blank, and unmatched
    descriptions are categorized as ``"Other"``.

    Args:
        description: Merchant or transaction description.

    Returns:
        The matching category, or ``"Other"`` when no rule matches.
    """

    if description is None or pd.isna(description):
        return "Other"

    normalized = str(description).strip().casefold()
    if not normalized:
        return "Other"

    for category, keywords in CATEGORY_RULES.items():
        if any(keyword in normalized for keyword in keywords):
            return category
    return "Other"


def categorize_transactions(
    transactions: pd.DataFrame,
    *,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Fill transaction categories from descriptions and return a new DataFrame.

    Existing nonblank categories are preserved by default. Blank values and the
    preprocessing placeholder ``"Uncategorized"`` are automatically filled.
    Set ``overwrite=True`` to recategorize every row explicitly.

    Args:
        transactions: DataFrame containing a ``description`` column.
        overwrite: Whether to replace valid existing categories.

    Returns:
        A copy of ``transactions`` with an updated ``category`` column.

    Raises:
        ValueError: If the required ``description`` column is absent.
    """

    if "description" not in transactions.columns:
        raise ValueError("Transactions must contain a 'description' column")

    categorized = transactions.copy()
    if "category" not in categorized.columns:
        categorized["category"] = pd.Series(pd.NA, index=categorized.index, dtype="string")
    else:
        categorized["category"] = categorized["category"].astype("string")

    normalized_categories = categorized["category"].fillna("").str.strip().str.casefold()
    rows_to_update = pd.Series(overwrite, index=categorized.index)
    if not overwrite:
        rows_to_update = normalized_categories.isin(UNCLASSIFIED_CATEGORIES)

    categorized.loc[rows_to_update, "category"] = categorized.loc[
        rows_to_update, "description"
    ].map(categorize_transaction)
    return categorized

