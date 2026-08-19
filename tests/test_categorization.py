import pandas as pd
import pytest

from fincoach.categorization import categorize_transaction, categorize_transactions


@pytest.mark.parametrize(
    ("description", "expected"),
    [
        ("REWE Markt Berlin", "Groceries"),
        ("Deutsche Bahn Ticket", "Transport"),
        ("NETFLIX.COM", "Entertainment"),
        ("Amazon Marketplace", "Shopping"),
        ("Lieferando Order", "Dining"),
        ("Monthly Rent", "Housing"),
        ("Freelance Payment August", "Income"),
    ],
)
def test_categorizes_known_merchants(description: str, expected: str) -> None:
    assert categorize_transaction(description) == expected


@pytest.mark.parametrize("description", ["lidl", "LiDl Store", "LIDL STORE"])
def test_matching_is_case_insensitive(description: str) -> None:
    assert categorize_transaction(description) == "Groceries"


def test_unknown_merchant_is_other() -> None:
    assert categorize_transaction("Local Book Shop") == "Other"


@pytest.mark.parametrize("description", [None, "", "   ", pd.NA])
def test_missing_description_is_other(description: str | None) -> None:
    assert categorize_transaction(description) == "Other"


def test_preserves_existing_categories_by_default() -> None:
    source = pd.DataFrame(
        {
            "description": ["REWE Markt", "Spotify", "Uber"],
            "category": ["Household", "", "Uncategorized"],
        }
    )

    result = categorize_transactions(source)

    assert result["category"].tolist() == ["Household", "Entertainment", "Transport"]
    assert source["category"].tolist() == ["Household", "", "Uncategorized"]


def test_overwrite_recategorizes_existing_categories() -> None:
    source = pd.DataFrame({"description": ["REWE Markt"], "category": ["Household"]})

    result = categorize_transactions(source, overwrite=True)

    assert result.loc[0, "category"] == "Groceries"


def test_creates_category_column_when_missing() -> None:
    source = pd.DataFrame({"description": ["Salary ACME", None]})

    result = categorize_transactions(source)

    assert result["category"].tolist() == ["Income", "Other"]

