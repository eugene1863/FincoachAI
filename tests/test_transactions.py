from io import StringIO

import pandas as pd
import pytest

from fincoach.exceptions import TransactionSchemaError
from fincoach.transactions import TransactionColumns, TransactionPreprocessor, load_transactions


def test_load_transactions_returns_clean_dataframe() -> None:
    csv = StringIO(
        "date,description,amount,category\n"
        "2026-02-02,  Groceries  ,-45.50, Food \n"
        "2026-01-31,Salary,2500,Income\n"
        "invalid,Broken row,-10,Other\n"
        "2026-02-03,,not-a-number,\n"
        "2026-02-04,Coffee,-4.50,\n"
    )

    result = load_transactions(csv)

    assert result.columns.tolist() == ["date", "description", "amount", "category"]
    assert result["description"].tolist() == ["Salary", "Groceries", "Coffee"]
    assert result["amount"].tolist() == [2500.0, -45.5, -4.5]
    assert result["category"].tolist() == ["Income", "Food", "Uncategorized"]
    assert pd.api.types.is_datetime64_any_dtype(result["date"])
    assert pd.api.types.is_float_dtype(result["amount"])


@pytest.mark.parametrize("missing_column", ["date", "description", "amount", "category"])
def test_load_transactions_requires_phase_one_columns(missing_column: str) -> None:
    columns = {
        "date": ["2026-01-01"],
        "description": ["Salary"],
        "amount": [2000],
        "category": ["Income"],
    }
    del columns[missing_column]

    with pytest.raises(TransactionSchemaError, match=missing_column):
        load_transactions(StringIO(pd.DataFrame(columns).to_csv(index=False)))


def test_load_transactions_detects_semicolon_delimiter() -> None:
    csv = StringIO(
        "date;description;amount;category\n"
        "2026-08-01;Salary;2500;Income\n"
        "2026-08-02;REWE;-42.50;Groceries\n"
    )

    result = load_transactions(csv)

    assert result["amount"].tolist() == [2500.0, -42.5]


def test_load_transactions_normalizes_standard_headers() -> None:
    csv = StringIO(
        "\ufeff Date , DESCRIPTION , Amount , Category \n"
        "2026-08-01,Salary,2500,Income\n"
    )

    result = load_transactions(csv)

    assert result.columns.tolist() == ["date", "description", "amount", "category"]


def test_processes_signed_amounts_and_rejects_invalid_rows() -> None:
    source = pd.DataFrame(
        {
            "date": ["2026-01-03", "bad date", "2026-01-04", "2026-01-03"],
            "description": ["  Salary  ", "Shop", "Coffee", "Salary"],
            "amount": [2500, -40, -4.5, 2500],
        }
    )

    result = TransactionPreprocessor().process(source)

    assert len(result.transactions) == 2
    assert result.duplicate_count == 1
    assert result.transactions["description"].tolist() == ["Salary", "Coffee"]
    assert result.transactions["transaction_type"].tolist() == ["income", "expense"]
    assert result.rejected_rows["rejection_reason"].tolist() == ["invalid_date"]


def test_supports_separate_debit_and_credit_columns() -> None:
    source = pd.DataFrame(
        {
            "Booked": ["01/02/2026", "02/02/2026"],
            "Details": ["Rent", "Salary"],
            "Paid out": [900, None],
            "Paid in": [None, 2400],
        }
    )
    columns = TransactionColumns(
        date="Booked",
        description="Details",
        amount=None,
        debit="Paid out",
        credit="Paid in",
    )

    result = TransactionPreprocessor(columns, day_first=True).process(source)

    assert result.transactions["amount"].tolist() == [-900.0, 2400.0]


def test_reports_missing_required_columns() -> None:
    with pytest.raises(TransactionSchemaError, match="amount"):
        TransactionPreprocessor().process(pd.DataFrame({"date": [], "description": []}))
