"""Load and normalize bank transaction data.

The preprocessing boundary deliberately emits a small canonical schema. All
downstream analytics can therefore ignore bank-specific headings and formats.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import IO

import numpy as np
import pandas as pd

from fincoach.exceptions import TransactionSchemaError

CANONICAL_COLUMNS = [
    "date",
    "description",
    "amount",
    "category",
    "currency",
    "transaction_type",
    "year",
    "month",
]

PHASE_ONE_COLUMNS = ["date", "description", "amount", "category"]


def load_transactions(
    source: str | Path | IO[str],
    **read_csv_options: object,
) -> pd.DataFrame:
    """Load and clean a Phase 1 transaction CSV file.

    The CSV must contain ``date``, ``description``, ``amount``, and ``category``.
    Invalid dates, non-numeric amounts, and rows without a description cannot be
    analyzed reliably, so those rows are removed. Missing or blank categories
    are retained as ``"Uncategorized"``.

    Positive amounts represent income and negative amounts represent expenses.
    The returned frame contains only the required columns, is ordered by date,
    and has a fresh integer index.

    Args:
        source: CSV path or text file-like object accepted by ``pandas.read_csv``.
        **read_csv_options: Additional options passed to ``pandas.read_csv``.

    Returns:
        A clean transaction DataFrame.

    Raises:
        TransactionSchemaError: If any required CSV column is absent.
    """

    csv_options = dict(read_csv_options)
    if "sep" not in csv_options and "delimiter" not in csv_options:
        # Bank exports commonly use commas, semicolons, or tabs depending on the
        # user's locale. Python's CSV sniffer handles these without requiring a
        # dashboard setting; callers can still provide an explicit separator.
        csv_options.setdefault("sep", None)
        csv_options.setdefault("engine", "python")

    transactions = pd.read_csv(source, **csv_options)
    transactions.columns = [
        str(column).lstrip("\ufeff").strip().casefold() for column in transactions.columns
    ]
    missing = sorted(set(PHASE_ONE_COLUMNS) - set(transactions.columns))
    if missing:
        available = sorted(str(column) for column in transactions.columns)
        raise TransactionSchemaError(
            f"Missing required transaction columns: {missing}. "
            f"Detected columns: {available}"
        )

    clean = transactions[PHASE_ONE_COLUMNS].copy()
    clean["date"] = pd.to_datetime(clean["date"], errors="coerce")
    clean["amount"] = pd.to_numeric(clean["amount"], errors="coerce")
    clean["description"] = _clean_text(clean["description"])
    clean["category"] = _clean_text(clean["category"]).fillna("Uncategorized")

    clean = clean.dropna(subset=["date", "description", "amount"])
    clean = clean.drop_duplicates().sort_values("date", kind="stable")
    clean["amount"] = clean["amount"].astype("float64")
    return clean.reset_index(drop=True)


def _clean_text(values: pd.Series) -> pd.Series:
    """Trim text, collapse internal whitespace, and represent blanks as missing."""

    cleaned = values.astype("string").str.strip().str.replace(r"\s+", " ", regex=True)
    return cleaned.mask(cleaned.eq(""))


@dataclass(frozen=True, slots=True)
class TransactionColumns:
    """Map bank-specific CSV headings to FinCoach transaction fields.

    Supply ``amount`` for a signed amount column, or supply both ``debit`` and
    ``credit``. Debit values are converted to negative amounts and credit values
    to positive amounts.
    """

    date: str = "date"
    description: str = "description"
    amount: str | None = "amount"
    debit: str | None = None
    credit: str | None = None
    category: str | None = "category"
    currency: str | None = "currency"

    def validate(self) -> None:
        uses_amount = self.amount is not None
        uses_debit_credit = self.debit is not None or self.credit is not None
        if uses_amount == uses_debit_credit:
            raise ValueError("Configure either amount or debit/credit columns, but not both")
        if uses_debit_credit and (self.debit is None or self.credit is None):
            raise ValueError("Both debit and credit columns are required")


@dataclass(slots=True)
class PreprocessingResult:
    """Clean transactions plus source rows that could not be processed."""

    transactions: pd.DataFrame
    rejected_rows: pd.DataFrame
    duplicate_count: int


class TransactionPreprocessor:
    """Convert a bank export into FinCoach's canonical transaction schema."""

    def __init__(
        self,
        columns: TransactionColumns | None = None,
        *,
        day_first: bool = False,
        default_currency: str = "EUR",
    ) -> None:
        self.columns = columns or TransactionColumns()
        self.columns.validate()
        self.day_first = day_first
        self.default_currency = default_currency.strip().upper()

    def from_csv(
        self,
        source: str | Path | IO[str],
        **read_csv_options: object,
    ) -> PreprocessingResult:
        """Read a CSV and preprocess it.

        Extra keyword arguments are passed to :func:`pandas.read_csv`, allowing
        callers to specify delimiters, encodings, decimal conventions, and other
        bank-specific settings.
        """

        frame = pd.read_csv(source, **read_csv_options)
        return self.process(frame)

    def process(self, frame: pd.DataFrame) -> PreprocessingResult:
        """Clean a DataFrame without mutating the caller's object."""

        self._validate_schema(frame)
        working = frame.copy()
        canonical = pd.DataFrame(index=working.index)
        canonical["date"] = pd.to_datetime(
            working[self.columns.date], errors="coerce", dayfirst=self.day_first
        )
        canonical["description"] = (
            working[self.columns.description].astype("string").str.strip().str.replace(
                r"\s+", " ", regex=True
            )
        )
        canonical["amount"] = self._build_amount(working)

        if self.columns.category and self.columns.category in working:
            canonical["category"] = self._clean_optional_text(working[self.columns.category])
        else:
            canonical["category"] = pd.Series("Uncategorized", index=working.index, dtype="string")

        if self.columns.currency and self.columns.currency in working:
            currency = self._clean_optional_text(working[self.columns.currency]).str.upper()
            canonical["currency"] = currency.fillna(self.default_currency)
        else:
            canonical["currency"] = pd.Series(
                self.default_currency, index=working.index, dtype="string"
            )

        invalid_date = canonical["date"].isna()
        invalid_description = canonical["description"].isna() | canonical["description"].eq("")
        invalid_amount = canonical["amount"].isna()
        invalid = invalid_date | invalid_description | invalid_amount

        reasons = np.select(
            [invalid_date, invalid_description, invalid_amount],
            ["invalid_date", "missing_description", "invalid_amount"],
            default="",
        )
        rejected = working.loc[invalid].copy()
        rejected["rejection_reason"] = pd.Series(reasons, index=working.index).loc[invalid]

        clean = canonical.loc[~invalid].copy()
        clean["amount"] = clean["amount"].astype("float64")
        before_deduplication = len(clean)
        clean = clean.drop_duplicates(
            subset=["date", "description", "amount", "currency"], keep="first"
        )
        duplicate_count = before_deduplication - len(clean)

        clean["transaction_type"] = np.select(
            [clean["amount"].gt(0), clean["amount"].lt(0)],
            ["income", "expense"],
            default="neutral",
        )
        clean["year"] = clean["date"].dt.year.astype("int64")
        clean["month"] = clean["date"].dt.to_period("M").astype("string")
        clean = clean.sort_values("date", kind="stable").reset_index(drop=True)

        return PreprocessingResult(
            transactions=clean[CANONICAL_COLUMNS],
            rejected_rows=rejected.reset_index(drop=True),
            duplicate_count=duplicate_count,
        )

    def _validate_schema(self, frame: pd.DataFrame) -> None:
        required = {self.columns.date, self.columns.description}
        if self.columns.amount:
            required.add(self.columns.amount)
        else:
            required.update({self.columns.debit, self.columns.credit})
        missing = sorted(str(column) for column in required if column not in frame.columns)
        if missing:
            raise TransactionSchemaError(f"Missing required transaction columns: {missing}")

    def _build_amount(self, frame: pd.DataFrame) -> pd.Series:
        if self.columns.amount:
            return pd.to_numeric(frame[self.columns.amount], errors="coerce")

        parsed_debit = pd.to_numeric(frame[self.columns.debit], errors="coerce")
        parsed_credit = pd.to_numeric(frame[self.columns.credit], errors="coerce")
        amount = parsed_credit.fillna(0).abs() - parsed_debit.fillna(0).abs()

        # A row with no parseable numeric value should be rejected, while a real
        # zero-value transaction remains valid.
        no_numeric_value = parsed_debit.isna() & parsed_credit.isna()
        return amount.mask(no_numeric_value)

    @staticmethod
    def _clean_optional_text(values: pd.Series) -> pd.Series:
        return _clean_text(values)
