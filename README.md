# FinCoach AI

FinCoach AI is a modular personal-finance coaching application. This initial
foundation focuses on importing, cleaning, categorizing, summarizing, and
coaching from bank transactions. API, user-interface, anomaly-detection, LLM,
and chat features are intentionally deferred.

## Current capabilities

- Load transaction CSV files with configurable column names.
- Support either a signed `amount` column or separate `debit`/`credit` columns.
- Normalize descriptions, dates, amounts, currencies, and optional categories.
- Reject invalid rows with explicit reason codes instead of silently losing data.
- Remove duplicate transactions.
- Calculate income, expenses, net cash flow, and savings rate.
- Produce monthly and category-level summaries.
- Categorize known merchants with transparent, case-insensitive rules.
- Preserve valid user-provided categories unless replacement is requested.
- Compare monthly category spending with user-defined budgets.
- Track the monthly pace required for dated savings goals.
- Explain month-over-month spending changes and generate deterministic advice.

Amounts follow one convention throughout the project: income is positive and
expenses are negative.

## Project layout

```text
src/fincoach/
  transactions.py   CSV loading and transaction preprocessing
  analytics.py      Financial summary calculations
  categorization.py Rule-based merchant categorization
  coaching.py       Budgets, goals, spending changes, and recommendations
  exceptions.py     Domain-specific exceptions
tests/               Automated tests
data/                Local raw and processed data directories
```

## Setup

Python 3.11 or newer is recommended.

```bash
python -m venv .venv
pip install -e ".[dev]"
pytest
```

## Run the dashboard

Start the Streamlit application from the repository root:

```bash
python -m streamlit run app.py
```

Upload a CSV containing `date`, `description`, `amount`, and `category`, or select
the bundled sample-data option. Processing happens locally: the dashboard calls
the transaction loader, rule-based categorizer, analytics engine, and coaching
functions already provided by the `fincoach` package.

## Configure the conversational assistant

The dashboard can explain calculated financial summaries through an optional
OpenAI provider. Copy `.env.example` to `.env` and replace the placeholder:

```text
OPENAI_API_KEY=your-real-api-key
OPENAI_MODEL=gpt-5.2
```

Alternatively, set `OPENAI_API_KEY` in the shell before launching the app. Never
commit `.env`; it is excluded by `.gitignore`. The provider adapter uses the
OpenAI Responses API, and its interface can be replaced by another provider.

The assistant receives calculated monthly, category, budget, savings-goal, and
spending-change summaries. It does not receive the complete transaction table,
and it is explicitly instructed not to calculate or invent financial values.

## Example

```python
from fincoach import TransactionPreprocessor, calculate_financial_summary

result = TransactionPreprocessor().from_csv("transactions.csv")
print(result.transactions)
print(result.rejected_rows)
print(calculate_financial_summary(result.transactions))
```

By default, a CSV must contain `date`, `description`, and `amount`. Bank-specific
headings can be mapped with `TransactionColumns`; see the class docstrings and
tests for examples.

## Automatic categorization

The Phase 2 categorizer recognizes common merchants in transaction descriptions.
Unknown or missing descriptions become `Other`. Existing categories are kept by
default; pass `overwrite=True` only when they should be recalculated.

```python
from fincoach import categorize_transactions, load_transactions

transactions = load_transactions("examples/sample_transactions.csv")
categorized = categorize_transactions(transactions)
print(categorized[["description", "category"]])
```

## Financial coaching

Coaching recommendations are generated directly from calculated values—no LLM
is involved. Budgets use `under_budget`, `near_limit` (80–100% used), and
`over_budget` statuses. Goal tracking compares the required monthly contribution
with the user's current average, while spending-change analysis compares the
latest calendar month with the immediately preceding month.

```python
from datetime import date

from fincoach import (
    analyze_budgets,
    analyze_savings_goal,
    categorize_transactions,
    generate_recommendations,
    load_transactions,
)

transactions = categorize_transactions(
    load_transactions("examples/sample_transactions.csv")
)
budgets = {"Groceries": 50, "Transport": 60, "Entertainment": 20}
goal = analyze_savings_goal(5000, 1500, date(2027, 8, 1), 200)

print(analyze_budgets(transactions, budgets))
print(generate_recommendations(transactions, budgets=budgets, savings_goal=goal))
```

## Data safety

Never commit real bank exports. The contents of `data/raw` and `data/processed`
are ignored by Git. This project is an analytics foundation, not financial,
investment, tax, or legal advice.
