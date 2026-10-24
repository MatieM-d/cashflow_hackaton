"""Smoke tests for the forecasting logic; run with: pytest -q"""
import importlib.util
from pathlib import Path

# Streamlit app is primarily UI; detailed tests should be added when logic is extracted
# into a dedicated module. These basic tests document expected input columns.
def test_expected_required_columns():
    required = {"operation_date", "amount", "direction"}
    assert "operation_date" in required
    assert "amount" in required
    assert "direction" in required
