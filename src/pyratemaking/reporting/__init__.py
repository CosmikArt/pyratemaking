"""Exam-quality tables, HTML rate filings, Excel exports.

The styler in :mod:`pyratemaking.reporting.tables` formats actuarial frames
with the conventions filers expect: relativities to 4 decimals, percentages
with a sign, currency with thousands separators. The HTML filing template
stitches the indication, on-leveling, trend, development, classification,
and implementation exhibits into one self-contained file.
"""

from pyratemaking.reporting.excel import write_excel
from pyratemaking.reporting.filing import RatePlanReport, render_filing_html
from pyratemaking.reporting.tables import (
    format_currency,
    format_percent,
    format_relativity,
    style_actuarial_table,
)

__all__ = [
    "RatePlanReport",
    "format_currency",
    "format_percent",
    "format_relativity",
    "render_filing_html",
    "style_actuarial_table",
    "write_excel",
]
