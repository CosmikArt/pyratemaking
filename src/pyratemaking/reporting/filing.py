"""HTML rate filing template rendered with Jinja2.

The filing is a single self-contained HTML document with sections for
indication, on-leveling, trending, development, classification, and
implementation. Each section is optional — pass only what's available.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import jinja2
import pandas as pd

from pyratemaking._version import __version__
from pyratemaking.reporting.tables import (
    format_percent,
    format_relativity,
    style_actuarial_table,
)

_ENV = jinja2.Environment(
    loader=jinja2.PackageLoader("pyratemaking.reporting", "templates"),
    autoescape=jinja2.select_autoescape(["html"]),
    trim_blocks=True,
    lstrip_blocks=True,
)


def _to_html(styler) -> str:
    return styler.to_html(table_attributes='class="actuarial-table"')


@dataclass
class RatePlanReport:
    """Lightweight container for the pieces a filing template uses.

    Each attribute is optional; the template renders only the sections
    that are populated.
    """

    title: str = "Rate filing"
    indication: dict | None = None
    onleveling: pd.DataFrame | None = None
    trending: pd.DataFrame | None = None
    development: pd.DataFrame | None = None
    classification_relativities: dict[str, pd.DataFrame] = field(default_factory=dict)
    implementation_dispersion: pd.DataFrame | None = None
    implementation_segment: pd.DataFrame | None = None

    def render_html(self) -> str:
        return render_filing_html(self)

    def write(self, path: str | Path) -> Path:
        out = Path(path)
        out.write_text(self.render_html(), encoding="utf-8")
        return out


def render_filing_html(report: RatePlanReport) -> str:
    template = _ENV.get_template("filing.html.j2")
    ctx: dict[str, object] = {
        "title": report.title,
        "prepared_on": date.today().isoformat(),
        "version": __version__,
    }
    if report.indication is not None:
        ind = dict(report.indication)
        ind.setdefault("credibility", "1.00")
        if "table" not in ind:
            df = pd.DataFrame(ind.get("table_data", {}))
            ind["table"] = _to_html(style_actuarial_table(df))
        ctx["indication"] = ind
    if report.onleveling is not None:
        ctx["onleveling"] = {
            "table": _to_html(
                style_actuarial_table(
                    report.onleveling,
                    currency_cols=("earned_premium", "on_level_premium"),
                    relativity_cols=("avg_rate_level", "on_level_factor"),
                )
            )
        }
    if report.trending is not None:
        ctx["trending"] = {
            "table": _to_html(
                style_actuarial_table(
                    report.trending,
                    percent_cols=("annual_change",),
                )
            )
        }
    if report.development is not None:
        ctx["development"] = {
            "table": _to_html(
                style_actuarial_table(
                    report.development,
                    currency_cols=("ultimate", "latest_cumulative", "reserve"),
                    relativity_cols=("cdf_to_ult",),
                )
            )
        }
    if report.classification_relativities:
        ctx["classification"] = {
            "relativities": {
                var: _to_html(
                    style_actuarial_table(
                        rel,
                        relativity_cols=tuple(rel.columns),
                    )
                )
                for var, rel in report.classification_relativities.items()
            }
        }
    if report.implementation_dispersion is not None:
        ctx["implementation"] = {
            "dispersion": _to_html(
                style_actuarial_table(
                    report.implementation_dispersion,
                    percent_cols=("pct_of_book",),
                )
            ),
            "segment": (
                _to_html(
                    style_actuarial_table(
                        report.implementation_segment,
                        percent_cols=("avg_change", "median_change", "min", "max"),
                    )
                )
                if report.implementation_segment is not None
                else None
            ),
        }
    # Defensive defaults when sections are missing.
    for key in ("indication", "onleveling", "trending", "development", "classification", "implementation"):
        ctx.setdefault(key, None)

    # Make scalar formatting deterministic.
    if ctx.get("indication") and "rate_change" in ctx["indication"]:
        rc = ctx["indication"]["rate_change"]
        if isinstance(rc, (int, float)):
            ctx["indication"]["rate_change"] = format_percent(rc)
        if isinstance(ctx["indication"]["credibility"], (int, float)):
            ctx["indication"]["credibility"] = format_relativity(
                ctx["indication"]["credibility"], decimals=2
            )

    return template.render(**ctx)
