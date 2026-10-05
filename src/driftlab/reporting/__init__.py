"""Paper tables (T1-T10, T7b, reported T3/T4) and their Markdown / CSV / LaTeX rendering."""

from __future__ import annotations

from driftlab.reporting.render import (
    render_csv,
    render_latex,
    render_markdown,
    write_tables,
)
from driftlab.reporting.tables import TABLE_ORDER, TableSpec, build_tables, load_reported

__all__ = [
    "TABLE_ORDER",
    "TableSpec",
    "build_tables",
    "load_reported",
    "render_csv",
    "render_latex",
    "render_markdown",
    "write_tables",
]
