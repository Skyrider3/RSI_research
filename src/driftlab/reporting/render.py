"""Render :class:`~driftlab.reporting.tables.TableSpec` objects as Markdown, CSV and LaTeX, and write them.

``write_tables`` is what the pipeline's analyze stage and ``driftlab tables`` call. It refuses to write a
SYNTHETIC bundle into any path with a ``results`` directory component: synthetic numbers must never land
next to the experimental results (they stay in the run's own ``exports/tables``). CSV files of a synthetic run
start with a ``# SYNTHETIC DATA ...`` line, so the marking survives when a CSV leaves the run directory.
"""

from __future__ import annotations

import csv
import io
import re
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from driftlab.analysis.bundle_io import SYNTHETIC_NOTE, AnalysisBundle
from driftlab.reporting.tables import DASH, TABLE_ORDER, TableSpec, build_tables, is_synthetic

FORMATS: tuple[str, ...] = ("md", "csv", "tex")
ALL_TABLES = "all_tables.md"
ALL_TABLES_EXTENDED = "all_tables_extended.md"
# Names write_tables owns in an output directory (stale ones are removed so old numbers never linger).
_MANAGED = re.compile(r"^T\d+b?(_reported)?(_extended)?\.(md|csv|tex)$")
_CODE_SPAN = re.compile(r"(`[^`]*`)")

# LaTeX: one replacement per character (so inserted backslashes are never escaped twice).
_LATEX_MAP: dict[str, str] = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
    "<": r"\textless{}",
    ">": r"\textgreater{}",
    "‡": r"\ensuremath{\ddagger}",
    "±": r"$\pm$",
    "↑": r"$\uparrow$",
    "↓": r"$\downarrow$",
    "→": r"$\rightarrow$",
    "−": r"$-$",
    "–": "--",
    "—": "---",
    "×": r"$\times$",
    "≥": r"$\geq$",
    "≤": r"$\leq$",
    "·": r"$\cdot$",
    "Σ": r"$\Sigma$",
    "…": r"\ldots{}",
    "`": r"\textasciigrave{}",
}


def _latex_chars(text: str) -> str:
    return "".join(_LATEX_MAP.get(ch, ch) for ch in text)


def latex_escape(text: object) -> str:
    """Escape LaTeX specials (``% & _ # $ { } ~ ^ \\``) and map ‡ ± ↑ ↓ − – — × ≥ ≤ to LaTeX.

    Markdown code spans (`` `driftlab tables --run-dir X` ``) become ``\\texttt{...}`` with ``-`` protected so
    ``--`` is not typeset as a dash.
    """
    parts = _CODE_SPAN.split(str(text))
    out = []
    for i, part in enumerate(parts):
        if i % 2:
            out.append("\\texttt{" + _latex_chars(part[1:-1]).replace("-", "-{}") + "}")
        else:
            out.append(_latex_chars(part))
    return "".join(out)


def _cells(df: pd.DataFrame) -> list[list[str]]:
    if df is None or df.empty:
        return [[DASH] * (len(df.columns) if df is not None else 1)]
    return [[str(v) if str(v) != "" else DASH for v in row] for row in df.itertuples(index=False, name=None)]


def _numeric(value: str) -> bool:
    v = value.strip().replace(",", "").lstrip("+-").replace("‡", "").strip()
    if not v or v == DASH:
        return False
    head = v.split(" ", 1)[0]
    try:
        float(head)
    except ValueError:
        return False
    return True


def _right_aligned(df: pd.DataFrame) -> list[bool]:
    """Columns whose every non-missing cell starts with a number are right-aligned."""
    out = []
    for j, _ in enumerate(df.columns):
        vals = [str(v) for v in df.iloc[:, j]] if not df.empty else []
        present = [v for v in vals if v not in ("", DASH)]
        out.append(bool(present) and all(_numeric(v) for v in present) and j > 0)
    return out


def _title(spec: TableSpec, extended: bool) -> str:
    return spec.title + (" (extended)" if extended and spec.extended is not None else "")


def _heading(spec: TableSpec, extended: bool) -> str:
    return f"{spec.label}. {_title(spec, extended)}"


def _md_cell(v: object) -> str:
    """Markdown-safe text: ``<`` outside code spans becomes ``&lt;`` (``s<seed>`` would otherwise be read as an
    HTML tag and vanish on GitHub / in Streamlit), pipes are escaped, newlines become ``<br>``."""
    parts = _CODE_SPAN.split(str(v))
    text = "".join(part if i % 2 else part.replace("<", "&lt;") for i, part in enumerate(parts))
    return text.replace("|", "\\|").replace("\r", " ").replace("\n", "<br>")


def render_markdown(spec: TableSpec, extended: bool = False) -> str:
    """GitHub-flavoured Markdown: heading, table, numbered footnotes, caption (italic)."""
    df = spec.frame(extended)
    cols = [str(c) for c in df.columns]
    align = _right_aligned(df)
    lines = [f"### {_heading(spec, extended)}", ""]
    lines.append("| " + " | ".join(_md_cell(c) for c in cols) + " |")
    lines.append("|" + "|".join("---:" if a else "---" for a in align) + "|")
    for row in _cells(df):
        lines.append("| " + " | ".join(_md_cell(v) for v in row) + " |")
    notes = spec.notes(extended)
    if notes:
        lines.append("")
        lines.extend(f"{i}. {_md_cell(n)}" for i, n in enumerate(notes, 1))
    if spec.caption:
        lines += ["", f"*{spec.caption}*"]
    return "\n".join(lines) + "\n"


def _synthetic_marker(spec: TableSpec, extended: bool) -> str:
    """The table's own synthetic footnote (the reported tables carry a qualified variant)."""
    for note in reversed(spec.notes(extended)):
        if note.startswith(SYNTHETIC_NOTE):
            return note
    return SYNTHETIC_NOTE


def render_csv(spec: TableSpec, extended: bool = False, *, mark_synthetic: bool = True) -> str:
    """The table as CSV (header + rows, no index); footnotes live in the Markdown / LaTeX versions.

    A table of a synthetic run starts with one ``# SYNTHETIC DATA ...`` comment line (read it back with
    ``pandas.read_csv(path, comment="#")``) unless ``mark_synthetic=False``: a CSV copied out of the run
    directory must not pass for experimental results.
    """
    df = spec.frame(extended)
    buf = io.StringIO()
    if spec.synthetic and mark_synthetic:
        buf.write(f"# {_synthetic_marker(spec, extended)}\n")
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow([str(c) for c in df.columns])
    for row in df.itertuples(index=False, name=None):
        writer.writerow([str(v) for v in row])
    return buf.getvalue()


def render_latex(spec: TableSpec, extended: bool = False) -> str:
    """A booktabs ``table`` environment (needs ``\\usepackage{booktabs}``; extended tables also ``graphicx``)."""
    df = spec.frame(extended)
    is_ext = extended and spec.extended is not None
    align = "".join("r" if a else "l" for a in _right_aligned(df))
    label = spec.id.lower().replace("_", "-") + ("-extended" if is_ext else "")
    lines = [
        f"% {_heading(spec, extended)} -- generated by driftlab; do not edit by hand",
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        f"\\caption{{{latex_escape(_title(spec, extended))}}}",
        f"\\label{{tab:{label}}}",
    ]
    if is_ext:
        lines.append("\\resizebox{\\linewidth}{!}{%")
    lines += [
        f"\\begin{{tabular}}{{{align}}}",
        "\\toprule",
        " & ".join(latex_escape(c) for c in df.columns) + " \\\\",
        "\\midrule",
    ]
    lines += [" & ".join(latex_escape(v) for v in row) + " \\\\" for row in _cells(df)]
    lines += ["\\bottomrule", "\\end{tabular}"]
    if is_ext:
        lines.append("}")
    notes = spec.notes(extended)
    if notes or spec.caption:
        lines += ["\\par\\smallskip", "\\begin{minipage}{\\linewidth}\\footnotesize"]
        body = [f"({i}) {latex_escape(n)}" for i, n in enumerate(notes, 1)]
        if spec.caption:
            body.append(f"\\textit{{{latex_escape(spec.caption)}}}")
        lines.append(" \\\\\n".join(body))
        lines.append("\\end{minipage}")
    lines.append("\\end{table}")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- writing


def has_results_component(path: str | Path) -> bool:
    """True when ``path`` (as given or resolved) has a directory component named ``results``."""
    p = Path(path)
    parts = [*p.parts, *p.expanduser().resolve().parts]
    return any(part.lower() == "results" for part in parts)


def _check_destination(bundle: AnalysisBundle, out_dir: Path, allow_results_dir: bool) -> None:
    if is_synthetic(bundle) and has_results_component(out_dir):
        hint = " (allow_results_dir cannot override this for synthetic data)" if allow_results_dir else ""
        raise ValueError(
            f"refusing to write tables of a SYNTHETIC run into {out_dir}: paths with a 'results' directory are "
            f"reserved for experimental results{hint}; write them into the run's exports/tables instead"
        )


def _all_tables_md(bundle: AnalysisBundle, specs: dict[str, TableSpec], extended: bool) -> str:
    m = bundle.meta or {}
    head = [f"# DriftLab paper tables{' (extended)' if extended else ''}: run {m.get('run_id', '?')}", ""]
    if is_synthetic(bundle):
        head += [f"> **{SYNTHETIC_NOTE}**", ""]
    head += [
        f"Analysis plan {m.get('plan_version', '?')} (plan hash {bundle.plan_hash or '?'}); config hash "
        f"{m.get('config_hash', '?')}; analysis created {m.get('created_at', '?')}.",
        "",
    ]
    warnings = [str(w) for w in m.get("warnings") or []]
    if warnings:
        head += [f"> **Analysis warnings ({len(warnings)})** (from bundle.json):", ">"]
        head += [f"> - {_md_cell(w)}" for w in warnings]
        head += [""]
    body = [render_markdown(specs[t], extended=extended) for t in TABLE_ORDER if t in specs]
    return "\n".join(head) + "\n" + "\n".join(body)


def write_tables(
    bundle: AnalysisBundle,
    out_dir: str | Path,
    formats: Sequence[str] = FORMATS,
    *,
    reported: dict | None = None,
    allow_results_dir: bool = False,
    extended: bool = True,
    app_dir: Path | None = None,
) -> list[Path]:
    """Write every table as ``T*.md`` / ``.csv`` / ``.tex`` (+ ``T*_extended.*``) and ``all_tables.md`` /
    ``all_tables_extended.md`` (table order) into ``out_dir``; returns the written paths.

    Raises ``ValueError`` for an unknown format and when a synthetic bundle would be written into a path with a
    ``results`` directory component; ``allow_results_dir`` never lifts that refusal (real bundles may be written
    anywhere). ``extended=False`` skips the extended variants; ``reported`` / ``app_dir`` go to
    :func:`~driftlab.reporting.tables.build_tables`. Table files of the written formats that this call does not
    rewrite (``T*.<fmt>`` / ``T*_extended.<fmt>`` / ``all_tables*.md`` from an earlier bundle, e.g. a reported
    table that is now disabled) are removed, so ``out_dir`` never mixes numbers of two analyses.
    """
    fmts = [str(f).strip().lower().lstrip(".") for f in formats]
    unknown = [f for f in fmts if f not in FORMATS]
    if unknown or not fmts:
        raise ValueError(f"unknown table format(s) {unknown or fmts}; expected some of {FORMATS}")
    out = Path(out_dir)
    _check_destination(bundle, out, allow_results_dir)
    specs = build_tables(bundle, reported=reported, app_dir=app_dir)
    out.mkdir(parents=True, exist_ok=True)
    renderers = {"md": render_markdown, "csv": render_csv, "tex": render_latex}
    written: list[Path] = []

    def put(name: str, text: str) -> None:
        path = out / name
        path.write_text(text, encoding="utf-8")
        written.append(path)

    for tid in TABLE_ORDER:
        spec = specs.get(tid)
        if spec is None:
            continue
        for f in dict.fromkeys(fmts):
            put(f"{tid}.{f}", renderers[f](spec, False))
            if extended and spec.extended is not None:
                put(f"{tid}_extended.{f}", renderers[f](spec, True))
    if "md" in fmts:
        put(ALL_TABLES, _all_tables_md(bundle, specs, extended=False))
        if extended:
            put(ALL_TABLES_EXTENDED, _all_tables_md(bundle, specs, extended=True))
    keep = {p.name for p in written}
    for path in out.iterdir():
        managed = _MANAGED.match(path.name) or path.name in (ALL_TABLES, ALL_TABLES_EXTENDED)
        if managed and path.name not in keep and path.suffix.lstrip(".") in fmts and path.is_file():
            path.unlink()
    return written


__all__ = [
    "ALL_TABLES",
    "ALL_TABLES_EXTENDED",
    "FORMATS",
    "has_results_component",
    "latex_escape",
    "render_csv",
    "render_latex",
    "render_markdown",
    "write_tables",
]
