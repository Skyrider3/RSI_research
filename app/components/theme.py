"""Visual system for the DriftLab dashboard (validated palette + Plotly styling).

Rules (from the team's data-viz method; see docs/ARCHITECTURE.md §9):
* Colour follows the ENTITY, never its rank: every environment and policy has a fixed colour below.
* One y-axis per chart (no dual axes). Thin marks (2px lines, 8px markers), recessive solid hairline grid.
* Sequential = one hue (blue) light->dark; diverging (signed drift inflation) = blue <-> red with a gray midpoint.
* Status colours are reserved for good/warning/serious/critical and always paired with an icon + label.
* Aqua and yellow are below 3:1 contrast on the light surface: charts using them must carry direct labels or a
  table view (every chart in this app is paired with a data table).
* Text uses ink tokens, never series colours.
The categorical order (blue, orange, aqua, yellow, magenta, green, violet, red) was validated with the
data-viz palette validator in both light and dark mode (adjacent CVD ΔE ≥ 8.4, normal-vision ΔE ≥ 19.3).
"""

from __future__ import annotations

from dataclasses import dataclass

LIGHT_SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
DARK_SERIES = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"]

SEQUENTIAL_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIVERGING = {"neg": "#2a78d6", "mid_light": "#f0efec", "mid_dark": "#383835", "pos": "#e34948"}

STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}
STATUS_ICON = {"good": "✅", "warning": "⚠️", "serious": "🟠", "critical": "⛔"}

# Fixed entity -> categorical slot (index into the series lists). Never reassign by rank.
ENV_SLOT = {"E1": 0, "E2": 1, "E3": 2, "E4": 3}
ENV_LABEL = {
    "E1": "E1 · Unchanged",
    "E2": "E2 · Decoding change",
    "E3": "E3 · Extraction change",
    "E4": "E4 · Multiple changes",
}
ENV_SYMBOL = {"E1": "circle", "E2": "square", "E3": "diamond", "E4": "triangle-up"}  # secondary encoding
# Headline policies take the first three slots (validated all-pairs for scatter / Pareto charts); slot 4
# (yellow) is skipped for policies so yellow never sits beside orange; ORACLE is a neutral reference.
POLICY_SLOT = {"P1": 0, "P2": 1, "P3": 2, "P1b": 4, "P5": 5, "P4_k3": 6}
POLICY_NEUTRAL = {"light": "#898781", "dark": "#898781"}
REFERENCE_KIND_SLOT = {"stored": 0, "rescored": 4, "rerun": 1, "fresh": 2}


@dataclass(frozen=True)
class Ink:
    surface: str
    plane: str
    primary: str
    secondary: str
    muted: str
    grid: str
    axis: str
    border: str


LIGHT_INK = Ink(
    "#fcfcfb", "#f9f9f7", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "rgba(11,11,11,0.10)"
)
DARK_INK = Ink(
    "#1a1a19", "#0d0d0d", "#ffffff", "#c3c2b7", "#898781", "#2c2c2a", "#383835", "rgba(255,255,255,0.10)"
)

FONT_FAMILY = 'system-ui, -apple-system, "Segoe UI", sans-serif'


def is_dark() -> bool:
    """Best-effort detection of the active Streamlit theme (defaults to light)."""
    try:
        import streamlit as st

        ctx_theme = getattr(getattr(st, "context", None), "theme", None)
        t = getattr(ctx_theme, "type", None) if ctx_theme is not None else None
        if t:
            return str(t).lower() == "dark"
        base = st.get_option("theme.base")
        return str(base).lower() == "dark"
    except Exception:  # pragma: no cover - outside streamlit
        return False


def series(dark: bool | None = None) -> list[str]:
    return DARK_SERIES if (is_dark() if dark is None else dark) else LIGHT_SERIES


def ink(dark: bool | None = None) -> Ink:
    return DARK_INK if (is_dark() if dark is None else dark) else LIGHT_INK


def env_color(env_id: str, dark: bool | None = None) -> str:
    return series(dark)[ENV_SLOT.get(env_id, 7)]


def policy_color(policy: str, dark: bool | None = None) -> str:
    if policy.upper().startswith("ORACLE"):
        return POLICY_NEUTRAL["dark" if (is_dark() if dark is None else dark) else "light"]
    key = policy if policy in POLICY_SLOT else ("P4_k3" if policy.startswith("P4") else policy)
    return series(dark)[POLICY_SLOT.get(key, 7)]


def ref_color(kind: str, dark: bool | None = None) -> str:
    return series(dark)[REFERENCE_KIND_SLOT.get(kind, 7)]


def diverging_scale(dark: bool | None = None) -> list[list[object]]:
    """Plotly colorscale for signed values centred on 0 (use with zmid=0)."""
    mid = DIVERGING["mid_dark"] if (is_dark() if dark is None else dark) else DIVERGING["mid_light"]
    return [[0.0, DIVERGING["neg"]], [0.5, mid], [1.0, DIVERGING["pos"]]]


def sequential_scale() -> list[list[object]]:
    n = len(SEQUENTIAL_BLUE) - 1
    return [[i / n, c] for i, c in enumerate(SEQUENTIAL_BLUE)]


def style_figure(fig, *, dark: bool | None = None, height: int | None = None, hovermode: str | None = None):
    """Apply the dashboard's chart chrome to a Plotly figure (in place) and return it."""
    k = ink(dark)
    fig.update_layout(
        font={"family": FONT_FAMILY, "color": k.secondary, "size": 13},
        title={"font": {"color": k.primary, "size": 15}, "x": 0, "xanchor": "left"},
        paper_bgcolor=k.surface,
        plot_bgcolor=k.surface,
        margin={"l": 56, "r": 24, "t": 48, "b": 48},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0,
            "font": {"color": k.secondary},
            "bgcolor": "rgba(0,0,0,0)",
        },
        hoverlabel={"font": {"family": FONT_FAMILY}},
        colorway=series(dark),
    )
    axis = {
        "gridcolor": k.grid,
        "gridwidth": 1,
        "griddash": "solid",
        "zerolinecolor": k.axis,
        "linecolor": k.axis,
        "showline": True,
        "ticks": "outside",
        "tickcolor": k.axis,
        "tickfont": {"color": k.muted},
        "title": {"font": {"color": k.secondary}},
    }
    fig.update_xaxes(**axis)
    fig.update_yaxes(**axis)
    if height:
        fig.update_layout(height=height)
    if hovermode:
        fig.update_layout(hovermode=hovermode)
    fig.update_traces(selector={"type": "scatter"}, line={"width": 2}, marker={"size": 8})
    return fig
