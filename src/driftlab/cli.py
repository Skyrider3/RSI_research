"""``driftlab`` command-line interface.

Subcommands: ``estimate | run | status | analyze | tables | demo | extract | audit | freeze-plan |
verify-extractors | dashboard | info``. Heavy modules (pandas, the pipeline, torch-backed backends) are
imported inside the command functions, so ``driftlab --help`` is fast and works without torch.

Exit codes: 0 success, 1 error (one-line message; set ``DRIFTLAB_DEBUG=1`` for a traceback), 2 usage error,
3 time budget exhausted (``run`` / ``demo`` / ``audit``: finished work is committed, re-run to continue).
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import os
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

EXIT_OK, EXIT_ERROR, EXIT_USAGE, EXIT_BUDGET = 0, 1, 2, 3
OPTIONAL_PACKAGES: tuple[str, ...] = ("torch", "transformers", "vllm", "streamlit")
TABLE_FORMATS: tuple[str, ...] = ("md", "csv", "tex")


class UserError(Exception):
    """An error caused by the invocation (bad path, refused operation): printed as one line, no traceback."""


def _repo_root() -> Path:
    from driftlab.config import REPO_ROOT

    return REPO_ROOT


def _print(*parts: object) -> None:
    print(*parts, flush=True)


def _one_line(e: BaseException) -> str:
    msg = " ".join(str(e).split())
    if len(msg) > 400:
        msg = msg[:397] + "..."
    if isinstance(e, UserError):
        return msg
    return f"{type(e).__name__}: {msg}" if msg else type(e).__name__


def _user_errors() -> tuple[type[BaseException], ...]:
    """Exception types reported as one-line user errors (invalid configs, paths, refused operations)."""
    import sqlite3

    errs: list[type[BaseException]] = [
        UserError,
        OSError,  # missing / unreadable files, read-only or full disks
        sqlite3.Error,  # e.g. "file is not a database", "database is locked"
        ValueError,  # includes pydantic.ValidationError
        KeyError,
        LookupError,
        RuntimeError,  # includes store.ConfigMismatch / EngineMismatch and backend 'auto' without a GPU
        ImportError,  # a missing optional dependency (vllm, transformers, ...)
    ]
    try:
        import yaml

        errs.append(yaml.YAMLError)
    except ImportError:  # pragma: no cover
        pass
    return tuple(errs)


def _existing_file(path: str | Path, what: str = "file") -> Path:
    p = Path(path).expanduser()
    if not p.is_file():
        raise UserError(f"{what} not found: {p}")
    return p


def _run_dir(path: str | Path, need: Sequence[str] = ("config.yaml",)) -> Path:
    p = Path(path).expanduser()
    if not p.is_dir():
        raise UserError(f"run directory not found: {p}")
    missing = [n for n in need if not (p / n).exists()]
    if missing:
        raise UserError(f"{p} is not a DriftLab run directory (missing {', '.join(missing)})")
    return p


def _split_list(
    raw: str | None, allowed: Sequence[str] | None = None, what: str = "value"
) -> list[str] | None:
    if raw is None:
        return None
    items = [s.strip() for s in raw.split(",") if s.strip()]
    if allowed is not None:
        bad = [s for s in items if s not in allowed]
        if bad:
            raise UserError(f"unknown {what}(s) {', '.join(bad)}; expected some of {', '.join(allowed)}")
    if not items:
        raise UserError(f"empty {what} list")
    return items


def _importable(name: str) -> bool:
    if name in sys.modules:
        return sys.modules[name] is not None
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


# --------------------------------------------------------------------------- estimate


def cmd_estimate(args: argparse.Namespace) -> int:
    from driftlab.config import load_config, load_plan
    from driftlab.estimate import estimate, format_estimate

    cfg = load_config(_existing_file(args.config, "config"), overrides=args.set)
    plan_path = cfg.resolve_path(cfg.analysis_plan)
    plan = load_plan(plan_path) if plan_path.is_file() else None
    backend = args.backend or (cfg.backend.kind if cfg.backend.kind != "auto" else "vllm")
    est = estimate(cfg, gpu=args.gpu, backend=backend, avg_completion_tokens=args.tokens, plan=plan)
    _print(format_estimate(est))
    return EXIT_OK


# --------------------------------------------------------------------------- run / status / audit


def _log(quiet: bool) -> Callable[[str], None]:
    return (lambda _m: None) if quiet else _print


def _print_run_result(res: dict[str, Any]) -> None:
    status = res.get("status", "?")
    _print(
        f"status: {status}   requested {int(res.get('requested', 0)):,}   executed {int(res.get('executed', 0)):,}"
        f"   complete: {'yes' if res.get('complete') else 'no'}"
    )
    if res.get("synthetic"):
        from driftlab.analysis.bundle_io import SYNTHETIC_NOTE

        _print(f"NOTE: {SYNTHETIC_NOTE}")
    for stage, summary in (res.get("stages") or {}).items():
        _print(f"  {stage:<10} {json.dumps(summary, default=str, sort_keys=True)}")
    if status == "budget_exhausted":
        _print(
            f"time budget exhausted in stage {res.get('stage', '?')}: finished work is committed; re-run to continue"
        )


def _pipeline_run(cfg: Any, run_dir: Path, stages: list[str] | None, args: argparse.Namespace) -> int:
    from driftlab.pipeline import Pipeline

    p = Pipeline(
        cfg,
        run_dir,
        allow_engine_change=bool(getattr(args, "allow_engine_change", False)),
        allow_config_change=bool(getattr(args, "allow_config_change", False)),
        log=_log(getattr(args, "quiet", False)),
    )
    try:
        p.open()
        res = p.run(stages, getattr(args, "max_minutes", None))
    finally:
        p.close()
    _print_run_result(res)
    tables = run_dir / "exports" / "tables" / "all_tables.md"
    if "analyze" in (res.get("stages") or {}) and tables.exists():
        _print(f"tables: {tables}")
    return EXIT_OK if res.get("status") == "ok" else EXIT_BUDGET


def cmd_run(args: argparse.Namespace) -> int:
    from driftlab.config import load_config

    cfg = load_config(_existing_file(args.config, "config"), overrides=args.set)
    stages = _split_list(args.stages, what="stage")
    return _pipeline_run(cfg, Path(args.run_dir).expanduser(), stages, args)


def cmd_audit(args: argparse.Namespace) -> int:
    from driftlab.config import load_config

    run_dir = _run_dir(args.run_dir, ("config.yaml", "store.sqlite"))
    cfg = load_config(run_dir / "config.yaml")
    return _pipeline_run(cfg, run_dir, ["audit"], args)


def format_status(st: dict[str, Any]) -> str:
    """Aligned plain-text rendering of ``Pipeline.status()``."""
    flag = "  [SYNTHETIC]" if st.get("synthetic") else ""
    comp = st.get("complete")
    lines = [
        f"run {st.get('run_id', '?')} at {st.get('run_dir', '?')}{flag}",
        f"complete: {'?' if comp is None else ('yes' if comp else 'no')}   requested "
        f"{int(st.get('requested', 0)):,}   executed {int(st.get('executed', 0)):,}",
    ]
    lines.append(f"{'stage':<11}{'planned':>10}{'done':>10}  complete")
    for name, s in (st.get("stages") or {}).items():
        planned = "" if s.get("planned") is None else f"{int(s['planned']):,}"
        done = "" if s.get("done") is None else f"{int(s['done']):,}"
        extra = []
        if "per_seed" in s:
            extra.append(f"per seed {s['per_seed']}")
        if s.get("unscored"):
            extra.append(f"{int(s['unscored']):,} unscored")
        if s.get("enabled") is False:
            extra.append("disabled")
        if s.get("error"):
            extra.append(f"error: {s['error']}")
        lines.append(
            f"{name:<11}{planned:>10}{done:>10}  {'yes' if s.get('complete') else 'no':<4}"
            + (f"  {'; '.join(extra)}" if extra else "")
        )
    return "\n".join(line.rstrip() for line in lines)


def cmd_status(args: argparse.Namespace) -> int:
    from driftlab.config import load_config
    from driftlab.pipeline import Pipeline

    run_dir = _run_dir(args.run_dir)
    cfg = load_config(run_dir / "config.yaml")
    # Never opened: status() of a closed pipeline reads the store read-only and builds no backend (no GPU / model
    # load, no rewrite of the run-dir files), whatever backend the run uses.
    p = Pipeline(cfg, run_dir, log=lambda _m: None)
    try:
        st = p.status()
    finally:
        p.close()
    _print(format_status(st))
    return EXIT_OK


# --------------------------------------------------------------------------- analyze / tables / demo


def cmd_analyze(args: argparse.Namespace) -> int:
    from driftlab.analysis.bundle import analyze, main_summary
    from driftlab.reporting.render import ALL_TABLES, write_tables

    run_dir = _run_dir(args.run_dir, ("config.yaml", "store.sqlite"))
    bundle = analyze(
        run_dir, B=args.B, log=_log(args.quiet), allow_stale_scores=bool(args.allow_stale_scores)
    )
    out = run_dir / "exports" / "tables"
    write_tables(bundle, out)
    _print(main_summary(bundle))
    _print(f"tables: {out / ALL_TABLES}")
    return EXIT_OK


def _load_bundle(run_dir: Path) -> Any:
    from driftlab.analysis.bundle_io import has_bundle, load_bundle

    if not has_bundle(run_dir):
        raise UserError(f"no analysis bundle in {run_dir}; run `driftlab analyze --run-dir {run_dir}` first")
    return load_bundle(run_dir)


def cmd_tables(args: argparse.Namespace) -> int:
    from driftlab.reporting.render import ALL_TABLES, ALL_TABLES_EXTENDED, write_tables

    run_dir = Path(args.run_dir).expanduser()
    if not run_dir.is_dir():
        raise UserError(f"run directory not found: {run_dir}")
    bundle = _load_bundle(run_dir)
    out = Path(args.out).expanduser() if args.out else run_dir / "exports" / "tables"
    formats = _split_list(args.formats, TABLE_FORMATS, "format") or list(TABLE_FORMATS)
    paths = write_tables(bundle, out, formats, extended=args.extended)
    _print(f"wrote {len(paths)} file(s) to {out}")
    if "md" in formats:
        _print(f"tables: {out / ALL_TABLES}")
        if args.extended:
            _print(f"extended: {out / ALL_TABLES_EXTENDED}")
    return EXIT_OK


def cmd_demo(args: argparse.Namespace) -> int:
    from driftlab.analysis.bundle import main_summary
    from driftlab.config import load_config
    from driftlab.reporting.render import render_markdown
    from driftlab.reporting.tables import build_tables

    root = _repo_root()
    name = "smoke_mock" if args.quick else "demo_mock"
    cfg_path = _existing_file(args.config or root / "configs" / f"{name}.yaml", "config")
    run_dir = Path(args.run_dir).expanduser() if args.run_dir else root / "runs" / name
    cfg = load_config(cfg_path)
    if cfg.backend.kind != "mock":
        _print(f"note: {cfg_path} uses backend '{cfg.backend.kind}', not the synthetic mock")
    _print(f"DriftLab demo: {cfg_path.name} -> {run_dir}")
    rc = _pipeline_run(cfg, run_dir, None, args)
    if rc != EXIT_OK:
        return rc
    bundle = _load_bundle(run_dir)
    tables = build_tables(bundle)
    _print("")
    _print(main_summary(bundle))
    for tid in ("T3", "T7"):
        _print("")
        _print(render_markdown(tables[tid]).rstrip())
    _print("")
    _print(f"all tables: {run_dir / 'exports' / 'tables' / 'all_tables.md'}")
    _print(f"dashboard:  driftlab dashboard --run-dir {run_dir}")
    return EXIT_OK


# --------------------------------------------------------------------------- extract / verify / freeze


def cmd_extract(args: argparse.Namespace) -> int:
    from driftlab.extraction import REGISTRY, extract, extractor_tag

    text = args.text
    if text is None and (sys.stdin is None or sys.stdin.isatty()):
        raise UserError(
            "give the response TEXT as an argument, or pipe it in on stdin (TEXT '-' reads stdin)"
        )
    if text is None or text == "-":
        text = sys.stdin.read()
    names = sorted(REGISTRY) if args.extractor == "all" else [args.extractor]
    for name in names:
        ex = extract(name, text)
        span = "—" if ex.span is None else f"[{ex.span[0]}, {ex.span[1]})"
        content = "—" if ex.content is None else repr(ex.content)
        _print(
            f"{name} ({extractor_tag(name)}): value={ex.extracted if ex.extracted is not None else '—'}  "
            f"method={ex.method}  span={span}  content={content}"
        )
    return EXIT_OK


def cmd_verify_extractors(args: argparse.Namespace) -> int:
    from driftlab.extraction import extractor_tags, verify_frozen

    mismatches = verify_frozen(args.pinned)
    if mismatches:
        _print(
            "extractor hashes differ from the pinned values (a behaviour change needs a NEW version module):"
        )
        for name, (want, have) in sorted(mismatches.items()):
            _print(f"  {name}: pinned {want or '—'}, current {have or '—'}")
        return EXIT_ERROR
    tags = extractor_tags()
    _print("extractors frozen: " + ", ".join(tags[k] for k in sorted(tags)))
    return EXIT_OK


def lock_path(plan_path: str | Path) -> Path:
    """``analysis_plans/prereg_v1.yaml`` -> ``analysis_plans/prereg_v1.lock.json``."""
    p = Path(plan_path)
    return p.with_name(p.stem + ".lock.json")


def freeze_plan(plan_path: str | Path, force: bool = False) -> tuple[dict[str, Any], bool]:
    """Write (or confirm) the plan's lock file; returns ``(record, written)``.

    Raises :class:`UserError` when a lock with a different plan hash or file sha256 exists (unless ``force``).
    """
    import hashlib
    from datetime import datetime, timezone

    import driftlab
    from driftlab.config import load_plan
    from driftlab.provenance import git_info

    path = _existing_file(plan_path, "analysis plan")
    plan = load_plan(path)
    digest = plan.plan_hash()
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    lock = lock_path(path)
    if lock.exists():
        try:
            old = json.loads(lock.read_text(encoding="utf-8"))
        except ValueError:
            old = {}
        if old.get("plan_hash") == digest and old.get("file_sha256") == sha:
            return old, False
        if not force:
            what = (
                "only the file bytes (comments / formatting)"
                if old.get("plan_hash") == digest
                else "the plan"
            )
            raise UserError(
                f"{lock} freezes plan hash {old.get('plan_hash')} (file sha256 {str(old.get('file_sha256'))[:12]}), "
                f"but {path} now has plan hash {digest} (file sha256 {sha[:12]}): {what} changed after freezing; "
                "refusing to re-freeze (use --force to override)"
            )
    git = git_info()
    record = {
        "plan_path": str(path),
        "plan_version": plan.version,
        "plan_hash": digest,
        "file_sha256": sha,
        "git_commit": git.get("commit"),
        "git_dirty": git.get("dirty"),
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "driftlab_version": driftlab.__version__,
    }
    tmp = lock.with_name(lock.name + ".tmp")
    tmp.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(lock)
    return record, True


def cmd_freeze_plan(args: argparse.Namespace) -> int:
    plan = Path(args.plan).expanduser()
    if not plan.is_absolute() and not plan.exists():
        plan = _repo_root() / plan
    record, written = freeze_plan(plan, force=args.force)
    lock = lock_path(plan)
    _print(f"plan_hash: {record['plan_hash']}")
    _print(f"file_sha256: {record['file_sha256']}")
    _print(f"lock: {lock} ({'written' if written else 'already frozen, unchanged'})")
    return EXIT_OK


# --------------------------------------------------------------------------- dashboard / info


def cmd_dashboard(args: argparse.Namespace) -> int:
    import subprocess

    home = _repo_root() / "app" / "Home.py"
    if not _importable("streamlit"):
        raise UserError("streamlit is not installed: pip install -e '.[ui]'")
    if not home.is_file():
        raise UserError(f"dashboard entry point not found: {home} (the dashboard has not been built yet)")
    env = dict(os.environ)
    if args.run_dir:
        run_dir = Path(args.run_dir).expanduser()
        if not run_dir.is_dir():
            raise UserError(f"run directory not found: {run_dir}")
        env["DRIFTLAB_RUN"] = str(run_dir.resolve())
    cmd = [sys.executable, "-m", "streamlit", "run", str(home), "--server.port", str(args.port)]
    _print("launching: " + " ".join(cmd))
    return int(subprocess.call(cmd, env=env, cwd=str(_repo_root())))


def _run_dirs(root: Path) -> list[tuple[str, str]]:
    """``(name, description)`` of run directories under ``root``."""
    out = []
    if not root.is_dir():
        return out
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        if not ((d / "store.sqlite").exists() or (d / "config.yaml").exists()):
            continue
        idx = d / "exports" / "analysis" / "bundle.json"
        desc = "no analysis bundle"
        if idx.exists():
            try:
                meta = json.loads(idx.read_text(encoding="utf-8")).get("meta", {})
                desc = f"bundle {meta.get('created_at', '?')}" + (
                    " [SYNTHETIC]" if meta.get("synthetic") else ""
                )
            except ValueError:
                desc = "unreadable bundle.json"
        out.append((d.name, desc))
    return out


def cmd_info(args: argparse.Namespace) -> int:
    import platform
    from importlib import metadata

    import driftlab

    root = _repo_root()
    _print(f"driftlab {driftlab.__version__}")
    _print(f"python {platform.python_version()} ({sys.executable})")
    _print(f"repo root: {root}")
    _print("optional packages:")
    for name in OPTIONAL_PACKAGES:
        if _importable(name):
            try:
                ver = metadata.version(name)
            except metadata.PackageNotFoundError:
                ver = "?"
            _print(f"  {name:<13} {ver}")
        else:
            _print(f"  {name:<13} not installed")
    if _importable("torch") and not args.no_cuda:
        try:
            import torch

            ok = bool(torch.cuda.is_available())
            dev = f" ({torch.cuda.get_device_name(0)})" if ok else ""
            _print(f"CUDA: {'available' + dev if ok else 'not available'}")
        except Exception as e:  # broken driver / partial torch build
            _print(f"CUDA: unknown ({type(e).__name__}: {e})")
    runs = _run_dirs(root / "runs")
    _print(f"runs ({root / 'runs'}):" + ("" if runs else " none"))
    for name, desc in runs:
        _print(f"  {name:<24} {desc}")
    return EXIT_OK


# --------------------------------------------------------------------------- parser / main


def build_parser() -> argparse.ArgumentParser:
    from driftlab import __version__

    p = argparse.ArgumentParser(
        prog="driftlab",
        description="DriftLab: environment drift, reference staleness and false promotions in self-improving "
        "LLM agents.",
        epilog="Exit codes: 0 ok, 1 error, 2 usage, 3 time budget exhausted (re-run to continue). "
        "Set DRIFTLAB_DEBUG=1 for tracebacks.",
    )
    p.add_argument("--version", action="version", version=f"driftlab {__version__}")
    sub = p.add_subparsers(dest="command", metavar="COMMAND")

    def add(name: str, func: Callable[[argparse.Namespace], int], help_: str) -> argparse.ArgumentParser:
        sp = sub.add_parser(name, help=help_, description=help_)
        sp.set_defaults(func=func)
        return sp

    def add_set(sp: argparse.ArgumentParser) -> None:
        sp.add_argument(
            "--set",
            action="append",
            default=[],
            metavar="KEY=VALUE",
            help="config override, e.g. run.rounds=4",
        )

    sp = add("estimate", cmd_estimate, "estimate request counts and wall time for a config")
    sp.add_argument("-c", "--config", required=True, help="experiment config (YAML)")
    sp.add_argument("--gpu", default="t4", choices=["t4", "l4", "a100", "cpu"])
    sp.add_argument("--backend", default=None, choices=["vllm", "hf", "openai_compat", "mock"])
    sp.add_argument("--tokens", type=float, default=300.0, help="average completion tokens per generation")
    add_set(sp)

    sp = add("run", cmd_run, "run (or resume) pipeline stages in a run directory (idempotent)")
    sp.add_argument("-c", "--config", required=True, help="experiment config (YAML)")
    sp.add_argument("--run-dir", required=True, help="run directory (created if missing)")
    sp.add_argument(
        "--stages", default=None, help="comma-separated subset of data,trajectory,matrix,score,audit,analyze"
    )
    sp.add_argument("--max-minutes", type=float, default=None, help="soft time budget; re-run to continue")
    add_set(sp)
    sp.add_argument(
        "--allow-engine-change", action="store_true", help="resume on a different engine (recorded)"
    )
    sp.add_argument(
        "--allow-config-change", action="store_true", help="resume with a changed config (recorded)"
    )
    sp.add_argument("-q", "--quiet", action="store_true", help="suppress progress messages")

    sp = add("status", cmd_status, "show per-stage progress of a run directory")
    sp.add_argument("--run-dir", required=True)

    sp = add("analyze", cmd_analyze, "build the analysis bundle and the paper tables of a run")
    sp.add_argument("--run-dir", required=True)
    sp.add_argument("--B", type=int, default=None, help="bootstrap replicates (default: the plan's B)")
    sp.add_argument(
        "--allow-stale-scores",
        action="store_true",
        help="analyse score rows computed by an older extractor source instead of refusing (every table is "
        "then marked STALE SCORES; normally re-score with `driftlab run ... --stages score`)",
    )
    sp.add_argument("-q", "--quiet", action="store_true", help="suppress progress messages")

    sp = add("tables", cmd_tables, "write the paper tables (md/csv/tex) from a run's analysis bundle")
    sp.add_argument("--run-dir", required=True)
    sp.add_argument("--out", default=None, help="output directory (default: <run-dir>/exports/tables)")
    sp.add_argument("--formats", default="md,csv,tex", help="comma-separated subset of md,csv,tex")
    sp.add_argument("--extended", action="store_true", help="also write the extended variants of every table")

    sp = add("demo", cmd_demo, "synthetic end-to-end demo (mock backend): run, analyze, print T3 and T7")
    sp.add_argument(
        "--quick", action="store_true", help="small smoke config (seconds) instead of the full demo"
    )
    sp.add_argument("--run-dir", default=None, help="default: runs/demo_mock (runs/smoke_mock with --quick)")
    sp.add_argument("--config", default=None, help="config to use instead of the demo config")
    sp.add_argument("-q", "--quiet", action="store_true", help="suppress progress messages")

    sp = add("extract", cmd_extract, "apply the frozen answer extractors to a response text")
    sp.add_argument("--extractor", default="all", choices=["v1", "v2", "all"])
    sp.add_argument("text", nargs="?", default=None, help="response text, or '-' / nothing to read stdin")

    sp = add("audit", cmd_audit, "run only the determinism audit stage of a run directory")
    sp.add_argument("--run-dir", required=True)
    sp.add_argument("--max-minutes", type=float, default=None)
    sp.add_argument("-q", "--quiet", action="store_true", help="suppress progress messages")

    sp = add(
        "freeze-plan", cmd_freeze_plan, "freeze the analysis plan: write <plan>.lock.json with its hashes"
    )
    sp.add_argument("--plan", default="analysis_plans/prereg_v1.yaml")
    sp.add_argument("--force", action="store_true", help="overwrite a lock that froze a different plan")

    sp = add(
        "verify-extractors",
        cmd_verify_extractors,
        "check the extractor source hashes against the pinned ones",
    )
    sp.add_argument(
        "--pinned", default=None, help="pinned hashes JSON (default: tests/extractors_frozen.json)"
    )

    sp = add("dashboard", cmd_dashboard, "launch the Streamlit dashboard")
    sp.add_argument("--run-dir", default=None, help="run directory to open (sets DRIFTLAB_RUN)")
    sp.add_argument("--port", type=int, default=8501)

    sp = add("info", cmd_info, "versions, optional packages, CUDA and known run directories")
    sp.add_argument("--no-cuda", action="store_true", help="skip the CUDA check (avoids importing torch)")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point of the ``driftlab`` console script; returns the exit code."""
    parser = build_parser()
    try:
        args = parser.parse_args(list(argv) if argv is not None else None)
    except SystemExit as e:  # --help / --version (0) or a usage error (2)
        return int(e.code) if isinstance(e.code, int) else EXIT_USAGE
    if not getattr(args, "command", None):
        parser.print_help()
        return EXIT_USAGE
    try:
        return int(args.func(args) or EXIT_OK)
    except KeyboardInterrupt:
        print("driftlab: interrupted", file=sys.stderr)
        return 130
    except BrokenPipeError:  # output piped into e.g. `head`: stop quietly (Python docs recipe)
        with contextlib.suppress(OSError, ValueError):
            os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        return EXIT_OK
    except _user_errors() as e:
        if os.environ.get("DRIFTLAB_DEBUG"):
            raise
        print(f"driftlab {args.command}: error: {_one_line(e)}", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
