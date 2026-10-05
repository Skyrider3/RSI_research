"""Run provenance: code, environment, data, model, engine, extractor and plan identifiers.

Provenance is metadata stored in ``runs.provenance_json``. It is never hashed into results, so it may contain
the wall-clock ``created_at``. :func:`provenance_digest` hashes it without the volatile fields.

Never imports torch: GPU details come from ``nvidia-smi`` and, only if torch is ALREADY imported (e.g. by
the HF / vLLM backend), from ``torch.cuda``.
"""

from __future__ import annotations

import copy
import hashlib
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import driftlab
from driftlab.config import REPO_ROOT, ExperimentConfig, load_plan
from driftlab.data import snapshot_info
from driftlab.keys import engine_fingerprint, sha256_json

PACKAGES: tuple[str, ...] = (
    "numpy",
    "pandas",
    "pydantic",
    "pyyaml",
    "torch",
    "transformers",
    "vllm",
    "accelerate",
    "streamlit",
    "plotly",
    "httpx",
    "huggingface_hub",
    "pyarrow",
)
# (dotted path) fields removed before hashing: they change without the run changing.
VOLATILE_FIELDS: tuple[tuple[str, ...], ...] = (("created_at",), ("git", "dirty"))
_SUBPROCESS_TIMEOUT_S = 10.0


def _run(cmd: list[str], cwd: Path | None = None) -> str | None:
    """stdout of a command (stripped), or ``None`` if it is unavailable or fails."""
    if shutil.which(cmd[0]) is None:
        return None
    try:
        out = subprocess.run(
            cmd, cwd=cwd, capture_output=True, text=True, timeout=_SUBPROCESS_TIMEOUT_S, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def git_info(repo: Path = REPO_ROOT) -> dict[str, Any]:
    """Commit, branch and dirty flag of the checkout (``None`` values when git or the repo is absent).

    ``repo`` must be the top level of the work tree: a non-editable install lives in site-packages, which may
    sit inside some unrelated repository whose commit must not be recorded as driftlab's.
    """
    none = {"commit": None, "branch": None, "dirty": None}
    top = _run(["git", "rev-parse", "--show-toplevel"], repo)
    if top is None or Path(top).resolve() != Path(repo).resolve():
        return none
    commit = _run(["git", "rev-parse", "HEAD"], repo)
    if commit is None:
        return none
    branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], repo)
    status = _run(["git", "status", "--porcelain", "--untracked-files=no"], repo)
    return {"commit": commit, "branch": branch, "dirty": None if status is None else bool(status)}


def _strip_credentials(url: str | None) -> str | None:
    if not url:
        return url
    parts = urlsplit(url)
    if parts.username is None and parts.password is None:
        return url
    host = parts.hostname or ""
    host = f"[{host}]" if ":" in host else host  # IPv6 literal
    netloc = f"{host}:{parts.port}" if parts.port else host
    return urlunsplit(parts._replace(netloc=netloc))


def install_info(dist: str = "driftlab") -> dict[str, Any]:
    """How the package was installed (PEP 610 ``direct_url.json``): editable flag, source URL (credentials
    stripped) and VCS commit. Records the code version when there is no git checkout, e.g. on Colab after
    ``pip install git+...@<commit>``."""
    out: dict[str, Any] = {"editable": None, "url": None, "vcs_commit": None}
    try:
        raw = metadata.distribution(dist).read_text("direct_url.json")
    except metadata.PackageNotFoundError:
        return out
    if not raw:  # installed from an index / wheel: no direct URL recorded
        return out
    try:
        info = json.loads(raw)
    except json.JSONDecodeError:
        return out
    out["editable"] = bool((info.get("dir_info") or {}).get("editable", False))
    out["url"] = _strip_credentials(info.get("url"))
    out["vcs_commit"] = (info.get("vcs_info") or {}).get("commit_id")
    return out


def package_versions(names: tuple[str, ...] = PACKAGES) -> dict[str, str | None]:
    """Installed distribution versions (``None`` if not installed); never imports the packages."""
    out: dict[str, str | None] = {}
    for name in names:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def gpu_info() -> dict[str, Any]:
    """``nvidia-smi`` rows plus torch.cuda details if (and only if) torch is already imported."""
    smi = None
    raw = _run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total,compute_cap",
            "--format=csv,noheader,nounits",
        ]
    ) or _run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader,nounits"])
    if raw:
        fields = ("name", "driver_version", "memory_total_mib", "compute_cap")
        smi = [
            dict(zip(fields, (c.strip() for c in line.split(",")), strict=False)) for line in raw.splitlines()
        ]
    torch_cuda = None
    torch = sys.modules.get("torch")
    if torch is not None:
        try:
            available = bool(torch.cuda.is_available())
            torch_cuda = {"available": available, "cuda": getattr(torch.version, "cuda", None)}
            if available:
                torch_cuda["device_name"] = torch.cuda.get_device_name(0)
                torch_cuda["capability"] = list(torch.cuda.get_device_capability(0))
                torch_cuda["device_count"] = int(torch.cuda.device_count())
        except Exception as e:  # broken driver / partial torch build
            torch_cuda = {"error": repr(e)}
    return {"nvidia_smi": smi, "torch_cuda": torch_cuda}


def _json_safe(obj: Any) -> Any:
    return json.loads(json.dumps(obj, sort_keys=True, default=str))


def _extractor_tags() -> dict[str, Any]:
    try:
        from driftlab.extraction import extractor_tags
    except Exception as e:  # module under construction / broken install: record, never fail provenance
        return {"tags": None, "error": repr(e)}
    try:
        return {"tags": dict(extractor_tags()), "error": None}
    except Exception as e:  # pragma: no cover - defensive
        return {"tags": None, "error": repr(e)}


def _plan_info(cfg: ExperimentConfig, plan_path: str | None) -> dict[str, Any]:
    path = cfg.resolve_path(plan_path or cfg.analysis_plan)
    info: dict[str, Any] = {
        "path": str(plan_path or cfg.analysis_plan),
        "file_sha256": None,
        "plan_hash": None,
    }
    if not path.is_file():
        info["error"] = f"plan file not found: {path}"
        return info
    info["file_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    try:
        plan = load_plan(path)
        info["plan_hash"] = plan.plan_hash()
        info["version"] = plan.version
    except Exception as e:  # invalid plan: record, do not fail provenance
        info["error"] = repr(e)
    return info


def _backend_info(cfg: ExperimentConfig, backend: Any) -> dict[str, Any]:
    if backend is None:
        return {"kind": cfg.backend.kind, "configured_kind": cfg.backend.kind, "engine_info": None}
    info: dict[str, Any] = {"kind": getattr(backend, "kind", type(backend).__name__)}
    info["configured_kind"] = cfg.backend.kind
    info["synthetic"] = bool(getattr(backend, "synthetic", False))
    try:
        engine = backend.engine_info()
        info["engine_info"] = _json_safe(engine)
        info["engine_fp"] = engine_fingerprint(engine)
    except Exception as e:
        info["engine_info"] = None
        info["error"] = repr(e)
    return info


def collect_provenance(
    cfg: ExperimentConfig,
    backend: Any = None,
    plan_path: str | None = None,
    extra: dict | None = None,
) -> dict[str, Any]:
    """JSON-serialisable provenance record of a run (see module docstring)."""
    prov: dict[str, Any] = {
        "driftlab_version": driftlab.__version__,
        "git": git_info(),
        "install": install_info(),
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "platform": platform.platform(),
        },
        "packages": package_versions(),
        "gpu": gpu_info(),
        "dataset": snapshot_info(cfg),
        "model": {
            "id": cfg.model.id,
            "revision": cfg.model.revision,
            "dtype": cfg.model.dtype,
            "max_new_tokens": cfg.model.max_new_tokens,
        },
        "backend": _backend_info(cfg, backend),
        "extractors": _extractor_tags(),
        "config_hash": cfg.config_hash(),
        "run_name": cfg.run.name,
        "smoke": cfg.smoke,
        "plan": _plan_info(cfg, plan_path),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if extra:
        prov["extra"] = _json_safe(extra)
    return prov


def provenance_digest(prov: dict[str, Any]) -> str:
    """16-hex digest of a provenance record without its volatile fields (``created_at``, ``git.dirty``)."""
    p = copy.deepcopy(prov)
    for path in VOLATILE_FIELDS:
        node: Any = p
        for key in path[:-1]:
            node = node.get(key) if isinstance(node, dict) else None
        if isinstance(node, dict):
            node.pop(path[-1], None)
    return sha256_json(p)[:16]


__all__ = [
    "PACKAGES",
    "VOLATILE_FIELDS",
    "collect_provenance",
    "git_info",
    "gpu_info",
    "install_info",
    "package_versions",
    "provenance_digest",
]
