"""Provenance: required keys, no torch import side effect, stable digest."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import types
from pathlib import Path

import pytest

from driftlab import provenance
from driftlab.config import REPO_ROOT, ExperimentConfig, load_config
from driftlab.provenance import (
    PACKAGES,
    collect_provenance,
    git_info,
    install_info,
    package_versions,
    provenance_digest,
)

REQUIRED = {
    "driftlab_version",
    "git",
    "python",
    "platform",
    "packages",
    "gpu",
    "dataset",
    "model",
    "backend",
    "extractors",
    "config_hash",
    "plan",
    "created_at",
}


@pytest.fixture(scope="module")
def cfg() -> ExperimentConfig:
    return load_config(REPO_ROOT / "configs" / "smoke_mock.yaml")


def test_keys_present(cfg: ExperimentConfig) -> None:
    prov = collect_provenance(cfg)
    assert set(prov) >= REQUIRED
    assert set(prov["git"]) == {"commit", "branch", "dirty"}
    assert set(prov["install"]) == {"editable", "url", "vcs_commit"}
    for name in ("numpy", "pandas", "pydantic", "torch", "transformers", "vllm", "accelerate", "streamlit"):
        assert name in prov["packages"]
    assert prov["packages"]["numpy"] is not None
    assert set(prov["gpu"]) == {"nvidia_smi", "torch_cuda"}
    assert prov["dataset"]["revision"] == cfg.data.revision and prov["dataset"]["pins_match"]
    assert prov["model"] == {
        "id": cfg.model.id,
        "revision": cfg.model.revision,
        "dtype": cfg.model.dtype,
        "max_new_tokens": cfg.model.max_new_tokens,
    }
    assert prov["config_hash"] == cfg.config_hash()
    assert prov["created_at"].endswith("+00:00")
    json.dumps(prov)  # JSON-serialisable


def test_extractor_tags_recorded(cfg: ExperimentConfig) -> None:
    ext = collect_provenance(cfg)["extractors"]
    if ext["tags"] is None:  # extraction module unavailable: tolerated, error recorded
        assert ext["error"]
    else:
        assert set(ext["tags"]) >= {"v1", "v2"}


def test_plan_hashes(cfg: ExperimentConfig) -> None:
    from driftlab.config import load_plan

    path = REPO_ROOT / cfg.analysis_plan
    plan = collect_provenance(cfg)["plan"]
    assert plan["file_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert plan["plan_hash"] == load_plan(path).plan_hash()
    missing = collect_provenance(cfg, plan_path="analysis_plans/does_not_exist.yaml")["plan"]
    assert missing["file_sha256"] is None and "error" in missing


def test_no_torch_import_side_effect(cfg: ExperimentConfig) -> None:
    if "torch" in sys.modules:
        pytest.skip("torch already imported by another test in this process (see subprocess test)")
    collect_provenance(cfg)
    assert "torch" not in sys.modules
    package_versions()
    assert "torch" not in sys.modules


def test_no_torch_import_side_effect_fresh_process() -> None:
    code = (
        "import sys\n"
        "from driftlab.config import ExperimentConfig\n"
        "from driftlab.provenance import collect_provenance\n"
        "assert 'torch' not in sys.modules\n"
        "collect_provenance(ExperimentConfig())\n"
        "print('torch' in sys.modules)\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120, check=True
    )
    assert out.stdout.strip() == "False"


def test_digest_stable_across_calls(cfg: ExperimentConfig) -> None:
    a, b = collect_provenance(cfg), collect_provenance(cfg)
    assert provenance_digest(a) == provenance_digest(b)
    assert len(provenance_digest(a)) == 16


def test_digest_ignores_volatile_fields(cfg: ExperimentConfig) -> None:
    prov = collect_provenance(cfg)
    d = provenance_digest(prov)
    changed = json.loads(json.dumps(prov))
    changed["created_at"] = "1999-01-01T00:00:00+00:00"
    changed["git"]["dirty"] = not prov["git"]["dirty"]
    assert provenance_digest(changed) == d
    assert prov["created_at"] != "1999-01-01T00:00:00+00:00"  # the input is not mutated
    changed["config_hash"] = "different"
    assert provenance_digest(changed) != d


def test_digest_tolerates_missing_fields() -> None:
    assert provenance_digest({"a": 1}) == provenance_digest({"a": 1, "created_at": "x"})


def test_backend_and_extra(cfg: ExperimentConfig) -> None:
    mock = pytest.importorskip("driftlab.backends.mock")
    backend = mock.MockBackend(cfg.model.id, cfg.model.revision, cfg.backend.mock)
    prov = collect_provenance(cfg, backend=backend, extra={"note": "unit test", "n": 3})
    assert prov["backend"]["kind"] == "mock" and prov["backend"]["synthetic"] is True
    assert prov["backend"]["engine_info"]["kind"] == "mock"
    assert len(prov["backend"]["engine_fp"]) == 16
    assert prov["extra"] == {"note": "unit test", "n": 3}
    plain = collect_provenance(cfg)
    assert plain["backend"]["engine_info"] is None and "extra" not in plain
    assert provenance_digest(prov) != provenance_digest(plain)


def test_git_info_requires_repo_top_level(tmp_path: Path) -> None:
    """Regression: a non-editable install lives in site-packages; git run there may find an unrelated
    enclosing repository. Only the work-tree top level counts."""
    assert git_info(REPO_ROOT / "src")["commit"] is None
    assert git_info(tmp_path) == {"commit": None, "branch": None, "dirty": None}
    top = git_info(REPO_ROOT)
    if top["commit"] is not None:  # git available in this environment
        assert len(top["commit"]) == 40 and isinstance(top["dirty"], bool)


def test_install_info_and_credential_stripping() -> None:
    info = install_info()
    assert set(info) == {"editable", "url", "vcs_commit"}
    assert install_info("definitely-not-a-real-package-xyz") == {
        "editable": None,
        "url": None,
        "vcs_commit": None,
    }
    strip = provenance._strip_credentials
    assert strip("git+https://user:tok@github.com/o/r.git") == "git+https://github.com/o/r.git"
    assert strip("https://tok@example.org:8443/x") == "https://example.org:8443/x"
    assert strip("file:///home/user/repo") == "file:///home/user/repo"
    assert strip(None) is None


def test_extractor_failure_never_breaks_provenance(
    cfg: ExperimentConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Broken(types.ModuleType):
        def __getattr__(self, name: str) -> object:
            raise RuntimeError("frozen extractor hash mismatch")

    monkeypatch.setitem(sys.modules, "driftlab.extraction", Broken("driftlab.extraction"))
    ext = collect_provenance(cfg)["extractors"]
    assert ext["tags"] is None and "frozen extractor hash mismatch" in ext["error"]
    monkeypatch.setitem(sys.modules, "driftlab.extraction", None)  # import itself fails
    ext = collect_provenance(cfg)["extractors"]
    assert ext["tags"] is None and ext["error"]


def test_packages_list_covers_spec() -> None:
    spec = {"numpy", "pandas", "pydantic", "torch", "transformers", "vllm", "accelerate", "streamlit"}
    assert spec | {"plotly", "httpx"} <= set(PACKAGES)
    assert package_versions(("definitely-not-a-real-package-xyz",)) == {
        "definitely-not-a-real-package-xyz": None
    }
