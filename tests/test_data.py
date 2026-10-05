"""Dataset loading: snapshot integrity, ordering, split type guards, answer key, Hub fallback."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from driftlab import data
from driftlab.config import REPO_ROOT, ExperimentConfig, load_config
from driftlab.data import (
    DataIntegrityError,
    DevSplit,
    EvalSplit,
    Item,
    LeakageError,
    answer_key,
    canonical_gold,
    canonical_number,
    load_dev,
    load_eval,
    load_items,
    load_split,
    snapshot_info,
)

SNAPSHOT = REPO_ROOT / "data" / "gsm8k"


def _cfg(**data_over: object) -> ExperimentConfig:
    cfg = ExperimentConfig()
    return cfg.model_copy(update={"data": cfg.data.model_copy(update=data_over)})


def _with_n(cfg: ExperimentConfig, dev_n: int | None = None, eval_n: int | None = None) -> ExperimentConfig:
    d = cfg.data
    upd = {}
    if dev_n is not None:
        upd["dev"] = d.dev.model_copy(update={"n": dev_n})
    if eval_n is not None:
        upd["eval"] = d.eval.model_copy(update={"n": eval_n})
    return cfg.model_copy(update={"data": d.model_copy(update=upd)})


@pytest.fixture()
def snap_copy(tmp_path: Path) -> Path:
    dst = tmp_path / "gsm8k"
    shutil.copytree(SNAPSHOT, dst)
    return dst


# --------------------------------------------------------------------------- integrity


def test_snapshot_verifies_and_loads_both_splits() -> None:
    cfg = ExperimentConfig()
    dev, ev = load_dev(cfg), load_eval(cfg)
    assert isinstance(dev, DevSplit) and isinstance(ev, EvalSplit)
    assert len(dev) == len(ev) == 200
    assert dev.idxs() == list(range(200)) and ev.idxs() == list(range(200))
    assert dev[0].question.startswith("Natalia sold clips") and dev[0].gold == "72"
    assert ev[0].split == "test" and ev[0].gold == "18"
    info = snapshot_info(cfg)
    assert info["pins_match"] and info["snapshot_present"]
    assert all(f["ok"] for f in info["files"].values())
    assert info["revision"] == cfg.data.revision
    assert info["dev"]["source"] == info["eval"]["source"] == "snapshot"
    assert snapshot_info(_with_n(cfg, dev_n=250))["dev"]["source"] == "hub"


def test_tampered_copy_raises(snap_copy: Path) -> None:
    path = snap_copy / "train_first200.jsonl"
    text = path.read_text(encoding="utf-8")
    path.write_text(text.replace("Natalia", "Natasha", 1), encoding="utf-8")
    cfg = _cfg(snapshot_dir=str(snap_copy))
    with pytest.raises(DataIntegrityError, match="sha256"):
        load_dev(cfg)
    load_eval(cfg)  # the untouched split still verifies
    assert not snapshot_info(cfg)["files"]["train_first200.jsonl"]["ok"]


def test_untampered_copy_loads_identically(snap_copy: Path) -> None:
    assert load_dev(_cfg(snapshot_dir=str(snap_copy))) == load_dev(ExperimentConfig())


def test_revision_mismatch_raises() -> None:
    with pytest.raises(DataIntegrityError, match="revision"):
        load_items(_cfg(revision="0" * 40), "dev")


def test_manifest_revision_mismatch_raises(snap_copy: Path) -> None:
    manifest = json.loads((snap_copy / "snapshot.json").read_text())
    manifest["revision"] = "f" * 40
    (snap_copy / "snapshot.json").write_text(json.dumps(manifest))
    with pytest.raises(DataIntegrityError, match="revision"):
        load_eval(_cfg(snapshot_dir=str(snap_copy)))


def test_wrong_stored_gold_is_detected(snap_copy: Path) -> None:
    path = snap_copy / "test_first200.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    row = json.loads(lines[3])
    row["gold"] = str(int(row["gold"]) + 1)
    lines[3] = json.dumps(row)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest = json.loads((snap_copy / "snapshot.json").read_text())
    manifest["files"]["test_first200.jsonl"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    (snap_copy / "snapshot.json").write_text(json.dumps(manifest))
    with pytest.raises(DataIntegrityError, match="gold"):
        load_eval(_cfg(snapshot_dir=str(snap_copy)))


def test_corrupt_manifest_raises_but_snapshot_info_tolerates(snap_copy: Path) -> None:
    (snap_copy / "snapshot.json").write_text("{not json", encoding="utf-8")
    cfg = _cfg(snapshot_dir=str(snap_copy))
    with pytest.raises(DataIntegrityError, match="JSON"):
        load_dev(cfg)
    info = snapshot_info(cfg)  # provenance must never crash on a bad snapshot
    assert not info["snapshot_present"] and not info["pins_match"]
    assert not any(f["ok"] for f in info["files"].values())


def test_misconfigured_dev_split_is_refused() -> None:
    """A config pointing the dev split at the test file can never produce a DevSplit."""
    cfg = ExperimentConfig()
    cfg = cfg.model_copy(
        update={
            "data": cfg.data.model_copy(update={"dev": cfg.data.dev.model_copy(update={"split": "test"})})
        }
    )
    with pytest.raises(LeakageError):
        load_dev(cfg)


# --------------------------------------------------------------------------- ordering and types


def test_first_n_in_order() -> None:
    full = load_dev(ExperimentConfig())
    small = load_dev(load_config(REPO_ROOT / "configs" / "smoke_mock.yaml"))
    assert len(small) == 24
    assert list(small) == list(full)[:24]
    assert small.idxs() == list(range(24))
    assert small.by_idx(7) == full.by_idx(7)
    assert small.questions() == full.questions()[:24] and small.golds() == full.golds()[:24]
    ev3 = load_eval(_with_n(ExperimentConfig(), eval_n=3))
    assert ev3.idxs() == [0, 1, 2]


def test_load_split_dispatch() -> None:
    cfg = _with_n(ExperimentConfig(), dev_n=5, eval_n=5)
    assert isinstance(load_split(cfg, "dev"), DevSplit)
    assert isinstance(load_split(cfg, "eval"), EvalSplit)
    with pytest.raises(ValueError):
        load_split(cfg, "train")  # type: ignore[arg-type]


def test_devsplit_rejects_test_items() -> None:
    ev = load_eval(_with_n(ExperimentConfig(), eval_n=4))
    with pytest.raises(LeakageError):
        DevSplit(ev.items)
    mixed = (Item("train", 0, "q", "a #### 1", "1"), ev[0])
    with pytest.raises(LeakageError):
        DevSplit(mixed)
    with pytest.raises(LeakageError):
        EvalSplit((Item("train", 0, "q", "a #### 1", "1"),))
    assert issubclass(LeakageError, ValueError)


def test_split_types_are_distinct() -> None:
    cfg = _with_n(ExperimentConfig(), dev_n=3, eval_n=3)
    dev, ev = load_dev(cfg), load_eval(cfg)
    assert not isinstance(ev, DevSplit) and not isinstance(dev, EvalSplit)
    assert dev.split == "train" and ev.split == "test"
    with pytest.raises(KeyError):
        dev.by_idx(99)


def test_duplicate_indices_rejected() -> None:
    it = Item("train", 0, "q", "a #### 1", "1")
    with pytest.raises(ValueError, match="duplicate"):
        DevSplit((it, it))


def test_item_row_round_trip() -> None:
    it = load_dev(_with_n(ExperimentConfig(), dev_n=1))[0]
    row = it.to_row()
    assert Item(**row) == it
    assert set(row) == {"split", "idx", "question", "answer_text", "gold"}


# --------------------------------------------------------------------------- gold + answer key


@pytest.mark.parametrize(
    ("raw", "want"),
    [
        ("72", "72"),
        ("1,234", "1234"),
        (" 5.00 ", "5"),
        ("0.50", "0.5"),
        ("-3", "-3"),
        ("−3", "-3"),
        ("1e3", "1000"),
        ("12.250", "12.25"),
    ],
)
def test_canonical_number(raw: str, want: str) -> None:
    assert canonical_number(raw) == want


@pytest.mark.parametrize("raw", ["", "abc", "1/2", "nan", "inf"])
def test_canonical_number_rejects(raw: str) -> None:
    with pytest.raises(ValueError):
        canonical_number(raw)


def test_parse_gold() -> None:
    assert canonical_gold("Some work\n#### 1,000") == "1000"
    assert canonical_gold("a #### 2 then #### 3") == "3"
    with pytest.raises(ValueError):
        canonical_gold("no marker 5")


def test_all_400_golds_parse_and_match_extraction() -> None:
    cfg = ExperimentConfig()
    items = load_items(cfg, "dev") + load_items(cfg, "eval")
    assert len(items) == 400
    assert all(it.gold == canonical_gold(it.answer_text) for it in items)
    assert all(it.gold.lstrip("-").isdigit() for it in items)  # GSM8K golds are integers
    try:
        from driftlab.extraction import canonical
        from driftlab.extraction import parse_gold as ext_parse_gold
    except ImportError:  # pragma: no cover - extraction module not available
        return
    assert all(canonical(ext_parse_gold(it.answer_text)) == it.gold for it in items)


def test_answer_key_covers_both_splits() -> None:
    cfg = ExperimentConfig()
    key = answer_key(cfg)
    assert len(key) == 400
    dev, ev = load_dev(cfg), load_eval(cfg)
    for it in [*dev, *ev]:
        assert key[it.question] == it.gold  # default user_template is "{question}"


def test_answer_key_uses_user_template() -> None:
    cfg = _with_n(ExperimentConfig(), dev_n=2, eval_n=2)
    cfg = cfg.model_copy(
        update={"trajectory": cfg.trajectory.model_copy(update={"user_template": "Problem: {question}"})}
    )
    key = answer_key(cfg)
    assert len(key) == 4
    assert all(k.startswith("Problem: ") for k in key)
    assert data.user_message(cfg, "x") == "Problem: x"


# --------------------------------------------------------------------------- Hub fallback


def _write_parquet(path: Path, n: int) -> Path:
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    rows = [{"question": f"Question {i}?", "answer": f"Work {i}.\n#### {i * 7:,}"} for i in range(n)]
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


def _patch_hub(monkeypatch: pytest.MonkeyPatch, parquet: Path, calls: list) -> None:
    def fake(repo_id: str, filename: str, revision: str) -> str:
        calls.append((repo_id, filename, revision))
        return str(parquet)

    monkeypatch.setattr(data, "_hf_hub_download", fake)


def test_hub_fallback_when_n_exceeds_snapshot(
    tmp_path: Path, snap_copy: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parquet = _write_parquet(tmp_path / "hub" / "train.parquet", 300)
    manifest = json.loads((snap_copy / "snapshot.json").read_text())
    manifest["source_files"]["main/train-00000-of-00001.parquet"]["sha256"] = hashlib.sha256(
        parquet.read_bytes()
    ).hexdigest()
    (snap_copy / "snapshot.json").write_text(json.dumps(manifest))
    calls: list = []
    _patch_hub(monkeypatch, parquet, calls)
    cfg = _with_n(_cfg(snapshot_dir=str(snap_copy)), dev_n=250)
    dev = load_dev(cfg)
    assert len(dev) == 250 and dev.idxs() == list(range(250))
    assert dev.by_idx(249).gold == str(249 * 7)
    assert calls == [("openai/gsm8k", "main/train-00000-of-00001.parquet", cfg.data.revision)]


def test_hub_fallback_hash_mismatch_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    parquet = _write_parquet(tmp_path / "hub" / "train.parquet", 300)
    _patch_hub(monkeypatch, parquet, [])
    cfg = _with_n(ExperimentConfig(), dev_n=201)  # pinned source sha256 != our fake parquet
    with pytest.raises(DataIntegrityError, match="sha256"):
        load_dev(cfg)


def test_hub_fallback_when_snapshot_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    parquet = _write_parquet(tmp_path / "hub" / "test.parquet", 10)
    calls: list = []
    _patch_hub(monkeypatch, parquet, calls)
    cfg = _with_n(_cfg(snapshot_dir=str(tmp_path / "missing")), eval_n=5)
    ev = load_eval(cfg)
    assert len(ev) == 5 and ev[4].gold == "28"
    assert calls[0][1] == "main/test-00000-of-00001.parquet"
    assert not snapshot_info(cfg)["snapshot_present"]
    with pytest.raises(ValueError, match="only 10 rows"):
        load_eval(_with_n(cfg, eval_n=11))


def test_n_must_be_positive() -> None:
    with pytest.raises(ValueError):
        load_items(_with_n(ExperimentConfig(), dev_n=0), "dev")
