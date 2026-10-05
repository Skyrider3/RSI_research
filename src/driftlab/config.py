"""Experiment configuration (YAML -> pydantic) and the pre-registered analysis plan.

Configs may inherit from another file with ``extends: other.yaml`` (path relative to the config file);
mappings are deep-merged, lists and scalars are replaced. ``--set a.b.c=value`` overrides are applied last.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from driftlab.environments import Decoding, Environment, EnvSchedule
from driftlab.keys import sha256_json

REPO_ROOT = Path(__file__).resolve().parents[2]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------------------------------- experiment config


class RunSection(_Model):
    name: str = "pilot"
    seeds: list[int] = Field(default_factory=lambda: [0, 1, 2])
    rounds: int = 11  # optimization rounds R; slots 0..R


class ModelSection(_Model):
    id: str = "Qwen/Qwen2.5-1.5B-Instruct"
    revision: str = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
    dtype: str = "auto"  # auto -> bf16 on sm>=80, fp16 otherwise (CPU: fp32)
    max_new_tokens: int = 640


class HFSection(_Model):
    batch_size: int = 32
    attn_impl: str = "sdpa"
    device: str = "auto"  # auto | cuda | cpu


class VLLMSection(_Model):
    gpu_memory_utilization: float = 0.85
    max_model_len: int = 4096
    enable_prefix_caching: bool = False
    max_num_seqs: int = 256
    enforce_eager: bool = False


class OpenAICompatSection(_Model):
    base_url: str | None = None
    api_key_env: str = "OPENAI_API_KEY"
    concurrency: int = 16
    timeout_s: float = 600.0
    served_model_name: str | None = None  # defaults to model.id


class MockSection(_Model):
    base_quality: float = 0.55  # baseline logit of solving an item
    sample_sd: float = 0.6  # reasoning noise scale per unit temperature
    greedy_flip_rate: float = 0.0  # emulated GPU nondeterminism for physical greedy reruns
    malformed_proposal_rate: float = 0.1


class BackendSection(_Model):
    kind: Literal["auto", "mock", "hf", "vllm", "openai_compat"] = "auto"
    hf: HFSection = Field(default_factory=HFSection)
    vllm: VLLMSection = Field(default_factory=VLLMSection)
    openai_compat: OpenAICompatSection = Field(default_factory=OpenAICompatSection)
    mock: MockSection = Field(default_factory=MockSection)


class SplitSection(_Model):
    split: str
    n: int = 200


class DataSection(_Model):
    dataset: str = "openai/gsm8k"
    config: str = "main"
    revision: str = "740312add88f781978c0658806c59bc2815b9866"
    snapshot_dir: str = "data/gsm8k"  # committed first-200 snapshot (relative to repo root)
    dev: SplitSection = Field(default_factory=lambda: SplitSection(split="train", n=200))
    eval: SplitSection = Field(default_factory=lambda: SplitSection(split="test", n=200))


class DecodingSection(_Model):
    temperature: float
    top_p: float = 1.0
    top_k: int = 0
    repetition_penalty: float = 1.0


class EnvSection(_Model):
    decoding: str
    extractor: str
    description: str = ""


class ProposerSection(_Model):
    temperature: float = 0.7
    top_p: float = 1.0
    n_errors: int = 5
    max_attempts: int = 3
    max_new_tokens: int = 640
    response_head_chars: int = 300
    response_tail_chars: int = 300
    template: str = "meta_v1.txt"
    require_format_instruction: bool = True
    min_chars: int = 40
    max_chars: int = 1500


class TrajectorySection(_Model):
    mode: Literal["dev_gated", "static"] = "dev_gated"
    env: str = "E1"  # environment used for dev runs while building the trajectory
    advance_rule: Literal["dev_gt", "dev_ge", "always"] = "dev_gt"
    initial_prompt_file: str = "initial_v1.txt"
    user_template: str = "{question}"
    proposer: ProposerSection = Field(default_factory=ProposerSection)


class MatrixSection(_Model):
    mode: Literal["full", "lean"] = "full"
    physical_greedy_reruns: Literal["none", "ages", "all"] = "ages"
    physical_ages: list[int] = Field(default_factory=lambda: [0, 1, 3, 5, 10])
    gt_draws: int = 1
    chunk_size: dict[str, int] = Field(
        default_factory=lambda: {"vllm": 2048, "hf": 256, "mock": 5000, "openai_compat": 512}
    )


class AuditSection(_Model):
    enabled: bool = True
    seed: int = 0  # which run seed's slots are audited
    slots: list[str] = Field(default_factory=lambda: ["first", "last_incumbent"])
    repeats: int = 2
    decodings: list[str] = Field(default_factory=lambda: ["greedy", "t02"])
    n_items: int = 200


class CheckpointSection(_Model):
    drive_dir: str | None = None
    snapshot_every_minutes: float = 20.0


class ExperimentConfig(_Model):
    run: RunSection = Field(default_factory=RunSection)
    model: ModelSection = Field(default_factory=ModelSection)
    backend: BackendSection = Field(default_factory=BackendSection)
    data: DataSection = Field(default_factory=DataSection)
    decodings: dict[str, DecodingSection] = Field(
        default_factory=lambda: {
            "greedy": DecodingSection(temperature=0.0),
            "t02": DecodingSection(temperature=0.2),
        }
    )
    environments: dict[str, EnvSection] = Field(
        default_factory=lambda: {
            "E1": EnvSection(decoding="greedy", extractor="v1", description="Unchanged environment"),
            "E2": EnvSection(decoding="t02", extractor="v1", description="Decoding change"),
            "E3": EnvSection(decoding="greedy", extractor="v2", description="Answer-extraction change"),
            "E4": EnvSection(decoding="t02", extractor="v2", description="Multiple environment changes"),
        }
    )
    trajectory: TrajectorySection = Field(default_factory=TrajectorySection)
    matrix: MatrixSection = Field(default_factory=MatrixSection)
    audit: AuditSection = Field(default_factory=AuditSection)
    checkpoint: CheckpointSection = Field(default_factory=CheckpointSection)
    analysis_plan: str = "analysis_plans/prereg_v1.yaml"
    smoke: bool = False  # True for configs that deviate from the protocol (e.g. max_new_tokens < 640)

    @model_validator(mode="after")
    def _check_refs(self) -> ExperimentConfig:
        for eid, e in self.environments.items():
            if e.decoding not in self.decodings:
                raise ValueError(f"environment {eid} references unknown decoding {e.decoding!r}")
        if self.trajectory.env not in self.environments:
            raise ValueError(f"trajectory.env {self.trajectory.env!r} is not a defined environment")
        if self.model.max_new_tokens != 640 and not self.smoke:
            raise ValueError("the protocol caps responses at 640 new tokens; set smoke: true to deviate")
        return self

    # -- helpers --------------------------------------------------------------------------
    def decoding(self, decoding_id: str) -> Decoding:
        d = self.decodings[decoding_id]
        return Decoding(
            id=decoding_id,
            temperature=d.temperature,
            top_p=d.top_p,
            top_k=d.top_k,
            repetition_penalty=d.repetition_penalty,
            max_new_tokens=self.model.max_new_tokens,
        )

    def proposer_decoding(self) -> Decoding:
        p = self.trajectory.proposer
        return Decoding(
            id="proposer", temperature=p.temperature, top_p=p.top_p, max_new_tokens=p.max_new_tokens
        )

    def environment(self, env_id: str, engine_fp: str = "") -> Environment:
        e = self.environments[env_id]
        return Environment(
            id=env_id,
            decoding=self.decoding(e.decoding),
            extractor=e.extractor,
            description=e.description,
            model=f"{self.model.id}@{self.model.revision}",
            engine=engine_fp,
        )

    def all_environments(self, engine_fp: str = "") -> dict[str, Environment]:
        return {eid: self.environment(eid, engine_fp) for eid in self.environments}

    def config_hash(self) -> str:
        return sha256_json(self.model_dump(mode="json"))[:16]

    def resolve_path(self, p: str) -> Path:
        path = Path(p)
        return path if path.is_absolute() else REPO_ROOT / path


# --------------------------------------------------------------------------- analysis plan


class PromotionRule(_Model):
    kind: Literal["net_win", "win_rate", "mcnemar"] = "net_win"
    margin: float = 0.0  # net_win: accept iff (wins - losses) / n > margin
    tau: float = 0.05  # win_rate: accept iff wins / n >= tau and wins > losses
    alpha: float = 0.10  # mcnemar: one-sided exact binomial p < alpha and wins > losses


class GTSection(_Model):
    mode: Literal["independent_draw", "same_draw", "split_half"] = "independent_draw"
    canonical_env: str = "E1"


class T4Section(_Model):
    age: int = 3


class T7Section(_Model):
    ages: list[int] = Field(default_factory=lambda: [0, 1, 3, 5, 10])
    unchanged_env: str = "E1"
    changed_env: str = "E4"


class AblationSection(_Model):
    schedule: dict[int, str]
    fixed_age: int | None = None
    description: str = ""


class BootstrapSection(_Model):
    B: int = 2000
    seed: int = 20261004


class ScheduleRandomization(_Model):
    n: int = 200
    n_changes: int = 2
    seed: int = 7


class AnalysisPlan(_Model):
    version: str = "prereg_v1"
    reference_mode: Literal["incumbent", "chain"] = "incumbent"
    storage_env: str = "E1"
    promotion_rule: PromotionRule = Field(default_factory=PromotionRule)
    gt: GTSection = Field(default_factory=GTSection)
    schedule: dict[int, str] = Field(default_factory=lambda: {0: "E1", 4: "E2", 8: "E4"})
    policies: list[str] = Field(default_factory=lambda: ["P1", "P1b", "P2", "P3", "P4_k3", "P5", "ORACLE"])
    headline_policies: list[str] = Field(default_factory=lambda: ["P1", "P2", "P3"])
    cost_convention: Literal["teammate_v1", "full"] = "teammate_v1"
    t4: T4Section = Field(default_factory=T4Section)
    t7: T7Section = Field(default_factory=T7Section)
    ablations: dict[str, AblationSection] = Field(default_factory=dict)
    primary_contrasts: dict[str, str] = Field(default_factory=dict)
    h3_equivalence_margin_pp: float = 10.0
    bootstrap: BootstrapSection = Field(default_factory=BootstrapSection)
    schedule_randomization: ScheduleRandomization = Field(default_factory=ScheduleRandomization)
    notes: str = ""

    @field_validator("schedule")
    @classmethod
    def _sched_has_zero(cls, v: dict[int, str]) -> dict[int, str]:
        if 0 not in v:
            raise ValueError("schedule must define round 0")
        return v

    def env_schedule(self) -> EnvSchedule:
        return EnvSchedule(dict(self.schedule))

    def plan_hash(self) -> str:
        return sha256_json(self.model_dump(mode="json"))[:16]


# --------------------------------------------------------------------------- loading


def _deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _load_yaml_with_extends(path: Path, _seen: set[Path] | None = None) -> dict:
    path = path.resolve()
    _seen = _seen or set()
    if path in _seen:
        raise ValueError(f"circular 'extends' chain at {path}")
    _seen.add(path)
    data = yaml.safe_load(path.read_text()) or {}
    parent = data.pop("extends", None)
    if parent:
        base = _load_yaml_with_extends((path.parent / parent), _seen)
        data = _deep_merge(base, data)
    return data


def _apply_override(data: dict, assignment: str) -> None:
    if "=" not in assignment:
        raise ValueError(f"override must look like a.b.c=value, got {assignment!r}")
    dotted, raw = assignment.split("=", 1)
    value: Any = yaml.safe_load(raw)
    keys = dotted.split(".")
    cur = data
    for k in keys[:-1]:
        cur = cur.setdefault(k, {})
    cur[keys[-1]] = value


def load_config(path: str | Path, overrides: list[str] | None = None) -> ExperimentConfig:
    data = _load_yaml_with_extends(Path(path))
    for o in overrides or []:
        _apply_override(data, o)
    return ExperimentConfig.model_validate(data)


def load_plan(path: str | Path) -> AnalysisPlan:
    data = _load_yaml_with_extends(Path(path))
    return AnalysisPlan.model_validate(data)


def dump_yaml(model: BaseModel) -> str:
    return yaml.safe_dump(model.model_dump(mode="json"), sort_keys=False)
