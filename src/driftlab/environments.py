"""Evaluation environments: decoding configuration x answer-extraction version.

An *environment* is everything outside the candidate prompt that can change a measured score:
the model checkpoint, the inference engine, the decoding parameters, the token cap and the
answer extractor. Environment versions are predefined and independent of candidate performance.

    E1 = greedy  + strict extractor v1   (baseline / unchanged control)
    E2 = T=0.2   + strict extractor v1   (decoding change)
    E3 = greedy  + lenient extractor v2  (answer-extraction change)
    E4 = T=0.2   + lenient extractor v2  (multiple changes)
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field

# Tracked environment components, in the order they are reported.
COMPONENTS: tuple[str, ...] = ("model", "engine", "decoding", "max_new_tokens", "extractor")


@dataclass(frozen=True)
class Decoding:
    """Fully explicit decoding parameters (never inherit a model's generation_config defaults)."""

    id: str
    temperature: float
    top_p: float = 1.0
    top_k: int = 0  # 0 = disabled
    repetition_penalty: float = 1.0
    max_new_tokens: int = 640

    @property
    def is_greedy(self) -> bool:
        return self.temperature == 0.0

    def params(self) -> dict:
        """Sampling parameters only (no id), used in cache keys and fingerprints."""
        d = asdict(self)
        d.pop("id")
        return d


@dataclass(frozen=True)
class Environment:
    id: str
    decoding: Decoding
    extractor: str  # registry name, e.g. "v1" / "v2"
    description: str = ""
    # Constant within a run but tracked so that a change (e.g. a resumed run on a new GPU) is detectable.
    model: str = ""  # "<model_id>@<revision>"
    engine: str = ""  # engine fingerprint

    def components(self, extractor_tags: Mapping[str, str] | None = None) -> dict[str, object]:
        ext = self.extractor
        if extractor_tags and ext in extractor_tags:
            ext = extractor_tags[ext]
        dec = self.decoding.params()
        max_new = dec.pop("max_new_tokens")
        return {
            "model": self.model,
            "engine": self.engine,
            "decoding": dec,
            "max_new_tokens": max_new,
            "extractor": ext,
        }

    def fingerprint(self, extractor_tags: Mapping[str, str] | None = None) -> str:
        blob = json.dumps(self.components(extractor_tags), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()[:16]


def diff(a: Environment, b: Environment, extractor_tags: Mapping[str, str] | None = None) -> list[str]:
    """Names of the tracked components that differ between two environments."""
    ca, cb = a.components(extractor_tags), b.components(extractor_tags)
    return [c for c in COMPONENTS if ca[c] != cb[c]]


@dataclass(frozen=True)
class EnvSchedule:
    """Predefined environment version per optimization round (piecewise constant).

    ``changes`` maps the first round of each segment to an environment id, e.g. {0: "E1", 4: "E2", 8: "E4"}.
    """

    changes: Mapping[int, str] = field(default_factory=lambda: {0: "E1"})

    def __post_init__(self) -> None:
        if 0 not in self.changes:
            raise ValueError("an environment schedule must define round 0")

    def env_at(self, round_: int) -> str:
        start = max(r for r in self.changes if r <= round_)
        return self.changes[start]

    def change_rounds(self) -> list[int]:
        """Rounds (>0) at which the environment id differs from the previous round."""
        rounds = sorted(self.changes)
        out = []
        for r in rounds[1:]:
            if self.changes[r] != self.env_at(r - 1):
                out.append(r)
        return out

    def as_list(self, n_rounds: int) -> list[str]:
        """Environment id for rounds 0..n_rounds (inclusive)."""
        return [self.env_at(t) for t in range(n_rounds + 1)]


DEFAULT_DECODINGS: dict[str, Decoding] = {
    "greedy": Decoding(id="greedy", temperature=0.0),
    "t02": Decoding(id="t02", temperature=0.2),
}

DEFAULT_ENVIRONMENTS: dict[str, tuple[str, str, str]] = {
    # id: (decoding id, extractor, description)
    "E1": ("greedy", "v1", "Unchanged environment (greedy decoding, strict extractor v1)"),
    "E2": ("t02", "v1", "Decoding change (temperature 0.2 sampling, strict extractor v1)"),
    "E3": ("greedy", "v2", "Answer-extraction change (greedy decoding, lenient extractor v2)"),
    "E4": ("t02", "v2", "Multiple changes (temperature 0.2 sampling + lenient extractor v2)"),
}


def build_environments(
    decodings: Mapping[str, Decoding] | None = None,
    spec: Mapping[str, tuple[str, str, str]] | None = None,
    model: str = "",
    engine: str = "",
) -> dict[str, Environment]:
    decodings = decodings or DEFAULT_DECODINGS
    spec = spec or DEFAULT_ENVIRONMENTS
    return {
        eid: Environment(
            id=eid, decoding=decodings[d], extractor=x, description=desc, model=model, engine=engine
        )
        for eid, (d, x, desc) in spec.items()
    }
