"""Validated H100 solo-Decode history envelope, separate from co-run guard.

The component assessment covers GPU device service on the frozen H100
WikiDPR deployment. It does not certify a complete predictor bundle or
request-level decode iteration latency.
"""
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

from sglang.srt.multiplex.stage_predictor import CausalDecodeGuard


@dataclass(frozen=True)
class FrozenSoloDecodeEnvelope:
    identity: tuple
    base: CausalDecodeGuard
    history_margin_by_sm: tuple
    cold_floor_by_sm: tuple
    assessment_sha256: str

    def __post_init__(self):
        if self.base.identity != self.identity:
            raise ValueError("Solo Decode guard belongs to another deployment")
        margins = dict(self.history_margin_by_sm)
        floors = dict(self.cold_floor_by_sm)
        if (set(margins) != set(floors) or not margins
                or any(type(sm) is not int or sm <= 0 or
                       type(value) not in (int, float) or
                       not math.isfinite(value) or value < 0
                       for sm, value in margins.items())
                or any(not isinstance(pair, tuple) or len(pair) != 2 or
                       any(type(value) not in (int, float) or
                           not math.isfinite(value) or value < 0
                           for value in pair)
                       for pair in floors.values())):
            raise ValueError("Valid complete per-SM solo Decode envelope required")

    def predict_ms(self, sm_count, batch_size, history, *, expected_identity):
        if expected_identity != self.identity:
            raise ValueError("Solo Decode envelope belongs to another deployment")
        if type(sm_count) is not int or sm_count not in dict(self.history_margin_by_sm):
            return None
        rows = tuple(history)
        baseline = self.base.predict_ms(batch_size, "solo", rows,
                                        expected_identity=expected_identity)
        if baseline is None:
            return None
        # The formal held-out used solo-only sequences. Mixed completed
        # histories take a conservative maximum over every known class; no
        # unfinished or future observation is read.
        if rows:
            previous = max(value for value, _ in rows)
            return max(baseline, previous + dict(self.history_margin_by_sm)[sm_count])
        small, large = dict(self.cold_floor_by_sm)[sm_count]
        return max(baseline, small if batch_size <= 32 else large)


def load_accepted_solo_decode_envelope(path, *, expected_sha256,
                                       expected_identity):
    payload = Path(path).read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != expected_sha256:
        raise ValueError("Solo Decode envelope assessment hash mismatch")
    artifact = json.loads(payload)
    assessment = artifact.get("assessment", {})
    if (artifact.get("schema") != "raglease.h100-solo-decode-envelope-assessment.v2"
            or artifact.get("component_accepted") is not True
            or artifact.get("formal_ready") is not False
            or assessment.get("accepted") is not True
            or assessment.get("evaluated_rows") != 1600
            or assessment.get("empirical_coverage", 0) < .98
            or assessment.get("false_safe_rate", 1) > .02):
        raise ValueError("Solo Decode envelope lacks independent component acceptance")
    model = artifact.get("model", {})
    identity = tuple(model.get("identity", ()))
    if identity != expected_identity:
        raise ValueError("Solo Decode envelope identity mismatch")
    if (tuple(model.get("feature_order", ())) != CausalDecodeGuard.feature_order
            or model.get("tau") != .99 or model.get("max_history") != 50):
        raise ValueError("Solo Decode envelope feature contract mismatch")
    base = CausalDecodeGuard(identity, tuple(model["coefficients"]),
                             model["tau"], model["max_history"])
    margins_raw = model.get("per_sm_history_margin_ms", {})
    floors_raw = model.get("per_sm_cold_floor_ms", {})
    if set(margins_raw) != {"24", "40", "56", "72"} or set(floors_raw) != set(margins_raw):
        raise ValueError("Expected four H100 Decode-SM profile cells")
    margins = tuple((int(sm), margins_raw[sm]) for sm in sorted(margins_raw, key=int))
    floors = tuple((int(sm), (floors_raw[sm]["small"], floors_raw[sm]["large"]))
                   for sm in sorted(floors_raw, key=int))
    return FrozenSoloDecodeEnvelope(identity, base, margins, floors, digest)
