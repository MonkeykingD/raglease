"""Independently assessed H100 132-SM Decode GPU-service component.

The offline acceptance covers observed serving batches, not complete Decode
iteration latency, an unobserved large-batch cold start, or the full bundle.
"""

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

from sglang.srt.multiplex.stage_predictor import CausalDecodeGuard


@dataclass(frozen=True)
class FrozenFullcardDecodeEnvelope:
    identity: tuple
    base: CausalDecodeGuard
    small_history_margin_ms: float
    small_cold_floor_ms: float
    large_history_margin_ms: float
    large_cold_floor_ms: float
    assessment_sha256: str
    large_history_quantile: float = 1.0
    ann_large_cold_floor_ms: float | None = None

    def __post_init__(self):
        if self.base.identity != self.identity:
            raise ValueError("Full-card Decode guard belongs to another deployment")
        if any(type(value) not in (int, float) or not math.isfinite(value) or value < 0
               for value in (self.small_history_margin_ms, self.small_cold_floor_ms,
                             self.large_history_margin_ms, self.large_cold_floor_ms)):
            raise ValueError("Finite nonnegative full-card Decode margins required")
        if (not math.isfinite(self.large_history_quantile)
                or not 0 < self.large_history_quantile <= 1):
            raise ValueError("Full-card Decode history quantile must be in (0, 1]")
        if (self.ann_large_cold_floor_ms is not None
                and (not math.isfinite(self.ann_large_cold_floor_ms)
                     or self.ann_large_cold_floor_ms < 0)):
            raise ValueError("Finite nonnegative ANN cold floor required")

    def predict_ms(self, sm_count, batch_size, overlap, history, *, expected_identity):
        if expected_identity != self.identity:
            raise ValueError("Full-card Decode envelope belongs to another deployment")
        if sm_count != 132 or overlap not in ("solo", "ann"):
            return None
        rows = tuple(history)
        if any(len(row) not in (2, 3) for row in rows):
            raise ValueError("Full-card Decode history must contain service and overlap")
        # The causal base was fitted on (service, overlap) pairs. The optional
        # third field records the completed batch size only for detecting a
        # first use of the large-batch band; do not change the base features.
        base_rows = tuple((service, coexec) for service, coexec, *_ in rows)
        baseline = self.base.predict_ms(batch_size, overlap, base_rows,
                                        expected_identity=expected_identity)
        if baseline is None:
            return None
        if batch_size <= 64:
            return (max(row[0] for row in rows) + self.small_history_margin_ms
                    if rows else self.small_cold_floor_ms)
        # Legacy two-field artifacts have no batch label and retain their
        # original semantics. New three-field runtime histories use only
        # completed large-band samples for the large-band history floor.
        large_rows = tuple(row for row in rows if len(row) == 2 or row[2] > 64)
        if large_rows:
            values = sorted(row[0] for row in large_rows)
            pos = (len(values) - 1) * self.large_history_quantile
            low = int(pos)
            high = min(low + 1, len(values) - 1)
            history_floor = (values[low] + (values[high] - values[low]) * (pos - low)
                             + self.large_history_margin_ms)
        else:
            history_floor = (self.ann_large_cold_floor_ms
                             if overlap == "ann" and self.ann_large_cold_floor_ms is not None
                             else self.large_cold_floor_ms)
        return max(baseline, history_floor)


def load_accepted_fullcard_decode_envelope(model_path, assessment_path, *,
                                           expected_model_sha256,
                                           expected_assessment_sha256,
                                           expected_identity,
                                           accepted_base,
                                           expected_model_schema=None):
    model_bytes = Path(model_path).read_bytes()
    assessment_bytes = Path(assessment_path).read_bytes()
    if (hashlib.sha256(model_bytes).hexdigest() != expected_model_sha256
            or hashlib.sha256(assessment_bytes).hexdigest() != expected_assessment_sha256):
        raise ValueError("H100 full-card Decode artifact hash mismatch")
    model = json.loads(model_bytes)
    assessment = json.loads(assessment_bytes)
    if expected_model_schema is not None and model.get("schema") != expected_model_schema:
        raise ValueError("H100 full-card Decode model schema differs from deployment")
    schema = model.get("schema")
    if schema == "raglease.h100-fullcard-decode-serving-envelope.v3":
        supported_model_sha = expected_model_sha256
        valid_status = model.get("status") == "development_only_requires_new_independent_trace"
    elif schema == "raglease.h100-fullcard-decode-serving-envelope.v4":
        supported_model_sha = model.get("prior_v3_sha256")
        valid_status = (
            model.get("status") == "engineering_cold_candidate_requires_new_trace"
            and model.get("observed_support_assessment_sha256")
            == expected_assessment_sha256
            and model.get("history_rule")
            == "last 50 completed same-stream observations, with batch size; large-band floor uses only completed batches >=65"
        )
    elif schema == "raglease.h100-fullcard-decode-serving-envelope.v5":
        supported_model_sha = model.get("prior_v3_sha256")
        valid_status = (
            model.get("status") == "engineering_quantile_candidate_requires_new_trace"
            and model.get("observed_support_assessment_sha256")
            == expected_assessment_sha256
            and model.get("history_rule")
            == "last 50 completed same-stream observations, with batch size; large-band floor uses completed batches >=65 and their fixed quantile"
            and model.get("large_history_quantile") == 0.75
            and model.get("large_history_margin_ms") == 0
            and model.get("ann_large_cold_floor_ms") == 140.)
    else:
        raise ValueError("Unknown H100 full-card Decode model version")
    if (model.get("formal_ready") is not False
            or not valid_status
            or model.get("fullcard_decode_sm") != 132
            or model.get("band_boundary") != 64
            or model.get("accepted_base_sha256")
            != "612a37c02270ca4f44b11e7759bb717184ce60527ae408d9eddc3a18feaa7cf2"
            or assessment.get("schema")
            != "raglease.h100-fullcard-decode-serving-holdout-assessment.v3"
            or assessment.get("model_sha256") != supported_model_sha
            or assessment.get("observed_support_accepted") is not True
            or assessment.get("support_coverage_sufficient") is not True
            or assessment.get("formal_ready") is not False
            or assessment.get("combined", {}).get("large_batch_cold_rows") != 0
            or accepted_base.identity != expected_identity):
        raise ValueError("Full-card Decode component lacks scoped independent acceptance")
    rules = assessment["rules"]
    combined = assessment["combined"]
    by_seed = assessment["by_seed"]
    if (set(by_seed) != {"9944", "9945"}
            or combined["fullcard_rows"] < rules["min_fullcard_rows_combined"]
            or combined["large_batch_rows"] < rules["min_large_batch_rows_combined"]
            or any(row["large_batch_rows"] < rules["min_large_batch_rows_each_seed"]
                   or row["overall_coverage"] < rules["min_overall_coverage_each_seed"]
                   for row in by_seed.values())
            or combined["overall_coverage"] < rules["min_overall_coverage"]
            or combined["large_batch_coverage"] < rules["min_large_batch_coverage"]
            or combined["false_safe_rate_at_20ms"] > rules["max_false_safe_rate_at_20ms"]
            or combined["median_overprediction_ratio"] > rules["max_median_overprediction_ratio"]
            or combined["p99_relative_underprediction"] > rules["max_p99_relative_underprediction"]):
        raise ValueError("Full-card Decode held-out gates do not pass")
    return FrozenFullcardDecodeEnvelope(
        expected_identity, accepted_base,
        model["small_history_margin_ms"], model["small_cold_floor_ms"],
        model["large_history_margin_ms"], model["large_cold_floor_ms"],
        expected_assessment_sha256,
        model.get("large_history_quantile", 1.0),
        model.get("ann_large_cold_floor_ms"))
