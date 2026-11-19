"""Engineering-only H100 online split-Prefill service-boundary correction.

The accepted solo stage model predicts full-model GPU service. An online
ScheduleBatch bundle has additional measured service time approximately
proportional to requests times executed layers. This wrapper keeps every
non-Prefill model query unchanged and adds that term *before* layer scaling,
so the compute chooser and execution budget query the same adjusted model.
"""

import hashlib
import json
import math
from pathlib import Path


class OnlinePrefillStage:
    def __init__(self, base, *, identity, total_layers, alpha_by_sm):
        if (base.identity != identity or type(total_layers) is not int
                or total_layers <= 0 or set(alpha_by_sm) != {56, 72, 88, 104, 132}
                or any(type(v) not in (int, float) or not math.isfinite(v)
                       or v <= 0 for v in alpha_by_sm.values())):
            raise ValueError("Invalid online Prefill service calibration")
        self.base = base
        self.identity = identity
        self.total_layers = total_layers
        self.alpha_by_sm = dict(alpha_by_sm)

    def __getattr__(self, name):
        return getattr(self.base, name)

    def _correct(self, base_ms, sm_count, new_tokens):
        if base_ms is None:
            return None
        alpha = self.alpha_by_sm.get(sm_count)
        if alpha is None:
            return None
        return base_ms + alpha * len(new_tokens) * self.total_layers

    def predict_prefill_ms(self, sm_count, new_tokens, cached_tokens, *, expected_identity):
        base_ms = self.base.predict_prefill_ms(
            sm_count, new_tokens, cached_tokens, expected_identity=expected_identity)
        return self._correct(base_ms, sm_count, new_tokens)

    def predict_prefill_nearest_ms(self, sm_count, new_tokens, cached_tokens,
                                   *, expected_identity):
        base_ms = self.base.predict_prefill_nearest_ms(
            sm_count, new_tokens, cached_tokens, expected_identity=expected_identity)
        return self._correct(base_ms, sm_count, new_tokens)


def load_engineering_online_prefill_stage(base, path, *, expected_sha256,
                                          expected_identity):
    payload = Path(path).read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError("Online Prefill calibration hash mismatch")
    data = json.loads(payload)
    alpha = data.get("alpha_ms_per_request_layer_by_sm", ())
    if (data.get("schema") != "raglease.h100-online-prefill-residual.v1"
            or data.get("engineering_only") is not True
            or data.get("formal_ready") is not False
            or tuple(data.get("identity", ())) != expected_identity
            or not isinstance(alpha, list)
            or len(alpha) != 5
            or any(not isinstance(row, list) or len(row) != 2
                   or type(row[0]) is not int for row in alpha)
            or data.get("calibration", {}).get("seed") != 9981
            or data["calibration"].get("server_info_sha256")
               != "3d64c165db236b53a5f1596145eb96d71d7a49ce5d5d9dd28317868fe497cd1e"):
        raise ValueError("Online Prefill development profile provenance mismatch")
    return OnlinePrefillStage(base, identity=expected_identity,
        total_layers=data.get("total_layers"), alpha_by_sm=dict(alpha))
