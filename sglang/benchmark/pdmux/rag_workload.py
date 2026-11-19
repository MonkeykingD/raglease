"""Canonical workload traces for reproducible RAG serving experiments.

The trace records the offered load independently of either SGLang or
VectorLiteRAG.  Both systems can therefore receive the same query order and
absolute arrival schedule instead of merely sharing an average request rate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


SCHEMA_VERSION = 1
VALID_STAGES = {"ramp_up", "measure"}


@dataclass(frozen=True)
class RagRequest:
    request_id: int
    query_id: int
    arrival_offset_ms: float
    stage: str
    output_len: int
    topk: int

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "RagRequest":
        return cls(
            request_id=int(payload["request_id"]),
            query_id=int(payload["query_id"]),
            arrival_offset_ms=float(payload["arrival_offset_ms"]),
            stage=str(payload["stage"]),
            output_len=int(payload["output_len"]),
            topk=int(payload["topk"]),
        )


@dataclass(frozen=True)
class RagWorkload:
    metadata: Dict[str, Any]
    requests: List[RagRequest]

    @property
    def measured(self) -> List[RagRequest]:
        return [request for request in self.requests if request.stage == "measure"]

    @property
    def ramp_up(self) -> List[RagRequest]:
        return [request for request in self.requests if request.stage == "ramp_up"]

    @property
    def warmup_requests(self) -> int:
        return int(self.metadata.get("warmup_requests", 0))

    @property
    def warmup_output_len(self) -> int:
        default = self.requests[0].output_len if self.requests else 1
        return int(self.metadata.get("warmup_output_len", default))

    @property
    def warmup_interval_ms(self) -> float:
        return float(self.metadata.get("warmup_interval_ms", 1000.0))

    @property
    def temperature(self) -> float:
        return float(self.metadata.get("temperature", 1.0))

    @property
    def ignore_eos(self) -> bool:
        return bool(self.metadata.get("ignore_eos", True))

    def validate(self) -> None:
        schema_version = int(self.metadata.get("schema_version", -1))
        if schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported workload schema_version={schema_version}; "
                f"expected {SCHEMA_VERSION}."
            )
        if self.warmup_requests < 0:
            raise ValueError("warmup_requests must be non-negative.")
        if self.warmup_output_len <= 0:
            raise ValueError("warmup_output_len must be positive.")
        if self.warmup_interval_ms < 0:
            raise ValueError("warmup_interval_ms must be non-negative.")
        if self.temperature < 0:
            raise ValueError("temperature must be non-negative.")
        if not self.requests:
            raise ValueError("A workload must contain at least one request.")
        if not self.measured:
            raise ValueError("A workload must contain at least one measured request.")

        request_ids = set()
        previous_offset = -math.inf
        seen_measurement = False
        for request in self.requests:
            if request.request_id in request_ids:
                raise ValueError(f"Duplicate request_id={request.request_id}.")
            request_ids.add(request.request_id)
            if request.query_id < 0:
                raise ValueError("query_id must be non-negative.")
            if request.arrival_offset_ms < 0:
                raise ValueError("arrival_offset_ms must be non-negative.")
            if request.arrival_offset_ms < previous_offset:
                raise ValueError("Requests must be ordered by arrival_offset_ms.")
            previous_offset = request.arrival_offset_ms
            if request.stage not in VALID_STAGES:
                raise ValueError(
                    f"Unsupported stage={request.stage!r}; expected one of "
                    f"{sorted(VALID_STAGES)}."
                )
            if request.stage == "measure":
                seen_measurement = True
            elif seen_measurement:
                raise ValueError("Ramp-up requests must precede measured requests.")
            if request.output_len <= 0:
                raise ValueError("output_len must be positive.")
            if request.topk <= 0:
                raise ValueError("topk must be positive.")

    def canonical_payload(self) -> Dict[str, Any]:
        metadata = dict(self.metadata)
        metadata["schema_version"] = SCHEMA_VERSION
        return {
            "metadata": metadata,
            "requests": [asdict(request) for request in self.requests],
        }

    def sha256(self) -> str:
        encoded = json.dumps(
            self.canonical_payload(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def to_metadata_dict(self) -> Dict[str, Any]:
        return {
            **self.metadata,
            "schema_version": SCHEMA_VERSION,
            "ramp_up_requests": len(self.ramp_up),
            "measured_requests": len(self.measured),
            "sha256": self.sha256(),
        }


def generate_workload(
    *,
    num_requests: int,
    ramp_up_requests: int,
    warmup_requests: int,
    request_rate: float,
    output_len: int,
    topk: int,
    seed: int,
    deterministic_arrivals: bool = False,
    query_count: Optional[int] = None,
    query_id_start: int = 0,
    warmup_output_len: Optional[int] = None,
    warmup_interval_ms: float = 1000.0,
    temperature: float = 1.0,
    ignore_eos: bool = True,
    burst_size: Optional[int] = None,
    inter_burst_gap_ms: float = 0.0,
) -> RagWorkload:
    if num_requests <= 0:
        raise ValueError("num_requests must be positive.")
    if ramp_up_requests < 0:
        raise ValueError("ramp_up_requests must be non-negative.")
    if warmup_requests < 0:
        raise ValueError("warmup_requests must be non-negative.")
    if request_rate <= 0 or math.isnan(request_rate):
        raise ValueError("request_rate must be positive.")
    if output_len <= 0 or topk <= 0:
        raise ValueError("output_len and topk must be positive.")
    if query_count is not None and query_count <= 0:
        raise ValueError("query_count must be positive when supplied.")
    if burst_size is not None and burst_size <= 0:
        raise ValueError("burst_size must be positive when supplied.")
    if inter_burst_gap_ms < 0:
        raise ValueError("inter_burst_gap_ms must be non-negative.")

    random_gen = random.Random(seed)
    total_requests = ramp_up_requests + num_requests
    offset_ms = 0.0
    requests: List[RagRequest] = []
    for sequence_id in range(total_requests):
        query_id = query_id_start + sequence_id
        if query_count is not None:
            query_id %= query_count
        is_ramp_up = sequence_id < ramp_up_requests
        requests.append(
            RagRequest(
                request_id=sequence_id - ramp_up_requests,
                query_id=query_id,
                arrival_offset_ms=offset_ms,
                stage="ramp_up" if is_ramp_up else "measure",
                output_len=output_len,
                topk=topk,
            )
        )
        if sequence_id + 1 < total_requests:
            if math.isinf(request_rate):
                interval_s = 0.0
            elif deterministic_arrivals:
                interval_s = 1.0 / request_rate
            else:
                interval_s = random_gen.expovariate(request_rate)
            offset_ms += interval_s * 1000.0
            if (
                burst_size is not None
                and sequence_id + 1 > ramp_up_requests
                and (sequence_id + 1 - ramp_up_requests) % burst_size == 0
            ):
                offset_ms += inter_burst_gap_ms

    workload = RagWorkload(
        metadata={
            "schema_version": SCHEMA_VERSION,
            "seed": seed,
            "arrival_process": (
                "deterministic" if deterministic_arrivals else "poisson"
            ),
            "request_rate": "inf" if math.isinf(request_rate) else request_rate,
            "warmup_requests": warmup_requests,
            "warmup_output_len": warmup_output_len or output_len,
            "warmup_interval_ms": warmup_interval_ms,
            "temperature": temperature,
            "ignore_eos": ignore_eos,
            "query_count": query_count,
            "query_id_start": query_id_start,
            "burst_size": burst_size,
            "inter_burst_gap_ms": inter_burst_gap_ms,
        },
        requests=requests,
    )
    workload.validate()
    return workload


def save_workload(path: Path, workload: RagWorkload) -> None:
    workload.validate()
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata_record = {
        "record_type": "metadata",
        **workload.to_metadata_dict(),
    }
    lines = [json.dumps(metadata_record, sort_keys=True)]
    lines.extend(
        json.dumps(
            {"record_type": "request", **asdict(request)},
            sort_keys=True,
        )
        for request in workload.requests
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_workload(path: Path) -> RagWorkload:
    metadata: Optional[Dict[str, Any]] = None
    stored_sha256: Optional[str] = None
    requests: List[RagRequest] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, raw_line in enumerate(file, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path} at line {line_number}: {exc}"
                ) from exc
            record_type = payload.pop("record_type", None)
            if record_type == "metadata":
                if metadata is not None:
                    raise ValueError("A workload may contain only one metadata record.")
                stored_sha256 = payload.pop("sha256", None)
                payload.pop("ramp_up_requests", None)
                payload.pop("measured_requests", None)
                metadata = payload
            elif record_type == "request":
                requests.append(RagRequest.from_dict(payload))
            else:
                raise ValueError(
                    f"Unknown record_type={record_type!r} at line {line_number}."
                )
    if metadata is None:
        raise ValueError("The workload is missing its metadata record.")
    workload = RagWorkload(metadata=metadata, requests=requests)
    workload.validate()
    if stored_sha256 is not None and workload.sha256() != stored_sha256:
        raise ValueError(
            f"Workload fingerprint mismatch for {path}; the trace was modified."
        )
    return workload


def unique_value(requests: Iterable[RagRequest], field_name: str) -> int:
    values = {int(getattr(request, field_name)) for request in requests}
    if len(values) != 1:
        raise ValueError(
            f"VectorLiteRAG requires one {field_name} per run, got {sorted(values)}."
        )
    return values.pop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-requests", type=int, default=200)
    parser.add_argument("--ramp-up-requests", type=int, default=50)
    parser.add_argument("--warmup-requests", type=int, default=16)
    parser.add_argument("--request-rate", type=float, default=4.0)
    parser.add_argument("--output-len", type=int, default=256)
    parser.add_argument("--warmup-output-len", type=int)
    parser.add_argument("--warmup-interval-ms", type=float, default=1000.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--allow-eos", action="store_true")
    parser.add_argument("--topk", type=int, default=25)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--deterministic-arrivals", action="store_true")
    parser.add_argument("--query-count", type=int)
    parser.add_argument("--query-id-start", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workload = generate_workload(
        num_requests=args.num_requests,
        ramp_up_requests=args.ramp_up_requests,
        warmup_requests=args.warmup_requests,
        request_rate=args.request_rate,
        output_len=args.output_len,
        topk=args.topk,
        seed=args.seed,
        deterministic_arrivals=args.deterministic_arrivals,
        query_count=args.query_count,
        query_id_start=args.query_id_start,
        warmup_output_len=args.warmup_output_len,
        warmup_interval_ms=args.warmup_interval_ms,
        temperature=args.temperature,
        ignore_eos=not args.allow_eos,
    )
    save_workload(args.output, workload)
    print(json.dumps(workload.to_metadata_dict(), indent=2))
    print(f"Saved workload to {args.output}")


if __name__ == "__main__":
    main()
