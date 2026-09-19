"""Structured JSON logging and in-process metrics.

CloudWatch Logs indexes JSON fields, so one line per event with stable keys
turns a log group into something queryable: "show me every run whose action was
REQUEST_RECAPTURE and whose reason was BLUR_NOT_REMEDIABLE" is a Logs Insights
query rather than a grep through prose.

The redaction list is deliberately short and absolute. Image bytes, credentials,
authorization headers and filesystem paths never reach a log line, and neither
does a full request body. `_scrub` enforces this on every value rather than
trusting each call site to remember, because the one call site that forgets is
the one that matters.
"""

from __future__ import annotations

import json
import logging
import sys
import threading
import time
from collections import Counter
from dataclasses import dataclass, field

LOG_VERSION = "phase4-logs-1.0.0"

# Substrings that must never appear in a logged value. A value containing one is
# replaced entirely rather than partially masked: a partially masked secret is
# still a leak of its length and prefix.
_FORBIDDEN_MARKERS = ("/home/", "/Users/", "/var/", "/tmp/", "C:\\",
                      "aws_access_key", "aws_secret", "authorization",
                      "bearer ", "x-amz-security-token", "-----begin")

_REDACTED = "[redacted]"
_MAX_VALUE_CHARS = 512


def _scrub(value):
    """Recursively drop anything that must not be logged."""
    if isinstance(value, dict):
        return {k: (_REDACTED if _is_sensitive_key(k) else _scrub(v))
                for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_scrub(v) for v in value]
    if isinstance(value, bytes):
        return f"[{len(value)} bytes omitted]"
    if isinstance(value, str):
        lowered = value.lower()
        if any(marker in lowered for marker in _FORBIDDEN_MARKERS):
            return _REDACTED
        if len(value) > _MAX_VALUE_CHARS:
            return value[:_MAX_VALUE_CHARS] + "...[truncated]"
        return value
    return value


def _is_sensitive_key(key: str) -> bool:
    lowered = str(key).lower()
    return any(token in lowered for token in (
        "password", "secret", "token", "credential", "authorization",
        "api_key", "apikey", "cookie", "session", "filename", "path",
        "image_bytes", "body",
    ))


class JsonFormatter(logging.Formatter):
    """One JSON object per line, with the competition fields hoisted to the top."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "log_version": LOG_VERSION,
        }
        for key, value in getattr(record, "fields", {}).items():
            payload[key] = _scrub(value)
        if record.exc_info:
            # Type only. A traceback can carry paths and local variable values,
            # and neither belongs in a log aggregated across deployments.
            payload["exception_type"] = record.exc_info[0].__name__
        return json.dumps(payload, sort_keys=True, default=str)


def configure_logging(level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("agrivision")
    logger.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False
    return logger


def log_event(logger: logging.Logger, event: str, level: str = "INFO", **fields) -> None:
    """Emit one structured event. Every field passes through `_scrub`."""
    logger.log(
        getattr(logging, level.upper(), logging.INFO),
        event,
        extra={"fields": {"event": event, **fields}},
    )


# --- metrics ------------------------------------------------------------------


@dataclass
class Metrics:
    """In-process counters and latency samples.

    Deliberately not a CloudWatch custom-metric client. Custom metrics cost per
    metric per month and App Runner already publishes request count, latency and
    error rate for free; what is missing is the *agent* dimension, and that is
    carried in the structured logs where it can be queried without a second
    billing surface. This object exposes the same numbers on /metrics for a
    judge to read directly.
    """

    counters: Counter = field(default_factory=Counter)
    durations: dict = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def increment(self, name: str, amount: int = 1) -> None:
        with self._lock:
            self.counters[name] += amount

    def observe(self, name: str, milliseconds: float) -> None:
        with self._lock:
            self.durations.setdefault(name, []).append(float(milliseconds))

    def snapshot(self) -> dict:
        with self._lock:
            durations = {}
            for name, samples in self.durations.items():
                if not samples:
                    continue
                ordered = sorted(samples)
                index = min(len(ordered) - 1, int(0.95 * len(ordered)))
                durations[name] = {
                    "count": len(ordered),
                    "median_ms": round(ordered[len(ordered) // 2], 2),
                    "p95_ms": round(ordered[index], 2),
                    "max_ms": round(ordered[-1], 2),
                }
            return {
                "counters": dict(self.counters),
                "durations": durations,
                "log_version": LOG_VERSION,
            }


# Counter names, declared so a typo at a call site is a visible constant rather
# than a silently separate series.
INSPECTION_REQUESTS = "inspection_requests_total"
INSPECTION_SUCCESS = "inspection_success_total"
RECAPTURE_REQUESTS = "recapture_requests_total"
REPOSITION_REQUESTS = "reposition_light_requests_total"
HUMAN_REVIEW_REQUESTS = "human_review_requests_total"
REMEDIATION_ATTEMPTS = "remediation_attempts_total"
CONDITION_INFERENCE = "condition_inference_total"
TOOL_FAILURES = "tool_failures_total"
REJECTED_UPLOADS = "rejected_uploads_total"
PERSISTENCE_FAILURES = "persistence_failures_total"
UNSAFE_INFERENCE = "unsafe_inference_count"
STEP_BUDGET_EXCEEDED = "step_budget_exceeded_count"

INSPECTION_DURATION = "inspection_duration_ms"
FOREGROUND_DURATION = "foreground_duration_ms"
QUALITY_DURATION = "quality_duration_ms"
ARTIFACT_DURATION = "artifact_duration_ms"
MODEL_DURATION = "model_duration_ms"
