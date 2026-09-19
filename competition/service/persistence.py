"""Run and trace persistence.

Two backends behind one interface: DynamoDB when a table is configured, and an
in-memory store otherwise. The in-memory store is not a stub for tests — it is
what a local container uses, and it is what the service degrades to if DynamoDB
is unreachable.

The degradation rule is the important part. **A persistence failure must never
change an inspection result.** The perception and decision work has already
happened and is already correct; losing the ability to store it is an
availability problem for later retrieval, not a reason to invent, suppress or
alter a verdict. Writes therefore fail soft and are counted, while the response
still carries the full result and trace.

Image bytes are never stored here. A trace identifies its image by SHA-256.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass


class PersistenceError(RuntimeError):
    """Raised only by explicit reads; writes fail soft and report a flag."""


@dataclass(frozen=True)
class WriteOutcome:
    stored: bool
    backend: str
    detail: str = ""


class InMemoryStore:
    """Process-local store. Survives nothing, and says so."""

    backend = "in_memory"

    def __init__(self, max_records: int = 500) -> None:
        self._records: dict = {}
        self._order: list = []
        self._lock = threading.Lock()
        self._max = max_records

    def put(self, run_id: str, record: dict) -> WriteOutcome:
        with self._lock:
            self._records[run_id] = record
            self._order.append(run_id)
            while len(self._order) > self._max:
                self._records.pop(self._order.pop(0), None)
        return WriteOutcome(stored=True, backend=self.backend)

    def get(self, run_id: str) -> dict | None:
        with self._lock:
            return self._records.get(run_id)

    def healthy(self) -> bool:
        return True


class DynamoDbStore:
    """DynamoDB-backed store with a TTL, for retrieval across container replacements.

    App Runner replaces instances freely and runs more than one, so a run
    inspected on instance A must be retrievable from instance B. That is the
    whole reason a shared store exists here; it is not chosen for scale.

    TTL is set on every record because these are demonstration runs, not
    business records, and an unbounded table of them is an unbounded cost.
    """

    backend = "dynamodb"

    def __init__(self, table_name: str, region: str, ttl_days: int = 14) -> None:
        import boto3

        self._table_name = table_name
        self._ttl_seconds = max(1, ttl_days) * 86400
        self._table = boto3.resource("dynamodb", region_name=region).Table(table_name)

    def put(self, run_id: str, record: dict) -> WriteOutcome:
        from botocore.exceptions import BotoCoreError, ClientError

        item = {
            "run_id": run_id,
            "expires_at": int(time.time()) + self._ttl_seconds,
            **record,
        }
        try:
            self._table.put_item(Item=_to_dynamo(item))
        except (BotoCoreError, ClientError) as error:
            # Soft failure by design. See the module docstring.
            return WriteOutcome(
                stored=False, backend=self.backend, detail=type(error).__name__
            )
        return WriteOutcome(stored=True, backend=self.backend)

    def get(self, run_id: str) -> dict | None:
        from botocore.exceptions import BotoCoreError, ClientError

        try:
            response = self._table.get_item(Key={"run_id": run_id})
        except (BotoCoreError, ClientError) as error:
            raise PersistenceError(type(error).__name__) from error
        return response.get("Item")

    def healthy(self) -> bool:
        from botocore.exceptions import BotoCoreError, ClientError

        try:
            self._table.load()
        except (BotoCoreError, ClientError):
            return False
        return True


def _to_dynamo(value):
    """Convert floats to Decimal; DynamoDB has no float type.

    Via `str` rather than binary, so 0.1 stores as 0.1 rather than as its
    binary expansion.
    """
    from decimal import Decimal

    if isinstance(value, float):
        return Decimal(str(value))
    if isinstance(value, dict):
        return {k: _to_dynamo(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_dynamo(v) for v in value]
    return value


def build_store(config) -> object:
    """Choose a backend. Falls back to memory if DynamoDB cannot be constructed."""
    if not config.persistence_enabled:
        return InMemoryStore()
    try:
        return DynamoDbStore(
            config.trace_table, config.aws_region, config.trace_ttl_days
        )
    except Exception:  # noqa: BLE001 - a store that cannot be built is not fatal
        return InMemoryStore()
