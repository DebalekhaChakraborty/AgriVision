"""Phase 7: close the observability gap Phase 6 left open, or leave it open.

Phase 6 reported observability at 5/6. The unverified row was whether the
deployed process actually emitted structured JSON to CloudWatch - the formatter
was covered by unit tests, but no log line had ever been read back, and asserting
emission from a passing unit test would have been a claim about the wrong thing.

This closes it with a read-only audit identity that exists for nothing else. The
deployment principal was not widened, and separating the two is right on its own
merits: deploying and auditing are different jobs.

**A Phase 6 misdiagnosis is corrected here rather than quietly dropped.** Phase 6
recorded that the deployment identity was "denied logs:FilterLogEvents by
design". That was wrong. It was inferred from a probe of `logs:DescribeLogGroups`
- an account-wide list call that requires `Resource: "*"` and is therefore
denied by a policy scoped to a log-group ARN - and generalised to the whole
`logs:` namespace without testing it. The deployer has in fact held scoped
`/aws/apprunner/*` read since Phase 4, so the Phase 6 gap could have been closed
at the time. The gap was real; the reason given for it was not.

Two things are checked, and the second matters more than the first: that the
records contain the causal fields a trace needs, and that they contain nothing
they should not - no image bytes, no credentials, no authorization headers, no
local paths.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from collections import Counter
from pathlib import Path

CLOUDWATCH_AUDIT_VERSION = "phase7-cloudwatch-audit-1.0.0"

RESULTS_DIR = Path("competition/evaluation/results/phase7")
AUDIT_PATH = RESULTS_DIR / "cloudwatch_audit.json"
PHASE6 = Path("competition/evaluation/results/phase6")

LOG_GROUP = (
    "/aws/apprunner/agrivision-inspection/"
    "9be40db3a6d147c79454d1e9c90fae08/application"
)

# Fields a causal record has to carry to be worth calling a trace.
REQUIRED_STEP_FIELDS = ("run_id", "state", "tool", "action", "reason_code",
                        "duration_ms", "log_version")

# Things that must never appear in a log line.
FORBIDDEN_PATTERNS = {
    "aws_access_key": re.compile(r"AKIA[0-9A-Z]{16}"),
    "aws_secret_key": re.compile(r"(?i)aws_secret_access_key\s*[=:]"),
    "authorization_header": re.compile(r"(?i)\bauthorization\b\s*[:=]"),
    "bearer_token": re.compile(r"(?i)bearer\s+[A-Za-z0-9._-]{10,}"),
    "private_key": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "local_path": re.compile(r"(/home/|/opt/|/tmp/|/var/task/|[A-Z]:\\\\)"),
    "base64_image_blob": re.compile(r"(?:data:image/|/9j/4[A-Za-z0-9+/]{40,})"),
    "s3_uri": re.compile(r"s3://"),
}


class CloudWatchAuditError(RuntimeError):
    """Raised when the audit cannot be performed as specified."""


def _aws(runner: str, *args: str) -> dict:
    """Run one read-only AWS call through the audit-identity wrapper."""
    result = subprocess.run(
        [runner, "aws", *args], capture_output=True, text=True)
    if result.returncode != 0:
        raise CloudWatchAuditError(result.stderr.strip()[:300])
    return json.loads(result.stdout or "{}")


def fetch_events(runner: str, pattern: str, limit: int = 400) -> list[dict]:
    """Pull matching events, following pagination until the limit is reached.

    CloudWatch returns an empty first page with a continuation token far more
    often than people expect, so stopping at the first empty response would
    conclude "no structured logging" from a paging artefact.
    """
    events: list[dict] = []
    token: str | None = None
    for _ in range(40):
        args = ["logs", "filter-log-events", "--log-group-name", LOG_GROUP,
                "--filter-pattern", pattern, "--limit", "500", "--output", "json"]
        if token:
            args += ["--next-token", token]
        payload = _aws(runner, *args)
        events.extend(payload.get("events", []))
        # The raw API returns `nextToken`; the CLI's own client-side paginator
        # returns `NextToken`. Reading only one of them is how an empty first
        # page turns into "no structured logging".
        token = payload.get("nextToken") or payload.get("NextToken")
        if not token or len(events) >= limit:
            break
    return events[:limit]


def audit(runner: str) -> dict:
    events = fetch_events(runner, '"run_id"')
    if not events:
        raise CloudWatchAuditError(
            "no structured events matched; the gap stays open rather than "
            "being reported closed")

    parsed, unparsed = [], 0
    for event in events:
        try:
            parsed.append(json.loads(event["message"]))
        except (ValueError, KeyError):
            unparsed += 1

    step_events = [record for record in parsed if record.get("event") == "agent_step"]
    field_coverage = {
        field: sum(1 for record in step_events if field in record)
        for field in REQUIRED_STEP_FIELDS
    }
    complete_steps = [
        record for record in step_events
        if all(field in record for field in REQUIRED_STEP_FIELDS)
    ]

    # Cross-check against a run this project actually recorded, so the evidence
    # is tied to a known inspection rather than to arbitrary traffic.
    known_runs = set()
    for name in ("track_r_raw.json", "track_c_raw.json"):
        path = PHASE6 / name
        if path.is_file():
            known_runs |= {
                o["run_id"] for o in json.loads(path.read_text())["observations"]
                if o.get("run_id")
            }
    logged_runs = {record.get("run_id") for record in parsed if record.get("run_id")}
    matched_runs = sorted(logged_runs & known_runs)

    # A broad scan returns whichever events CloudWatch hands back first, which
    # is not necessarily the run we care about. So one known inspection is also
    # looked up by id directly - that is the check B6 actually asks for, and it
    # ties the log evidence to a recorded run rather than to arbitrary traffic.
    targeted_run = sorted(known_runs)[0] if known_runs else None
    targeted_records: list[dict] = []
    if targeted_run:
        for event in fetch_events(runner, f'"{targeted_run}"', limit=60):
            try:
                targeted_records.append(json.loads(event["message"]))
            except (ValueError, KeyError):
                continue
        if targeted_records:
            matched_runs = sorted(set(matched_runs) | {targeted_run})

    violations = []
    for event in events:
        message = event["message"]
        for name, pattern in FORBIDDEN_PATTERNS.items():
            if pattern.search(message):
                violations.append({"pattern": name,
                                   "log_stream": event.get("logStreamName", "")})

    targeted_complete = [
        record for record in targeted_records
        if all(field in record for field in REQUIRED_STEP_FIELDS)
    ]
    sample = next(iter(targeted_complete), None)
    if sample is None:
        sample = next(
            (record for record in complete_steps
             if record.get("run_id") in known_runs), None)
    if sample is None and complete_steps:
        sample = complete_steps[0]
    redacted_sample = None
    if sample:
        redacted_sample = {
            key: sample[key] for key in REQUIRED_STEP_FIELDS if key in sample
        }
        redacted_sample["evidence_maturity"] = sample.get("evidence_maturity")

    verified = bool(complete_steps) and not violations and bool(matched_runs)
    return {
        "cloudwatch_audit_version": CLOUDWATCH_AUDIT_VERSION,
        "audit_identity": "arn:aws:iam::<account>:user/agrivision-auditor",
        "audit_identity_note": (
            "A separate read-only identity created for this audit, and verified "
            "denied for apprunner, s3, iam, dynamodb, ecr and every logs write. "
            "The deployment principal was NOT widened. It is not, however, "
            "denied log reads: it has held scoped logs read on "
            "/aws/apprunner/* since Phase 4, so this audit did not require a "
            "new identity - a separate audit principal is the preferred "
            "separation of duties, not a necessity. The Phase 6 claim that the "
            "deployer was 'denied logs:FilterLogEvents by design' was a "
            "misdiagnosis: it was inferred from a DescribeLogGroups probe, "
            "which needs Resource '*' and is denied by a resource-scoped "
            "policy, and generalised without being tested."),
        "log_group": LOG_GROUP,
        "events_examined": len(events),
        "events_parsed_as_json": len(parsed),
        "events_not_json": unparsed,
        "agent_step_events": len(step_events),
        "structured_logging_emitted": {
            "verified": bool(complete_steps),
            "complete_step_records": len(complete_steps),
            "required_fields": list(REQUIRED_STEP_FIELDS),
            "field_coverage": field_coverage,
            "how": ("Records were read back out of CloudWatch and parsed as "
                    "JSON, then checked field by field. Emission is established "
                    "by the records existing, not by the formatter's unit test."),
        },
        "tied_to_known_inspections": {
            "verified": bool(matched_runs),
            "matched_run_ids": len(matched_runs),
            "example_run_id": matched_runs[0] if matched_runs else None,
            "targeted_run_id": targeted_run,
            "targeted_records_found": len(targeted_records),
            "targeted_complete_records": len(targeted_complete),
            "targeted_states": sorted({
                r.get("state", "") for r in targeted_records if r.get("state")}),
            "how": ("One recorded inspection was looked up in CloudWatch by its "
                    "run id, and its step records read back. A broad scan was "
                    "also intersected with every run id this project recorded."),
        },
        "no_sensitive_content": {
            "verified": not violations,
            "patterns_checked": sorted(FORBIDDEN_PATTERNS),
            "violations": violations,
            "how": ("Every examined line was matched against credential, "
                    "authorization-header, private-key, local-path, S3-URI and "
                    "embedded-image patterns."),
        },
        "event_types_observed": dict(sorted(Counter(
            record.get("event", "") for record in parsed).items())),
        "redacted_sample_record": redacted_sample,
        "result": "6/6 VERIFIED" if verified else "5/6 - GAP REMAINS OPEN",
        "verified": verified,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runner", help="path to the read-only audit wrapper script")
    args = parser.parse_args(argv)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    document = audit(args.runner)
    AUDIT_PATH.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"events examined     : {document['events_examined']}")
    print(f"agent_step records  : {document['agent_step_events']}")
    print(f"complete records    : "
          f"{document['structured_logging_emitted']['complete_step_records']}")
    print(f"known runs matched  : "
          f"{document['tied_to_known_inspections']['matched_run_ids']}")
    print(f"sensitive content   : "
          f"{len(document['no_sensitive_content']['violations'])} violations")
    print(f"\nRESULT: {document['result']}")
    print(f"written: {AUDIT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
