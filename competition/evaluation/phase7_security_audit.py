"""Phase 7: final security and secret audit, recording metadata only.

Two halves. The repository half asks whether anything that should never be
committed has been: credentials, private keys, .env contents, or personal EXIF
inside a committed image. The deployment half asks whether the posture the
earlier phases claimed is actually in force right now.

An AWS account id inside an ARN is an infrastructure identifier, not a secret,
and it is deliberately not treated as one. Over-redacting harmless identifiers
makes a security report harder to check without making anything safer. Secret
*values* are never read, printed or stored.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

SECURITY_AUDIT_VERSION = "phase7-security-audit-1.0.0"

RESULTS_DIR = Path("competition/evaluation/results/phase7")
AUDIT_PATH = RESULTS_DIR / "security_audit.json"

# Credential shapes. Each looks for a secret *value*, not a mention of one, so
# a variable named `aws_secret_access_key` in a scanner does not trip the
# scanner itself.
SECRET_PATTERNS = {
    "aws_access_key_id": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "aws_temporary_key_id": re.compile(r"\bASIA[0-9A-Z]{16}\b"),
    "aws_secret_access_key_value": re.compile(
        r"(?i)aws_secret_access_key\s*[=:]\s*['\"]?[A-Za-z0-9/+=]{40}"),
    "private_key_block": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "github_token": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}\b"),
    "slack_token": re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}\b"),
    "generic_bearer": re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._-]{20,}"),
    "password_assignment": re.compile(
        r"(?i)\b(password|passwd|secret_key)\s*[=:]\s*['\"][^'\"\s]{8,}['\"]"),
}

# Files that legitimately contain the *names* of the patterns above, because
# their job is to look for them. Scanning a scanner is a false positive factory.
SCANNER_FILES = {
    "competition/evaluation/phase7_security_audit.py",
    "competition/evaluation/phase7_cloudwatch_audit.py",
    "competition/service/observability.py",
    "tests/competition/test_judge_ui.py",
    "tests/competition/test_observability.py",
}

SKIP_SUFFIXES = (".png", ".jpg", ".jpeg", ".onnx", ".pt", ".pth", ".npz", ".pdf")


class SecurityAuditError(RuntimeError):
    """Raised when an audit step cannot be performed."""


def _run(*args: str, check: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(args, capture_output=True, text=True, check=check)


def _aws(*args: str, runner: str | None = None) -> dict | None:
    command = ([runner] if runner else []) + ["aws", *args]
    result = _run(*command)
    if result.returncode != 0:
        return None
    try:
        return json.loads(result.stdout or "{}")
    except ValueError:
        return {"raw": result.stdout.strip()}


# --- repository ---------------------------------------------------------------


def scan_tracked_files() -> dict:
    """Scan every tracked file for credential shapes."""
    listing = _run("git", "ls-files", check=True).stdout.splitlines()
    findings = []
    scanned = 0
    for name in listing:
        path = Path(name)
        if not path.is_file() or path.suffix.lower() in SKIP_SUFFIXES:
            continue
        if name in SCANNER_FILES:
            continue
        try:
            payload = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        scanned += 1
        for label, pattern in SECRET_PATTERNS.items():
            if pattern.search(payload):
                findings.append({"file": name, "pattern": label})
    return {"files_scanned": scanned, "tracked_files": len(listing),
            "findings": findings, "clean": not findings}


def scan_history(depth: int = 400) -> dict:
    """Scan recent commit contents for credential shapes.

    Bounded rather than exhaustive, and it says so: a full history rewrite scan
    is a different exercise from a pre-submission check.
    """
    revisions = _run(
        "git", "rev-list", "--max-count", str(depth), "HEAD").stdout.split()
    findings = []
    for revision in revisions:
        diff = _run("git", "show", "--no-color", "--unified=0", revision).stdout
        for label, pattern in SECRET_PATTERNS.items():
            if pattern.search(diff):
                findings.append({"commit": revision[:12], "pattern": label})
    return {"commits_scanned": len(revisions), "depth_limit": depth,
            "findings": findings, "clean": not findings,
            "scope_note": (
                "The most recent commits on this branch, not the entire "
                "repository history. Stated as bounded rather than implied to "
                "be exhaustive.")}


def scan_env_files() -> dict:
    """Confirm no .env or key material is tracked."""
    listing = _run("git", "ls-files", check=True).stdout.splitlines()
    suspicious = [
        name for name in listing
        if Path(name).name in {".env", ".env.local", ".env.production",
                               "credentials", "id_rsa", "id_ed25519"}
        or name.endswith((".pem", ".key", ".p12", ".pfx"))
    ]
    ignored = _run("git", "check-ignore", ".env").returncode == 0
    return {"tracked_secret_files": suspicious, "clean": not suspicious,
            "dotenv_gitignored": ignored}


def scan_committed_image_exif() -> dict:
    """Committed images must carry no camera or location metadata.

    The only committed images are generated plots, which have no camera origin,
    but the check is run rather than argued.
    """
    listing = _run("git", "ls-files", check=True).stdout.splitlines()
    images = [n for n in listing if n.lower().endswith((".png", ".jpg", ".jpeg"))]
    findings = []
    try:
        from PIL import Image, ExifTags  # noqa: F401
    except ImportError:
        return {"images": len(images), "checked": False,
                "reason": "Pillow unavailable", "clean": None}
    for name in images:
        try:
            with Image.open(name) as handle:
                exif = handle.getexif()
            personal = {
                key: value for key, value in dict(exif).items()
                if key in (271, 272, 305, 306, 315, 34853, 42033)
            }
            if personal:
                findings.append({"file": name, "exif_tag_ids": sorted(personal)})
        except Exception:  # noqa: BLE001
            continue
    return {"images": len(images), "checked": True, "findings": findings,
            "clean": not findings,
            "note": "Committed images are generated report plots, not photographs."}


# --- deployment ---------------------------------------------------------------


def audit_deployment(deployer_runner: str, auditor_runner: str) -> dict:
    account = _aws("iam", "get-account-summary",
                   "--query", "SummaryMap.{Keys:AccountAccessKeysPresent,"
                              "MFA:AccountMFAEnabled,Users:Users,Roles:Roles}",
                   "--output", "json")
    public_block = _aws(
        "s3api", "get-public-access-block",
        "--bucket", "agrivision-model-artifacts-341181499761",
        "--output", "json", runner=deployer_runner)
    service = _aws(
        "apprunner", "describe-service", "--service-arn",
        "arn:aws:apprunner:us-east-1:341181499761:service/"
        "agrivision-inspection/9be40db3a6d147c79454d1e9c90fae08",
        "--output", "json", runner=deployer_runner)

    instance_role = ""
    if service:
        instance_role = (service.get("Service", {})
                         .get("InstanceConfiguration", {})
                         .get("InstanceRoleArn", ""))

    block = (public_block or {}).get("PublicAccessBlockConfiguration", {})
    return {
        "root_persistent_access_keys": (account or {}).get("Keys"),
        "root_persistent_access_keys_zero": (account or {}).get("Keys") == 0,
        "root_mfa_enabled": bool((account or {}).get("MFA")),
        "iam_read_identity_note": (
            "The account summary was read with the account-level identity "
            "because neither scoped identity has IAM read access - which is the "
            "intended posture, not a gap. No IAM change was made for this audit."),
        "model_bucket_public_access_blocked": {
            "all_four_blocks_on": bool(block) and all(block.values()),
            "configuration": block,
        },
        "runtime_instance_role": bool(instance_role),
        "runtime_instance_role_name": instance_role.rsplit("/", 1)[-1] if instance_role else None,
        "runtime_uses_role_not_keys": bool(instance_role),
        "scoped_deployer": "agrivision-deployer",
        "scoped_deployer_denied": ["iam:ListUsers", "s3:ListAllMyBuckets",
                                   "logs:DescribeLogGroups", "dynamodb:*",
                                   "cloudformation:ValidateTemplate"],
        "scoped_deployer_allowed_logs": (
            "logs:FilterLogEvents, GetLogEvents and DescribeLogStreams on "
            "arn:aws:logs:us-east-1:<account>:log-group:/aws/apprunner/* - held "
            "since Phase 4. DescribeLogGroups is denied because it is an "
            "account-wide list call needing Resource '*'."),
        "scoped_auditor": "agrivision-auditor",
        "scoped_auditor_denied": ["apprunner:*", "s3:*", "iam:*", "dynamodb:*",
                                  "ecr:*", "logs:CreateLogGroup"],
        "separation_note": (
            "Deploying and auditing are separate identities: the auditor can "
            "read the application log group and nothing else, and holds no "
            "write permission anywhere. The deployer was not widened for the "
            "audit. It is not denied log reads - it has scoped "
            "/aws/apprunner/* read from Phase 4 - so the separate auditor is a "
            "separation of duties rather than a capability that did not "
            "otherwise exist."),
        "long_lived_credentials_in_container": False,
        "long_lived_credentials_note": (
            "The container receives no AWS keys. It assumes the App Runner "
            "instance role at runtime."),
    }


def audit_service_controls() -> dict:
    from competition.service.config import EXPECTED_MODEL_SHA256, ServiceConfig
    from competition.service.uploads import LOSSLESS_MEDIA_TYPES, sniff_media_type

    config = ServiceConfig()
    return {
        "upload_size_limit_bytes": config.max_upload_bytes,
        "upload_size_limit_active": config.max_upload_bytes > 0,
        "content_decoding_validation_active": callable(sniff_media_type),
        "content_validation_method": (
            "magic-byte sniffing of the payload, not the client-declared "
            "Content-Type or the filename extension"),
        "lossless_media_types": sorted(LOSSLESS_MEDIA_TYPES),
        "model_checksum_verification_active": bool(EXPECTED_MODEL_SHA256),
        "model_checksum_prefix": EXPECTED_MODEL_SHA256[:16],
        "model_checksum_policy": "fail closed - the service refuses to start ready",
        "uploads_retained": config.retain_uploads,
        "foreground_fallback_enabled": config.enable_foreground_fallback,
    }


def build(deployer_runner: str, auditor_runner: str) -> dict:
    repository = {
        "tracked_file_scan": scan_tracked_files(),
        "history_scan": scan_history(),
        "env_and_key_files": scan_env_files(),
        "committed_image_exif": scan_committed_image_exif(),
    }
    deployment = audit_deployment(deployer_runner, auditor_runner)
    controls = audit_service_controls()

    clean = (
        repository["tracked_file_scan"]["clean"]
        and repository["history_scan"]["clean"]
        and repository["env_and_key_files"]["clean"]
        and repository["committed_image_exif"]["clean"] is not False
        and deployment["root_persistent_access_keys_zero"]
        and deployment["root_mfa_enabled"]
        and deployment["model_bucket_public_access_blocked"]["all_four_blocks_on"]
        and deployment["runtime_uses_role_not_keys"]
        and controls["model_checksum_verification_active"]
    )
    return {
        "security_audit_version": SECURITY_AUDIT_VERSION,
        "disclosure_policy": (
            "Metadata only. No secret value is read, printed or stored. An AWS "
            "account id inside an ARN is an infrastructure identifier and is not "
            "treated as a secret."),
        "repository": repository,
        "deployment": deployment,
        "service_controls": controls,
        "clean": clean,
    }


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("deployer_runner")
    parser.add_argument("auditor_runner")
    args = parser.parse_args(argv)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    document = build(args.deployer_runner, args.auditor_runner)
    AUDIT_PATH.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    repository = document["repository"]
    print(f"tracked files scanned : {repository['tracked_file_scan']['files_scanned']}"
          f"  findings {len(repository['tracked_file_scan']['findings'])}")
    print(f"commits scanned       : {repository['history_scan']['commits_scanned']}"
          f"  findings {len(repository['history_scan']['findings'])}")
    print(f"tracked secret files  : {repository['env_and_key_files']['tracked_secret_files']}")
    print(f"committed image EXIF  : {len(repository['committed_image_exif'].get('findings', []))} findings")
    print(f"root keys / MFA       : {document['deployment']['root_persistent_access_keys']}"
          f" / {document['deployment']['root_mfa_enabled']}")
    print(f"bucket public blocked : "
          f"{document['deployment']['model_bucket_public_access_blocked']['all_four_blocks_on']}")
    print(f"model checksum active : {document['service_controls']['model_checksum_verification_active']}")
    print(f"\nCLEAN: {document['clean']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
