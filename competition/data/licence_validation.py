"""Licence gate: decide whether a third-party image may enter the corpus.

This module is deliberately the *first* thing written in Phase 2c-B and the only
way an image is allowed in. Nothing downloads bytes without passing through
`evaluate_licence`.

Design stance
-------------
**Default deny.** An assertion is rejected unless it resolves to a licence on an
explicit allow-list. "I could not identify this licence" and "this licence is
forbidden" therefore have the same effect, which is the correct asymmetry: the
cost of wrongly admitting an image is a licensing violation, the cost of wrongly
rejecting one is a slightly smaller corpus.

**Never infer.** A licence is read from what the source actually states. There is
no fallback that guesses from context, from a sibling file, from a dataset-level
claim applied to an image, or from the absence of a notice. Absence of a notice
is not permission.

**Conflicts reject, they do not get resolved by preference.** When two fields of
the same assertion resolve to different licences, that is a fact about the
source's metadata quality, not a puzzle to solve. The one exception is handled
outside this module and explicitly documented: where the *same rights holder*
publishes two different grants, the corpus records the more restrictive of the
two, and the decision is recorded in the source table rather than hidden here.

Vocabulary
----------
`licence_id` values are SPDX-style where SPDX has an identifier, and lowercase
sentinel strings where it does not (`public-domain`). They are compared exactly;
no substring matching is used for the allow decision.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from enum import Enum

LICENCE_GATE_VERSION = "phase2c-licence-gate-1.0.0"


class LicenceVerdict(str, Enum):
    """Outcome of the gate. Only ALLOWED permits use."""

    ALLOWED = "ALLOWED"
    REJECTED_UNKNOWN = "REJECTED_UNKNOWN"
    REJECTED_NON_COMMERCIAL = "REJECTED_NON_COMMERCIAL"
    REJECTED_NO_DERIVATIVES = "REJECTED_NO_DERIVATIVES"
    REJECTED_CONFLICTING = "REJECTED_CONFLICTING"
    REJECTED_RESTRICTED = "REJECTED_RESTRICTED"


@dataclass(frozen=True)
class LicenceTerms:
    """The obligations a permitted licence carries."""

    licence_id: str
    licence_url: str
    commercial_use_allowed: bool
    derivatives_allowed: bool
    attribution_required: bool
    share_alike_required: bool

    def to_dict(self) -> dict:
        return asdict(self)


# --- the allow-list ----------------------------------------------------------
# Every entry permits commercial use AND derivatives, because the competition
# corpus is used commercially-adjacent (a public submission) and every image is
# adapted (controlled degradation). A licence that forbids either is useless
# here regardless of how convenient the image is.

ALLOWED_LICENCES: dict[str, LicenceTerms] = {
    "CC0-1.0": LicenceTerms(
        "CC0-1.0", "https://creativecommons.org/publicdomain/zero/1.0/",
        True, True, False, False,
    ),
    "public-domain": LicenceTerms(
        "public-domain", "https://en.wikipedia.org/wiki/Public_domain",
        True, True, False, False,
    ),
    "CC-BY-2.0": LicenceTerms(
        "CC-BY-2.0", "https://creativecommons.org/licenses/by/2.0/",
        True, True, True, False,
    ),
    "CC-BY-2.5": LicenceTerms(
        "CC-BY-2.5", "https://creativecommons.org/licenses/by/2.5/",
        True, True, True, False,
    ),
    "CC-BY-3.0": LicenceTerms(
        "CC-BY-3.0", "https://creativecommons.org/licenses/by/3.0/",
        True, True, True, False,
    ),
    "CC-BY-4.0": LicenceTerms(
        "CC-BY-4.0", "https://creativecommons.org/licenses/by/4.0/",
        True, True, True, False,
    ),
    "CC-BY-SA-2.0": LicenceTerms(
        "CC-BY-SA-2.0", "https://creativecommons.org/licenses/by-sa/2.0/",
        True, True, True, True,
    ),
    "CC-BY-SA-2.5": LicenceTerms(
        "CC-BY-SA-2.5", "https://creativecommons.org/licenses/by-sa/2.5/",
        True, True, True, True,
    ),
    "CC-BY-SA-3.0": LicenceTerms(
        "CC-BY-SA-3.0", "https://creativecommons.org/licenses/by-sa/3.0/",
        True, True, True, True,
    ),
    "CC-BY-SA-4.0": LicenceTerms(
        "CC-BY-SA-4.0", "https://creativecommons.org/licenses/by-sa/4.0/",
        True, True, True, True,
    ),
}

ALLOWED_LICENCE_IDS: frozenset[str] = frozenset(ALLOWED_LICENCES)

# Licences that are recognised but must never be admitted. Held explicitly so a
# rejection reports *which* restriction applied rather than a bare "unknown",
# and so a test can assert the NC and ND paths separately.
FORBIDDEN_LICENCE_IDS: dict[str, LicenceVerdict] = {
    "CC-BY-NC-2.0": LicenceVerdict.REJECTED_NON_COMMERCIAL,
    "CC-BY-NC-3.0": LicenceVerdict.REJECTED_NON_COMMERCIAL,
    "CC-BY-NC-4.0": LicenceVerdict.REJECTED_NON_COMMERCIAL,
    "CC-BY-NC-SA-2.0": LicenceVerdict.REJECTED_NON_COMMERCIAL,
    "CC-BY-NC-SA-3.0": LicenceVerdict.REJECTED_NON_COMMERCIAL,
    "CC-BY-NC-SA-4.0": LicenceVerdict.REJECTED_NON_COMMERCIAL,
    "CC-BY-NC-ND-3.0": LicenceVerdict.REJECTED_NON_COMMERCIAL,
    "CC-BY-NC-ND-4.0": LicenceVerdict.REJECTED_NON_COMMERCIAL,
    "CC-BY-ND-2.0": LicenceVerdict.REJECTED_NO_DERIVATIVES,
    "CC-BY-ND-3.0": LicenceVerdict.REJECTED_NO_DERIVATIVES,
    "CC-BY-ND-4.0": LicenceVerdict.REJECTED_NO_DERIVATIVES,
}

UNKNOWN_LICENCE_ID = "UNKNOWN"

# Identifiers that denote the *same grant of rights* stated two different ways.
# Commons routinely labels one file `CC0` in its short name, links the CC0 deed,
# and describes it in prose as a "Public Domain Dedication" - which it is. Read
# field-by-field those resolve to two different ids, and treating that as a
# conflict discards the most permissively licensed images in the pool for a
# disagreement that does not exist.
#
# This is not a licence-guessing fallback. A group may only be declared where
# every member grants identical commercial-use, derivative, attribution and
# share-alike terms, which `_assert_equivalence_groups_are_sound` verifies at
# import time. Resolution picks the most specific member - the named instrument
# with a deed URL - so the recorded licence stays checkable. Any disagreement
# that crosses a group boundary still rejects.
_EQUIVALENT_LICENCE_GROUPS: tuple[tuple[str, ...], ...] = (
    ("CC0-1.0", "public-domain"),
)


def _assert_equivalence_groups_are_sound() -> None:
    for group in _EQUIVALENT_LICENCE_GROUPS:
        terms = {
            (
                ALLOWED_LICENCES[member].commercial_use_allowed,
                ALLOWED_LICENCES[member].derivatives_allowed,
                ALLOWED_LICENCES[member].attribution_required,
                ALLOWED_LICENCES[member].share_alike_required,
            )
            for member in group
        }
        if len(terms) != 1:
            raise AssertionError(
                f"equivalence group {group} mixes licences with different obligations"
            )


_assert_equivalence_groups_are_sound()


def reconcile_equivalent(licence_ids: set[str]) -> str | None:
    """Collapse ids that denote one grant; return None if they genuinely differ.

    The first member of a group is its canonical form.
    """
    if len(licence_ids) == 1:
        return next(iter(licence_ids))
    for group in _EQUIVALENT_LICENCE_GROUPS:
        if licence_ids <= set(group):
            return group[0]
    return None


# --- normalisation -----------------------------------------------------------

_URL_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"creativecommons\.org/publicdomain/zero/1\.0", "CC0-1.0"),
    (r"creativecommons\.org/publicdomain/mark/1\.0", "public-domain"),
    (r"creativecommons\.org/licenses/by-nc-nd/(\d(?:\.\d)?)", "CC-BY-NC-ND-{0}"),
    (r"creativecommons\.org/licenses/by-nc-sa/(\d(?:\.\d)?)", "CC-BY-NC-SA-{0}"),
    (r"creativecommons\.org/licenses/by-nc/(\d(?:\.\d)?)", "CC-BY-NC-{0}"),
    (r"creativecommons\.org/licenses/by-nd/(\d(?:\.\d)?)", "CC-BY-ND-{0}"),
    (r"creativecommons\.org/licenses/by-sa/(\d(?:\.\d)?)", "CC-BY-SA-{0}"),
    (r"creativecommons\.org/licenses/by/(\d(?:\.\d)?)", "CC-BY-{0}"),
)

# Ordered longest-qualifier-first: "by-nc-sa" must be tested before "by-sa",
# otherwise a NonCommercial-ShareAlike licence would normalise to plain
# ShareAlike and be admitted. The ordering is load-bearing, not cosmetic.
_NAME_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\bcc0\b", "CC0-1.0"),
    (r"public domain|\bpd[- ]", "public-domain"),
    (r"\bby[- ]nc[- ]nd[- ]?(\d(?:\.\d)?)", "CC-BY-NC-ND-{0}"),
    (r"\bby[- ]nc[- ]sa[- ]?(\d(?:\.\d)?)", "CC-BY-NC-SA-{0}"),
    (r"\bby[- ]nd[- ]?(\d(?:\.\d)?)", "CC-BY-ND-{0}"),
    (r"\bby[- ]nc[- ]?(\d(?:\.\d)?)", "CC-BY-NC-{0}"),
    (r"\bby[- ]sa[- ]?(\d(?:\.\d)?)", "CC-BY-SA-{0}"),
    (r"\bby[- ]?(\d(?:\.\d)?)", "CC-BY-{0}"),
)

# Long-form prose used by Commons' UsageTerms field.
_PROSE_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"creative commons zero", "CC0-1.0"),
    (r"attribution[- ]noncommercial[- ]no ?deriv\w*[- ](\d(?:\.\d)?)", "CC-BY-NC-ND-{0}"),
    (r"attribution[- ]noncommercial[- ]share ?alike[- ](\d(?:\.\d)?)", "CC-BY-NC-SA-{0}"),
    (r"attribution[- ]no ?deriv\w*[- ](\d(?:\.\d)?)", "CC-BY-ND-{0}"),
    (r"attribution[- ]noncommercial[- ](\d(?:\.\d)?)", "CC-BY-NC-{0}"),
    (r"attribution[- ]share ?alike[- ](\d(?:\.\d)?)", "CC-BY-SA-{0}"),
    (r"attribution[- ](\d(?:\.\d)?)", "CC-BY-{0}"),
)


def _normalise_version(raw: str) -> str:
    """`2` -> `2.0`; `4.0` -> `4.0`. Commons writes both forms."""
    return raw if "." in raw else f"{raw}.0"


def _match(text: str, patterns: tuple[tuple[str, str], ...]) -> str | None:
    for pattern, template in patterns:
        found = re.search(pattern, text)
        if not found:
            continue
        if "{0}" in template:
            if not found.groups():
                continue
            return template.format(_normalise_version(found.group(1)))
        return template
    return None


def normalise_licence(text: str | None) -> str:
    """Resolve a free-text licence statement or URL to a canonical id.

    Returns `UNKNOWN` rather than raising, because "unrecognised" is a normal
    and frequent outcome that the caller must handle as a rejection, not an
    exception.
    """
    if not text:
        return UNKNOWN_LICENCE_ID
    lowered = " ".join(str(text).lower().split())

    for patterns in (_URL_PATTERNS, _PROSE_PATTERNS, _NAME_PATTERNS):
        resolved = _match(lowered, patterns)
        if resolved:
            return resolved
    return UNKNOWN_LICENCE_ID


@dataclass(frozen=True)
class LicenceAssertion:
    """What a source claims about one image, before any judgement is applied.

    Every field is optional because real metadata is incomplete; the gate's job
    is to decide what that incompleteness means.
    """

    short_name: str | None = None
    usage_terms: str | None = None
    licence_url: str | None = None
    restrictions: str | None = None
    attribution_flag: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class LicenceDecision:
    """The gate's answer, carrying its own reasoning."""

    verdict: LicenceVerdict
    licence_id: str
    reasons: list[str] = field(default_factory=list)
    terms: LicenceTerms | None = None
    resolved_from: dict[str, str] = field(default_factory=dict)
    gate_version: str = LICENCE_GATE_VERSION

    @property
    def allowed(self) -> bool:
        return self.verdict is LicenceVerdict.ALLOWED

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict.value,
            "licence_id": self.licence_id,
            "reasons": list(self.reasons),
            "terms": self.terms.to_dict() if self.terms else None,
            "resolved_from": dict(self.resolved_from),
            "gate_version": self.gate_version,
        }


# Restriction markers Commons attaches to files whose *subject* carries rights
# beyond the photographer's copyright. A permissive photo licence does not clear
# them, so they reject independently of the licence.
_BLOCKING_RESTRICTIONS = (
    "trademarked", "personality", "insignia", "currency",
    "design", "nonfree", "non-free", "consent",
)


def evaluate_licence(assertion: LicenceAssertion) -> LicenceDecision:
    """Decide whether one image may enter the corpus.

    The three licence-bearing fields are resolved *independently* and then
    compared. Agreement is required: a file whose short name says CC BY-SA 4.0
    while its URL says CC BY-NC 4.0 has unreliable metadata, and unreliable
    metadata is a rejection reason in its own right.
    """
    resolved = {
        "short_name": normalise_licence(assertion.short_name),
        "usage_terms": normalise_licence(assertion.usage_terms),
        "licence_url": normalise_licence(assertion.licence_url),
    }
    informative = {
        source: value for source, value in resolved.items()
        if value != UNKNOWN_LICENCE_ID
    }

    if not informative:
        return LicenceDecision(
            verdict=LicenceVerdict.REJECTED_UNKNOWN,
            licence_id=UNKNOWN_LICENCE_ID,
            reasons=["no field resolved to a recognised licence"],
            resolved_from=resolved,
        )

    distinct = set(informative.values())
    licence_id = reconcile_equivalent(distinct)
    if licence_id is None:
        detail = ", ".join(f"{source}={value}" for source, value in sorted(informative.items()))
        return LicenceDecision(
            verdict=LicenceVerdict.REJECTED_CONFLICTING,
            licence_id=UNKNOWN_LICENCE_ID,
            reasons=[f"fields disagree about the licence: {detail}"],
            resolved_from=resolved,
        )

    if licence_id in FORBIDDEN_LICENCE_IDS:
        verdict = FORBIDDEN_LICENCE_IDS[licence_id]
        restriction = (
            "NonCommercial" if verdict is LicenceVerdict.REJECTED_NON_COMMERCIAL
            else "NoDerivatives"
        )
        return LicenceDecision(
            verdict=verdict,
            licence_id=licence_id,
            reasons=[f"{licence_id} carries a {restriction} restriction"],
            resolved_from=resolved,
        )

    if licence_id not in ALLOWED_LICENCES:
        return LicenceDecision(
            verdict=LicenceVerdict.REJECTED_UNKNOWN,
            licence_id=licence_id,
            reasons=[f"{licence_id} is not on the allow-list"],
            resolved_from=resolved,
        )

    # A permitted licence still loses to a subject-level restriction.
    restrictions = (assertion.restrictions or "").strip().lower()
    if restrictions:
        blocking = [word for word in _BLOCKING_RESTRICTIONS if word in restrictions]
        if blocking:
            return LicenceDecision(
                verdict=LicenceVerdict.REJECTED_RESTRICTED,
                licence_id=licence_id,
                reasons=[f"subject-level restriction present: {', '.join(blocking)}"],
                resolved_from=resolved,
            )

    return LicenceDecision(
        verdict=LicenceVerdict.ALLOWED,
        licence_id=licence_id,
        reasons=[],
        terms=ALLOWED_LICENCES[licence_id],
        resolved_from=resolved,
    )
