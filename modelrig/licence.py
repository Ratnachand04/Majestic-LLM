"""Automated licence-chain resolution (GAP-08).

Composes base-weight terms, teacher-output terms, customer data rights and
target jurisdiction into a single artefact licence — or refuses the build before
any GPU is spent. Today this work is done by lawyers per deal; no published
solver does it, which is why it is listed as a genuinely-new contribution
(C-01) and why one error at enterprise scale is an existential legal event
rather than a bug.

The hardest rule comes from A-02: **outputs of a closed commercial API cannot be
used to train a competing model.** That single constraint eliminates the
strongest teachers and shapes the entire base catalogue.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Licence(str, Enum):
    """Licences the solver understands."""

    APACHE_2_0 = "apache-2.0"
    MIT = "mit"
    BSD_3 = "bsd-3-clause"
    LLAMA_COMMUNITY = "llama-community"
    GEMMA_TERMS = "gemma-terms"
    CC_BY_SA_4 = "cc-by-sa-4.0"
    NON_COMMERCIAL = "cc-by-nc-4.0"
    CLOSED_API = "closed-api"          # outputs of a commercial API
    PROPRIETARY = "proprietary"
    UNKNOWN = "unknown"


class DataRights(str, Enum):
    """What the customer may lawfully do with the seed data they supplied."""

    CUSTOMER_OWNED = "customer_owned"      # they own it outright
    LICENSED_FOR_TRAINING = "licensed"     # licensed with training rights
    PUBLIC_DOMAIN = "public_domain"
    THIRD_PARTY_NO_TRAINING = "no_training"  # present but not trainable
    UNKNOWN = "unknown"


# Licences that permit commercial redistribution of a derived artefact.
_PERMISSIVE = {Licence.APACHE_2_0, Licence.MIT, Licence.BSD_3}
# Licences that permit derivation but attach conditions to redistribution.
_CONDITIONAL = {Licence.LLAMA_COMMUNITY, Licence.GEMMA_TERMS, Licence.CC_BY_SA_4}
# Data rights sufficient to train on.
_TRAINABLE_RIGHTS = {
    DataRights.CUSTOMER_OWNED,
    DataRights.LICENSED_FOR_TRAINING,
    DataRights.PUBLIC_DOMAIN,
}


@dataclass
class LicenceChain:
    """The resolved licence position for one build."""

    resolved_licence: Licence | None
    permitted: bool
    reasons: list[str] = field(default_factory=list)
    obligations: list[str] = field(default_factory=list)
    provenance: dict[str, str] = field(default_factory=dict)

    def as_record(self) -> dict[str, object]:
        """Provenance record attached to the cartridge (B-08)."""
        return {
            "resolved_licence": self.resolved_licence.value if self.resolved_licence else None,
            "permitted": self.permitted,
            "obligations": list(self.obligations),
            "reasons": list(self.reasons),
            "provenance": dict(self.provenance),
        }


def _coerce(value: object, enum_cls):
    if isinstance(value, enum_cls):
        return value
    try:
        return enum_cls(str(value).lower())
    except ValueError:
        return enum_cls.UNKNOWN


def resolve_licence_chain(
    base_licence: Licence | str,
    teacher_licence: Licence | str | None,
    data_rights: DataRights | str,
    jurisdiction: str = "IN",
    commercial_redistribution: bool = True,
) -> LicenceChain:
    """Compose the four inputs into an artefact licence, or refuse.

    Parameters
    ----------
    base_licence:
        Licence of the base weights being adapted.
    teacher_licence:
        Licence of the teacher whose outputs seed the training data. ``None``
        means no teacher was used (no distillation).
    data_rights:
        What the customer may do with their seed corpus.
    jurisdiction:
        Deployment jurisdiction; recorded for provenance and used for
        jurisdiction-specific obligations.
    commercial_redistribution:
        Whether the artefact will be redistributed commercially.
    """
    base = _coerce(base_licence, Licence)
    teacher = _coerce(teacher_licence, Licence) if teacher_licence is not None else None
    rights = _coerce(data_rights, DataRights)

    reasons: list[str] = []
    obligations: list[str] = []

    # --- Rule 1: the hard legal boundary (A-02) -------------------------- #
    if teacher is Licence.CLOSED_API:
        reasons.append(
            "teacher is a closed commercial API: using its outputs to train a "
            "competing model breaches its terms (A-02 hard legal boundary)"
        )
    if teacher in (Licence.PROPRIETARY, Licence.UNKNOWN) and teacher is not None:
        reasons.append(f"teacher licence {teacher.value!r} does not grant training rights")
    if teacher is Licence.NON_COMMERCIAL and commercial_redistribution:
        reasons.append("teacher is non-commercial; commercial redistribution is not permitted")

    # --- Rule 2: base weights -------------------------------------------- #
    if base in (Licence.UNKNOWN, Licence.PROPRIETARY):
        reasons.append(f"base licence {base.value!r} does not grant derivation rights")
    elif base is Licence.NON_COMMERCIAL and commercial_redistribution:
        reasons.append("base is non-commercial; commercial redistribution is not permitted")
    elif base in _CONDITIONAL:
        obligations.append(f"comply with {base.value} redistribution terms (attribution, use policy)")
        if base is Licence.LLAMA_COMMUNITY:
            obligations.append("llama-community: name the base in the derived model's name/card")
        if base is Licence.CC_BY_SA_4:
            obligations.append("cc-by-sa-4.0: share-alike propagates to the derived artefact")

    # --- Rule 3: customer data rights ------------------------------------ #
    if rights not in _TRAINABLE_RIGHTS:
        reasons.append(
            f"data rights {rights.value!r} are insufficient to train on the seed corpus"
        )

    # --- Rule 4: jurisdiction obligations -------------------------------- #
    if jurisdiction.upper() in {"EU", "DE", "FR", "IE", "NL"}:
        obligations.append("EU AI Act: retain the training-data summary and model card")
    if jurisdiction.upper() == "IN":
        obligations.append("DPDP Act: retain the PII-scrub record for the training corpus")

    if reasons:
        return LicenceChain(
            resolved_licence=None,
            permitted=False,
            reasons=reasons,
            obligations=obligations,
            provenance={
                "base": base.value,
                "teacher": teacher.value if teacher else "none",
                "data_rights": rights.value,
                "jurisdiction": jurisdiction,
            },
        )

    # The resolved licence is the most restrictive of the inputs.
    resolved = base
    if base in _PERMISSIVE and teacher is not None and teacher in _CONDITIONAL:
        resolved = teacher
    if base is Licence.CC_BY_SA_4:
        resolved = Licence.CC_BY_SA_4

    return LicenceChain(
        resolved_licence=resolved,
        permitted=True,
        reasons=[],
        obligations=obligations,
        provenance={
            "base": base.value,
            "teacher": teacher.value if teacher else "none",
            "data_rights": rights.value,
            "jurisdiction": jurisdiction,
        },
    )
