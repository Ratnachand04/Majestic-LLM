"""Licence composition as a join-semilattice (§13).

§3.5 first described composition as a *meet* over permissiveness; §13 supersedes
that with the cleaner statement, and this module implements §13:

    lambda_out = lambda_base  v  lambda_teacher  v  lambda_data
    P_lic      = [ lambda_out  <=  lambda_allowed(s) ]

**Restrictions accumulate and never cancel.** Apache-2.0 joined with Gemma terms
yields Gemma terms; no later choice relaxes it.

The implementation models a licence as its **set of restriction flags** and takes
the join to be set union. That choice is not cosmetic — it buys the two
properties §13 calls out, as theorems rather than as tests:

* **Associativity.** Union is associative, so the solver needs no canonical
  ordering of components. ``(a v b) v c == a v (b v c)`` holds by construction.
* **Monotonicity.** Union only grows, so adding a component can only tighten the
  result. A plan that fails the licence check **cannot be rescued by adding
  anything** — which lets the enumerator prune the entire subtree the moment a
  licence conflict appears.

Licences are only a *partial* order (Gemma terms and CC-BY-SA are incomparable),
so a chain-based "most restrictive wins" implementation would be ill-defined.
The powerset lattice handles incomparability correctly and for free.

Refusing a build because the licence chain does not compose is, as far as I can
determine, unique to this system. Everywhere else licensing is a document the
customer reads afterwards; here it is a compile-time type error.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable

from modelrig.licence import DataRights, Licence


class Restriction(str, Enum):
    """An obligation or prohibition a licence attaches to a derived artefact.

    These are the lattice's atoms. The join of two licences is the union of
    their atoms; the artefact licence is whatever satisfies that union.
    """

    ATTRIBUTION = "attribution"                 # name the upstream work
    SHARE_ALIKE = "share_alike"                 # derivatives inherit the terms
    NAME_DISCLOSURE = "name_disclosure"         # derived name must cite the base
    USE_POLICY = "use_policy"                   # acceptable-use terms bind downstream
    NON_COMMERCIAL = "non_commercial"           # no commercial redistribution
    NO_REDISTRIBUTION = "no_redistribution"     # artefact may not be shipped at all
    NO_TRAINING = "no_training"                 # inputs may not be trained on
    NO_COMPETING_USE = "no_competing_use"       # may not train a competing model


#: Absorbing atoms: their presence makes any commercial artefact impossible.
#: These are the "bottom element" of §3.5 restated as flags.
FATAL = frozenset({
    Restriction.NO_REDISTRIBUTION,
    Restriction.NO_TRAINING,
    Restriction.NO_COMPETING_USE,
})

#: Licence -> its restriction atoms.
#: SOURCE: the published text of each licence. CLOSED_API carries
#: NO_COMPETING_USE because A-02's hard legal boundary is exactly that clause.
_ATOMS: dict[Licence, frozenset[Restriction]] = {
    Licence.APACHE_2_0: frozenset({Restriction.ATTRIBUTION}),
    Licence.MIT: frozenset({Restriction.ATTRIBUTION}),
    Licence.BSD_3: frozenset({Restriction.ATTRIBUTION}),
    Licence.LLAMA_COMMUNITY: frozenset({
        Restriction.ATTRIBUTION, Restriction.NAME_DISCLOSURE, Restriction.USE_POLICY,
    }),
    Licence.GEMMA_TERMS: frozenset({Restriction.ATTRIBUTION, Restriction.USE_POLICY}),
    Licence.CC_BY_SA_4: frozenset({Restriction.ATTRIBUTION, Restriction.SHARE_ALIKE}),
    Licence.NON_COMMERCIAL: frozenset({Restriction.ATTRIBUTION, Restriction.NON_COMMERCIAL}),
    Licence.CLOSED_API: frozenset({Restriction.NO_COMPETING_USE, Restriction.NO_REDISTRIBUTION}),
    Licence.PROPRIETARY: frozenset({Restriction.NO_REDISTRIBUTION}),
    Licence.UNKNOWN: frozenset({Restriction.NO_REDISTRIBUTION, Restriction.NO_TRAINING}),
}

#: Data rights -> restriction atoms contributed by the customer's corpus.
_DATA_ATOMS: dict[DataRights, frozenset[Restriction]] = {
    DataRights.CUSTOMER_OWNED: frozenset(),
    DataRights.LICENSED_FOR_TRAINING: frozenset({Restriction.ATTRIBUTION}),
    DataRights.PUBLIC_DOMAIN: frozenset(),
    DataRights.THIRD_PARTY_NO_TRAINING: frozenset({Restriction.NO_TRAINING}),
    DataRights.UNKNOWN: frozenset({Restriction.NO_TRAINING}),
}

#: What a standard commercial deployment can satisfy.
COMMERCIAL_ALLOWED = frozenset({
    Restriction.ATTRIBUTION, Restriction.NAME_DISCLOSURE, Restriction.USE_POLICY,
})
#: An audit tier additionally refuses anything whose terms bind downstream use.
AUDIT_ALLOWED = frozenset({Restriction.ATTRIBUTION})


@dataclass(frozen=True)
class LicencePosition:
    """A point in the lattice: the accumulated restrictions of a chain."""

    atoms: frozenset[Restriction] = frozenset()
    provenance: tuple[str, ...] = ()

    def join(self, other: LicencePosition) -> LicencePosition:
        """The least upper bound. Associative and monotone by construction."""
        return LicencePosition(
            atoms=self.atoms | other.atoms,
            provenance=self.provenance + other.provenance,
        )

    __or__ = join

    @property
    def fatal(self) -> frozenset[Restriction]:
        return self.atoms & FATAL

    def permitted_under(self, allowed: frozenset[Restriction]) -> bool:
        """``lambda_out <= lambda_allowed``: every accumulated atom is satisfiable."""
        return self.atoms <= allowed

    def violations(self, allowed: frozenset[Restriction]) -> list[Restriction]:
        return sorted(self.atoms - allowed, key=lambda r: r.value)


BOTTOM = LicencePosition()


def atoms_of(licence: Licence | str) -> frozenset[Restriction]:
    """The atoms a licence contributes. Unknown licences are maximally restrictive."""
    if isinstance(licence, str):
        try:
            licence = Licence(licence.lower())
        except ValueError:
            return _ATOMS[Licence.UNKNOWN]
    return _ATOMS.get(licence, _ATOMS[Licence.UNKNOWN])


def position_of(
    licence: Licence | str | None, label: str = ""
) -> LicencePosition:
    """Lift one component into the lattice."""
    if licence is None:
        return BOTTOM
    return LicencePosition(atoms=atoms_of(licence), provenance=(label,) if label else ())


def data_position(rights: DataRights | str, label: str = "data") -> LicencePosition:
    """Lift the customer's data rights into the lattice."""
    if isinstance(rights, str):
        try:
            rights = DataRights(rights.lower())
        except ValueError:
            rights = DataRights.UNKNOWN
    return LicencePosition(atoms=_DATA_ATOMS.get(rights, _DATA_ATOMS[DataRights.UNKNOWN]),
                           provenance=(label,))


def compose(*positions: LicencePosition) -> LicencePosition:
    """Fold the join over a chain. Order-independent, by associativity."""
    out = BOTTOM
    for p in positions:
        out = out.join(p)
    return out


def allowed_for(audit_tier: bool = False, commercial: bool = True) -> frozenset[Restriction]:
    """What the deployment context can satisfy."""
    if audit_tier:
        return AUDIT_ALLOWED
    if not commercial:
        return COMMERCIAL_ALLOWED | {Restriction.NON_COMMERCIAL}
    return COMMERCIAL_ALLOWED


@dataclass
class LicenceOutcome:
    """The resolved position plus a human-readable account of it."""

    position: LicencePosition
    permitted: bool
    violations: list[Restriction] = field(default_factory=list)
    obligations: list[str] = field(default_factory=list)
    reason: str = ""

    def as_record(self) -> dict[str, object]:
        return {
            "permitted": self.permitted,
            "atoms": sorted(a.value for a in self.position.atoms),
            "violations": [v.value for v in self.violations],
            "obligations": list(self.obligations),
            "provenance": list(self.position.provenance),
            "reason": self.reason,
        }


_OBLIGATION_TEXT = {
    Restriction.ATTRIBUTION: "attribute the upstream base and teacher in the model card",
    Restriction.NAME_DISCLOSURE: "the derived model's name must cite its base",
    Restriction.USE_POLICY: "the upstream acceptable-use policy binds downstream users",
    Restriction.SHARE_ALIKE: "share-alike propagates to the derived artefact",
    Restriction.NON_COMMERCIAL: "no commercial redistribution",
}


def resolve(
    base_licence: Licence | str,
    teacher_licence: Licence | str | None,
    data_rights: DataRights | str,
    *,
    audit_tier: bool = False,
    commercial: bool = True,
    extra: Iterable[LicencePosition] = (),
) -> LicenceOutcome:
    """Compose the chain and decide whether the artefact may be released."""
    position = compose(
        position_of(base_licence, "base"),
        position_of(teacher_licence, "teacher"),
        data_position(data_rights),
        *extra,
    )
    allowed = allowed_for(audit_tier=audit_tier, commercial=commercial)
    violations = position.violations(allowed)
    obligations = [
        _OBLIGATION_TEXT[a] for a in sorted(position.atoms & allowed, key=lambda r: r.value)
        if a in _OBLIGATION_TEXT
    ]

    if not violations:
        return LicenceOutcome(position, True, [], obligations)

    fatal = sorted(position.fatal, key=lambda r: r.value)
    if fatal:
        reason = (
            "licence chain does not compose: "
            + ", ".join(f"{f.value!r}" for f in fatal)
            + " is absorbing — no downstream choice can relax it"
        )
    else:
        reason = (
            "licence chain composes to restrictions this deployment cannot satisfy: "
            + ", ".join(v.value for v in violations)
        )
    return LicenceOutcome(position, False, violations, obligations, reason)
