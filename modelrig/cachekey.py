"""Two hashes, and the constraint that keeps them apart (§7-§8).

    h_ident(s)  = H(entire spec)      identifies THIS spec, for audit
    h_cache(s)  = H(semantic subset)  identifies the ARTEFACT it produces

They must differ or the cache never hits: two requests identical in everything
that matters, differing only in owner or budget ceiling, would key differently
and rebuild from scratch. §7 excludes ``owner_id``, timestamps, provenance and
``budget.*`` for exactly that reason — a customer with a larger budget and
identical requirements should share an artefact.

.. warning::

   **§8 — the correction, and it is a data-protection bug rather than a
   refinement.**

   The obvious next step is to exclude ``data.seed_ref`` too: it names *data*,
   not *requirements*, so hashing the requirements alone looks like a free
   improvement to the hit rate. It is not. Two customers with identical
   requirements and different confidential corpora would then collide, and the
   second would be served **a model trained on the first customer's documents**.

   For a system whose central promise is that data never leaves the customer's
   control, that is the worst failure available, and it arrives through a
   plausible optimisation. Hence the invariant:

       data.seed_ref is in h_cache. Always.

   Two independent barriers, because one is not enough against this class of
   failure: the seed reference is in the key, **and** :func:`lookup_allowed`
   refuses a cross-owner hit on private data at retrieval time. Either alone
   would prevent the leak; both are required so that a mistake in one is not
   sufficient to cause it.

What this bounds, stated plainly so nobody is surprised by the hit rate:

* cross-customer exact hits happen **only on shared or public corpora**;
* within-customer hits are the common case — rebuilds, idempotent retries, and
  spec edits that touch only excluded fields;
* the compose tier is unaffected, because composition builds a *new* adapter
  from a weighted combination rather than serving somebody else's artefact. It
  is the mechanism that captures cross-customer value legitimately.
"""
from __future__ import annotations

from typing import Any

from majestic.logging_utils import get_logger
from modelrig.ir import SpecIR, content_hash
from modelrig.licence import DataRights

logger = get_logger(__name__)

#: Fields that identify the *requester* rather than the artefact. Excluding them
#: is the whole point of a second hash.
CACHE_EXCLUDED: frozenset[str] = frozenset({
    "owner_id",          # who asked
    "created_at",        # when
    "notes",             # free-text provenance
    "budget_ceiling_usd",  # §7: two ceilings, one artefact
    "spec_version",
})

#: Fields that MUST be in the cache key. Listed separately from "not excluded"
#: so the invariant is a positive assertion rather than the absence of a line.
CACHE_REQUIRED: frozenset[str] = frozenset({
    "task_primitive",    # what is being built
    "seed_data_ref",     # §8 — on WHOSE data. Never remove this.
    "data_rights",
    "io_schema",
    "offline_required",
    "quality_gate",
})

#: Device-profile keys that identify the *unit* rather than its behaviour. Two
#: probes of two units of the same phone predict the same plan, so keeping the
#: serial number in the key would rebuild for every handset.
_PROFILE_INCIDENTAL: frozenset[str] = frozenset({
    "device_id", "measured_at", "probe_version",
})


def _normalise_profile(profile: Any) -> Any:
    if not isinstance(profile, dict):
        return profile
    return {k: v for k, v in sorted(profile.items()) if k not in _PROFILE_INCIDENTAL}


def h_ident(spec: SpecIR) -> str:
    """The identity hash: this spec, including who asked and when.

    Used for audit and lineage. Never for the cache — it is unique per request
    almost by construction, so keying the cache on it guarantees a miss.
    """
    return content_hash(spec)


def h_cache(spec: SpecIR) -> str:
    """The artefact hash: everything that changes what gets built.

    Two specs share this hash exactly when they would produce the same
    cartridge — which includes producing it from the same data.
    """
    data = spec.to_dict()
    payload = {k: v for k, v in data.items() if k not in CACHE_EXCLUDED}
    payload["device_profile"] = _normalise_profile(payload.get("device_profile"))

    missing = CACHE_REQUIRED - set(payload)
    if missing:
        # A guard rather than a comment. If a schema change ever drops one of
        # these from the payload, the cache stops distinguishing builds it must
        # distinguish, and nothing else would report it.
        raise ValueError(
            f"h_cache is missing required fields {sorted(missing)}: the cache key "
            "must distinguish these or it will serve the wrong artefact"
        )
    return content_hash(payload)


def data_is_public(spec: SpecIR) -> bool:
    """Whether this build's corpus may be shared across owners.

    Only public-domain data qualifies. ``licensed`` does not: a licence to use
    data is not a licence to serve a model trained on it to somebody else.
    """
    rights = spec.data_rights
    if isinstance(rights, str) and not isinstance(rights, DataRights):
        try:
            rights = DataRights(rights.lower())
        except ValueError:
            return False
    return rights is DataRights.PUBLIC_DOMAIN


def lookup_allowed(
    *, requester: str, owner: str, hit_is_public: bool, strict: bool = True
) -> bool:
    """§8's second barrier: may ``requester`` be served ``owner``'s artefact?

    Independent of the cache key by design. The key stops the collision from
    forming; this stops it being served if one ever forms anyway — through a
    hash collision, a restored index, a schema change, or a future optimisation
    written by somebody who has not read §8.
    """
    if owner == requester:
        return True
    if hit_is_public:
        return True
    if strict:
        logger.warning(
            "registry: refusing a cross-owner cache hit on private data "
            "(owner=%s requester=%s)", owner or "<none>", requester or "<none>",
        )
    return False


def cache_key(spec: SpecIR) -> dict[str, Any]:
    """Both hashes plus the sharing facts, for storing alongside an entry."""
    return {
        "h_ident": h_ident(spec),
        "h_cache": h_cache(spec),
        "owner_id": spec.owner_id,
        "data_is_public": data_is_public(spec),
        "seed_data_ref": spec.seed_data_ref,
    }


def explain(spec: SpecIR) -> dict[str, Any]:
    """Why these two hashes differ, and what that permits."""
    return {
        **cache_key(spec),
        "excluded_from_cache": sorted(CACHE_EXCLUDED),
        "required_in_cache": sorted(CACHE_REQUIRED),
        "shareable_across_owners": data_is_public(spec),
        "why": (
            "seed_data_ref is in the cache key so two customers with identical "
            "requirements and different confidential data cannot collide; the "
            "owner check refuses a cross-owner hit even if they somehow do"
        ),
    }


__all__ = [
    "CACHE_EXCLUDED", "CACHE_REQUIRED",
    "cache_key", "data_is_public", "explain", "h_cache", "h_ident", "lookup_allowed",
]
