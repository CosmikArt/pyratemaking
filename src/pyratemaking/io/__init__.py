"""Schema validation for policy and claims tables.

The rest of the package assumes a small canonical schema: one row per policy
period for policies, one row per claim for claims, with the exposure and AY
columns named explicitly. The helpers in this module raise informative errors
when expected columns are missing or have the wrong dtype, so failures surface
at the boundary rather than 200 lines into the pipeline.
"""

from pyratemaking.io.claims import ClaimsSchema, validate_claims
from pyratemaking.io.policies import PolicySchema, validate_policies

__all__ = [
    "ClaimsSchema",
    "PolicySchema",
    "validate_claims",
    "validate_policies",
]
