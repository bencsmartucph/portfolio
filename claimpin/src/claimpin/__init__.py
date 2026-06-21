"""claimpin — pin every numeric claim in a manuscript to reproducible ground truth.

A claims file (claims.yaml) maps each number a paper asserts to the committed
artifact or recomputation that grounds it. `claimpin verify` resolves every
binding and fails loudly on drift; `claimpin extract` bootstraps a skeleton
claims file from a manuscript; `claimpin audit` checks coverage and prose drift.

Projects extend the built-in binding ops with a plain-Python plugin file:

    # ops.py
    import claimpin

    @claimpin.op("regime_slope")
    def regime_slope(ctx, source, regime):
        df = ctx.load_csv(source)
        ...
        return float(slope)

    @claimpin.check("headline_sign")
    def headline_sign(ctx):
        results = ctx.load_json("analysis/final_results.json")
        assert results["model3"]["coef_interaction"] < 0, "headline sign flip"
"""
from __future__ import annotations

from typing import Callable

__version__ = "0.1.0"

# Registries the plugin decorators populate. Module-level on purpose: the
# plugin file is imported once per verify run, and ops/checks must be visible
# to the resolver without threading a registry object through user code.
OPS: dict[str, Callable] = {}
CHECKS: dict[str, Callable] = {}


def op(name: str) -> Callable:
    """Register a custom binding op usable as `binds_to: {op: <name>, ...}`.

    The decorated function receives (ctx, **params) where params are the
    binding's keys minus `op`, and must return the ground-truth value as float.
    """

    def deco(fn: Callable) -> Callable:
        if name in OPS:
            raise ValueError(f"duplicate custom op: {name!r}")
        OPS[name] = fn
        return fn

    return deco


def check(name: str) -> Callable:
    """Register a named custom check run by `claimpin verify`.

    The decorated function receives (ctx,) and signals failure by raising
    AssertionError; any return value is ignored. Use for assertions that are
    not single-number claims: sign/direction expectations, plausibility bands,
    model refits compared against committed results.
    """

    def deco(fn: Callable) -> Callable:
        if name in CHECKS:
            raise ValueError(f"duplicate custom check: {name!r}")
        CHECKS[name] = fn
        return fn

    return deco


def clear_registry() -> None:
    """Reset plugin registries (used by the test suite between plugin loads)."""
    OPS.clear()
    CHECKS.clear()
