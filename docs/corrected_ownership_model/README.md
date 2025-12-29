# Corrected Ownership Model Documentation

**Status:** CANONICAL - This supersedes all prior architecture documents
**Date:** 2025-12-28 (Revised)

## Documents in This Directory

| File | Purpose |
|------|---------|
| `00-canonical-architecture.md` | Authoritative architecture spec defining port and adapter ownership |
| `prompt.md` | Agent prompt for executing the full refactor using TDD |

## Quick Summary

The corrected ownership model establishes:

1. **crucible_train** - Owns ALL training-related ports (8 port behaviours)
2. **crucible_telemetry** - Owns MetricsStore port and metrics storage
3. **crucible_kitchen** - Orchestration layer + adapters that WRAP external SDKs
4. **SDK repos** - INDEPENDENT with NO crucible dependencies (tinkex, hf_datasets_ex, hf_hub_ex)

## CRITICAL: Dependency Direction

```
                     CORRECT                          WRONG

crucible_kitchen ──depends on──► tinkex      tinkex ──depends on──► crucible_train
       │                                            │
       └── kitchen adapter wraps tinkex             └── THIS INVERTS DEPENDENCIES
           to implement crucible_train port             SDK repos must stay independent!
```

**Adapters WRAP external SDKs. They don't live IN external SDKs.**

## Scope

### IN SCOPE
- crucible_telemetry (Phase 1 done)
- crucible_kitchen (Phases 2-4)
- tinkex_cookbook (Phase 5)

### OUT OF SCOPE (DO NOT TOUCH)
- tinkex
- hf_datasets_ex
- hf_hub_ex
- crucible_train

## To Execute the Plan

From `~/p/g/North-Shore-AI/`:

```bash
# Read the canonical architecture first
cat crucible_kitchen/docs/corrected_ownership_model/00-canonical-architecture.md

# Then follow the detailed prompt
cat crucible_kitchen/docs/corrected_ownership_model/prompt.md
```

## Why This Refactor?

**Problem discovered:** 5 of 7 ports in `crucible_kitchen` duplicated ports already in `crucible_train`.

**Principle violated:** "If crucible_train already does it, delegate don't duplicate."

**Solution:**
- Delete duplicate ports from kitchen
- Update adapters to implement crucible_train ports instead of kitchen ports
- Update stages to import from crucible_train
- Adapters STAY in kitchen (they wrap SDKs, they don't live in SDKs)
