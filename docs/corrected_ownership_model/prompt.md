# Agent Prompt: Corrected Ownership Model Implementation

**Date:** 2025-12-28 (Revised)
**Execution Directory:** `~/p/g/North-Shore-AI/`
**Approach:** Test-Driven Development (TDD)
**Goal:** Zero warnings, zero errors, all tests passing, dialyzer clean, credo clean

---

## CRITICAL: Scope and Boundaries

### IN SCOPE (you may modify these repos)
- `crucible_telemetry` - Phase 1 DONE, verify only
- `crucible_kitchen` - Phases 2-4
- `tinkex_cookbook` - Phase 4

### OUT OF SCOPE (DO NOT TOUCH)
- `tinkex` - INDEPENDENT SDK, no crucible dependencies
- `hf_datasets_ex` - INDEPENDENT SDK, no crucible dependencies
- `hf_hub_ex` - INDEPENDENT SDK, no crucible dependencies
- `crucible_train` - Ports already exist, no changes needed

**If you find yourself modifying tinkex, hf_datasets_ex, hf_hub_ex, or crucible_train - STOP. You are off track.**

---

## Mission Statement

You are implementing the **Corrected Ownership Model** for the crucible ecosystem. This refactor:

1. **Eliminates duplicate ports** - Kitchen has 5 ports that duplicate crucible_train's
2. **Consolidates MetricsStore** - Move from kitchen to telemetry (DONE)
3. **Updates adapters** - Change `@behaviour` from kitchen ports to train ports
4. **Updates stages** - Import ports from crucible_train instead of kitchen

**The Key Insight:**
- Adapters that WRAP external SDKs stay in crucible_kitchen
- They implement crucible_train port behaviours by delegating to SDK functions
- SDK repos (tinkex, hf_*_ex) remain INDEPENDENT with zero crucible knowledge

---

## Dependency Direction (CRITICAL)

```
                     CORRECT                          WRONG

crucible_kitchen ──depends on──► tinkex      tinkex ──depends on──► crucible_train
       │                                            │
       └── kitchen adapter wraps tinkex             └── THIS INVERTS DEPENDENCIES
           to implement crucible_train port             SDK repos must stay independent!
```

**Adapters WRAP external SDKs. They don't live IN external SDKs.**

---

## Required Reading (MANDATORY)

Read these files IN ORDER before making any changes:

```bash
# Canonical Architecture (START HERE - read this FIRST)
cat crucible_kitchen/docs/corrected_ownership_model/00-canonical-architecture.md

# Supporting Analysis
cat crucible_kitchen/docs/20251228/metrics-evolution/02-architectural-analysis.md
```

### Source Files to Understand

```bash
# crucible_train ports (AUTHORITATIVE - adapters must implement these)
ls crucible_train/lib/crucible_train/ports/

# crucible_kitchen duplicate ports (TO BE DELETED)
ls crucible_kitchen/lib/crucible_kitchen/ports/

# crucible_kitchen adapters (TO BE UPDATED, not moved)
ls crucible_kitchen/lib/crucible_kitchen/adapters/

# crucible_telemetry MetricsStore (ALREADY CREATED in Phase 1)
cat crucible_telemetry/lib/crucible_telemetry/ports/metrics_store.ex
cat crucible_telemetry/lib/crucible_telemetry/adapters/jsonl_metrics.ex
```

---

## Implementation Phases

### Phase 1: MetricsStore in crucible_telemetry - ALREADY DONE

The previous agent correctly completed this phase:
- Created `crucible_telemetry/lib/crucible_telemetry/ports/metrics_store.ex`
- Created `crucible_telemetry/lib/crucible_telemetry/adapters/jsonl_metrics.ex`
- Tests pass, dialyzer clean, credo clean

**Your task:** Verify Phase 1 is complete, then proceed to Phase 2.

```bash
cd crucible_telemetry && mix test && mix dialyzer && mix credo --strict
```

---

### Phase 2: Update crucible_kitchen Adapters

**Goal:** Change adapters from implementing kitchen ports to implementing crucible_train ports.

**TDD Approach:**
1. Read existing adapter tests
2. Update tests to expect `@behaviour CrucibleTrain.Ports.*`
3. Update adapters to implement the new behaviours
4. Run tests, fix any failures

**Files to UPDATE (not move, not delete):**

```
crucible_kitchen/lib/crucible_kitchen/adapters/tinkex/training_client.ex
  CHANGE: @behaviour CrucibleTrain.Ports.TrainingClient
  TO:     @behaviour CrucibleTrain.Ports.TrainingClient

crucible_kitchen/lib/crucible_kitchen/adapters/hf_datasets/dataset_store.ex
  CHANGE: @behaviour CrucibleTrain.Ports.DatasetStore
  TO:     @behaviour CrucibleTrain.Ports.DatasetStore

crucible_kitchen/lib/crucible_kitchen/adapters/hf_hub/hub_client.ex
  CHANGE: @behaviour CrucibleTrain.Ports.HubClient
  TO:     @behaviour CrucibleTrain.Ports.HubClient
```

**Important:** The callback signatures must match crucible_train's port definitions. Read each crucible_train port file to understand the expected callbacks.

**Validation:**
```bash
cd crucible_kitchen
mix compile --warnings-as-errors
mix test
```

---

### Phase 3: Update crucible_kitchen Stages

**Goal:** Change stages to import ports from crucible_train instead of kitchen.

**TDD Approach:**
1. Read existing stage code to find port references
2. Update imports/aliases
3. Run tests, fix any failures

**Pattern:**
```elixir
# BEFORE (wrong - using kitchen's duplicate port)
alias CrucibleKitchen.Ports.TrainingClient

# AFTER (correct - using train's authoritative port)
alias CrucibleTrain.Ports.TrainingClient
```

**Files to check and update:**
```bash
grep -r "CrucibleKitchen.Ports" crucible_kitchen/lib/crucible_kitchen/stages/
```

Also update LogStepMetrics to use:
```elixir
alias CrucibleTelemetry.Ports.MetricsStore
```

**Validation:**
```bash
cd crucible_kitchen
mix compile --warnings-as-errors
mix test
```

---

### Phase 4: Delete Duplicate Ports from crucible_kitchen

**Goal:** Remove the duplicate port files that are now unused.

**CRITICAL:** Only do this AFTER Phases 2 and 3 are complete and passing!

**Files to DELETE:**
```
crucible_kitchen/lib/crucible_kitchen/ports/training_client.ex
crucible_kitchen/lib/crucible_kitchen/ports/dataset_store.ex
crucible_kitchen/lib/crucible_kitchen/ports/blob_store.ex
crucible_kitchen/lib/crucible_kitchen/ports/hub_client.ex
crucible_kitchen/lib/crucible_kitchen/ports/metrics_store.ex
```

**Files to KEEP:**
```
crucible_kitchen/lib/crucible_kitchen/ports/completer.ex  # Kitchen-specific
```

**Update the ports.ex module** (if it exists) to remove re-exports of deleted ports.

**Validation:**
```bash
cd crucible_kitchen
mix compile --warnings-as-errors
mix test
mix dialyzer
mix credo --strict
```

---

### Phase 5: Update tinkex_cookbook Recipes

**Goal:** Update recipes to use the corrected architecture.

**Changes:**
1. Recipes should use adapters from crucible_kitchen (where they still live)
2. Provide MetricsStore adapter from crucible_telemetry
3. Ensure telemetry handlers are attached for observability

**Example recipe config update:**
```elixir
# Add MetricsStore adapter
adapters: %{
  training_client: CrucibleKitchen.Adapters.Tinkex.TrainingClient,
  dataset_store: CrucibleKitchen.Adapters.HfDatasets.DatasetStore,
  metrics_store: CrucibleTelemetry.Adapters.JSONLMetrics  # NEW
}
```

**Validation:**
```bash
cd tinkex_cookbook
mix compile --warnings-as-errors
mix test
mix dialyzer
mix credo --strict
```

---

### Phase 6: Final Integration Testing

**Goal:** Verify the entire stack works end-to-end.

```bash
# Test each repo in dependency order
cd crucible_telemetry && mix test
cd crucible_kitchen && mix test
cd tinkex_cookbook && mix test

# Full validation
for dir in crucible_telemetry crucible_kitchen tinkex_cookbook; do
  echo "=== $dir ==="
  cd /home/home/p/g/North-Shore-AI/$dir
  mix compile --warnings-as-errors
  mix test
  mix dialyzer
  mix credo --strict
done
```

---

## Success Criteria

All of these MUST be true before considering the work complete:

1. **Zero duplicate ports** - Each port defined in exactly one place:
   - crucible_train: TrainingClient, DatasetStore, BlobStore, HubClient, LLMClient, EmbeddingClient, VectorStore
   - crucible_telemetry: MetricsStore
   - crucible_kitchen: Completer (only)

2. **Adapters implement correct behaviours** - Kitchen adapters use `@behaviour CrucibleTrain.Ports.*`

3. **SDK repos UNTOUCHED** - tinkex, hf_datasets_ex, hf_hub_ex have NO changes

4. **All tests pass**
   ```bash
   mix test  # in crucible_telemetry, crucible_kitchen, tinkex_cookbook
   ```

5. **No warnings**
   ```bash
   mix compile --warnings-as-errors  # in each repo
   ```

6. **Dialyzer clean**
   ```bash
   mix dialyzer  # in each repo
   ```

7. **Credo clean**
   ```bash
   mix credo --strict  # in each repo
   ```

---

## Common Pitfalls to Avoid

1. **DON'T modify SDK repos (tinkex, hf_datasets_ex, hf_hub_ex)**
   - If you think you need to, STOP - you're misunderstanding the architecture
   - SDK repos are INDEPENDENT with no crucible knowledge

2. **DON'T move adapters out of crucible_kitchen**
   - Adapters STAY in kitchen, they just change their @behaviour

3. **DON'T delete ports before updating adapters and stages**
   - This will break compilation

4. **DON'T skip the TDD cycle**
   - Read existing tests first
   - Update tests for new expectations
   - Then update implementation

5. **Check callback signatures carefully**
   - crucible_train ports may have different callback signatures than kitchen ports
   - Read the port behaviour definitions before updating adapters

---

## Verification Commands

After each phase:
```bash
mix deps.get
mix compile --warnings-as-errors
mix test
```

After all phases:
```bash
mix dialyzer
mix credo --strict
```

---

## File Changes Summary

### VERIFY ONLY (Phase 1 - already done)
```
crucible_telemetry/lib/crucible_telemetry/ports/metrics_store.ex
crucible_telemetry/lib/crucible_telemetry/adapters/jsonl_metrics.ex
```

### UPDATE in crucible_kitchen (Phases 2-4)
```
lib/crucible_kitchen/adapters/tinkex/training_client.ex  # @behaviour change
lib/crucible_kitchen/adapters/hf_datasets/dataset_store.ex  # @behaviour change
lib/crucible_kitchen/adapters/hf_hub/hub_client.ex  # @behaviour change
lib/crucible_kitchen/stages/*.ex  # import from crucible_train
```

### DELETE from crucible_kitchen (Phase 4)
```
lib/crucible_kitchen/ports/training_client.ex
lib/crucible_kitchen/ports/dataset_store.ex
lib/crucible_kitchen/ports/blob_store.ex
lib/crucible_kitchen/ports/hub_client.ex
lib/crucible_kitchen/ports/metrics_store.ex
```

### UPDATE in tinkex_cookbook (Phase 5)
```
lib/tinkex_cookbook/recipes/sl_basic_v2.ex
```

### DO NOT TOUCH
```
tinkex/*
hf_datasets_ex/*
hf_hub_ex/*
crucible_train/*
```

---

## Begin Implementation

**First, verify Phase 1 is complete:**
```bash
cd crucible_telemetry && mix test && mix dialyzer && mix credo --strict
```

**Then read the canonical architecture:**
```bash
cat crucible_kitchen/docs/corrected_ownership_model/00-canonical-architecture.md
```

**Then proceed with Phase 2.**

Good luck, agent.
