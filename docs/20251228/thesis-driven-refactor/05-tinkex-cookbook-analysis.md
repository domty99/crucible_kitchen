# tinkex_cookbook Refactoring Analysis

## Executive Summary

**Goal**: Refactor tinkex_cookbook to be a thin configuration layer on top of crucible_kitchen.

| Metric | Before | After |
|--------|--------|-------|
| Files | 32 | 8 |
| LOC | ~3K | ~1.5K |
| Focus | Mixed | Recipes only |

## Current tinkex_cookbook Structure

```
tinkex_cookbook/lib/
├── adapters/           (14 files) - 7 port types
├── datasets/           (1 file)
├── eval/               (3 files)
├── recipes/            (3 files) - sl_basic
├── runtime/            (2 files)
└── utils/              (6 files)
```

**Key characteristics:**
- No OTP supervision (library)
- Runtime.Manifests for port wiring
- ChzEx config schemas
- Direct Tinkex integration

## Target State

### STAY in tinkex_cookbook (minimal)
1. **Recipes** (sl_basic, rl, dpo, distillation)
2. **Recipe config schemas** (ChzEx)
3. **Tinkex-specific adapter** (training_client only)
4. **Config files**

### MOVE to crucible_kitchen
1. **Generic adapters** (dataset_store, blob_store, hub_client)
2. **Datasets** (NoRobots, etc.)
3. **Evaluation utilities**
4. **Tinkex adapter** (for cross-cookbook reuse)

## Port Mapping

| Current (tinkex_cookbook) | New (crucible_kitchen) |
|---------------------------|------------------------|
| `TrainingClient.Tinkex` | `Adapters.Tinkex.TrainingClient` |
| `DatasetStore.HfDatasets` | `Adapters.HfDatasets.DatasetStore` |
| `HubClient.HfHub` | `Adapters.HfHub.HubClient` |
| `BlobStore.Local` | `Adapters.Local.BlobStore` |

## Recipe Pattern

**Before (custom behaviour):**
```elixir
defmodule TinkexCookbook.Recipes.SlBasic do
  @behaviour TinkexCookbook.Recipe
  def build_spec(config), do: %CrucibleIR.Experiment{...}
end
```

**After (CrucibleKitchen.Recipe):**
```elixir
defmodule TinkexCookbook.Recipes.SlBasic do
  use CrucibleKitchen.Recipe

  def name, do: :sl_basic
  def workflow, do: CrucibleKitchen.Workflows.Supervised
  def default_config, do: %{...}
  def required_adapters, do: [:training_client, :dataset_store]
end
```

## Configuration Pattern

**Before (Manifests):**
```elixir
TinkexCookbook.Runtime.Manifests.get(:prod)
```

**After (App config):**
```elixir
# config/prod.exs
config :tinkex_cookbook,
  adapters: %{
    training_client: {CrucibleKitchen.Adapters.Tinkex.TrainingClient, [...]},
    dataset_store: CrucibleKitchen.Adapters.HfDatasets.DatasetStore
  }
```

## Migration Checklist

### Phase 1: Preparation
- [ ] Create Tinkex adapter in crucible_kitchen
- [ ] Move generic adapters
- [ ] Add noop adapters for all ports

### Phase 2: Recipe Refactoring
- [ ] Update sl_basic to use CrucibleKitchen.Recipe
- [ ] Remove old TinkexCookbook.Recipe behaviour
- [ ] Remove Runtime facade

### Phase 3: Configuration
- [ ] Remove Manifests module
- [ ] Implement app config
- [ ] Update mix.exs for OTP app (optional)

### Phase 4: Testing
- [ ] Adapter validation tests
- [ ] Recipe integration tests

## Impact Summary

```
TINKEX_COOKBOOK: 32 files → 8 files (-75%)
CRUCIBLE_KITCHEN: 40 files → 60+ files (+50%)

Reusability:       +100%
Maintenance:       -50%
Code Duplication:  -70%
Time to New Cookbook: 2-3 days → 4-6 hours
```
