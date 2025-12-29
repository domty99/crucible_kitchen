# Crucible Kitchen: Industrial ML Training Infrastructure

**Date:** 2025-12-27
**Status:** Architecture Design
**Vision:** Backend-agnostic ML training orchestration that makes tinkex_cookbook (and future cookbooks) trivially thin

---

## The Problem

We have built an extensive crucible ecosystem:
- `crucible_train` (8,249 LOC) - Types, Renderers, Training loops, Ports
- `crucible_ir` (3,177 LOC) - Experiment specs, Training configs
- `crucible_framework` (2,500+ LOC) - Pipeline runner, Stage lifecycle
- `crucible_telemetry`, `crucible_harness`, `crucible_bench`, etc.

Yet `tinkex_cookbook` bypasses all of it to make direct API calls. The "industrial kitchen" we built sits unused.

**Root cause**: There's no unified layer that:
1. Wires crucible components together coherently
2. Provides a clean API for recipe authors
3. Handles backend-agnostic orchestration
4. Makes the right thing easy and the wrong thing hard

---

## The Solution: Crucible Kitchen

**crucible_kitchen** is the missing orchestration layer - a backend-agnostic industrial core that:

1. **Composes** the crucible ecosystem into unified workflows
2. **Abstracts** backend-specific details behind clean ports
3. **Orchestrates** training pipelines with full telemetry and persistence
4. **Enables** thin cookbook frontends (tinkex_cookbook, fireworks_cookbook, etc.)

```
┌─────────────────────────────────────────────────────────────────┐
│                    COOKBOOK FRONTENDS                            │
│  tinkex_cookbook    fireworks_cookbook    modal_cookbook        │
│  (config + adapters only, <2K LOC each)                         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     CRUCIBLE KITCHEN                             │
│  The Industrial Core - Backend-Agnostic Orchestration           │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   Recipes    │  │  Workflows   │  │  Telemetry   │          │
│  │  (SL/RL/DPO) │  │  (Pipelines) │  │  (Metrics)   │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │    Ports     │  │   Registry   │  │  Evaluation  │          │
│  │  (Backends)  │  │  (Artifacts) │  │  (Harness)   │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└─────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│ crucible_    │      │ crucible_    │      │ crucible_    │
│ train        │      │ framework    │      │ ir           │
└──────────────┘      └──────────────┘      └──────────────┘
```

---

## What Crucible Kitchen Is NOT

1. **Not another layer of abstraction for abstraction's sake**
   - Every module must add concrete value
   - If crucible_train already does it, delegate don't duplicate

2. **Not a kitchen-sink framework**
   - Focused on ML training orchestration
   - Not general-purpose infrastructure

3. **Not Tinker-specific**
   - Zero Tinker/Tinkex dependencies in core
   - Tinker is ONE adapter among many

---

## Core Design Principles

### 1. Inversion of Control
Cookbooks don't call the kitchen; the kitchen orchestrates and uses adapters defined in crucible_kitchen.

```elixir
# Kitchen adapters (selected by cookbook recipes/config)
adapters = %{
  training_client: {CrucibleKitchen.Adapters.Tinkex.TrainingClient, []},
  dataset_store: {CrucibleKitchen.Adapters.HfDatasets.DatasetStore, []}
}

# Kitchen orchestrates using those adapters
CrucibleKitchen.run(:sl_basic, config, adapters: adapters)
```

### 2. Workflow-Centric (Not Just Ports/Adapters)
Hexagonal architecture provides ports/adapters, but we need more:
- **Workflows**: Composable sequences of operations
- **Stages**: Lifecycle-managed execution units
- **Events**: Telemetry throughout the pipeline

### 3. Configuration-Driven
Recipes are mostly configuration; the kitchen does the work.

```elixir
# Recipe is just config + adapter selection
%Recipe{
  name: :sl_basic,
  workflow: :supervised_training,
  config: %{
    model: "meta-llama/Llama-3.1-8B",
    epochs: 3,
    batch_size: 128
  },
  adapters: %{
    training_client: TinkexAdapter,
    dataset_store: HfDatasetsAdapter
  }
}
```

### 4. Observable by Default
Every operation emits telemetry; nothing is invisible.

```elixir
:telemetry.execute([:crucible_kitchen, :training, :step],
  %{loss: 0.5, lr: 0.0002, tokens_per_sec: 15000},
  %{recipe: :sl_basic, step: 42, epoch: 1})
```

---

## Document Index

| Document | Purpose |
|----------|---------|
| [01_ARCHITECTURE_PATTERNS.md](./01_ARCHITECTURE_PATTERNS.md) | Architectural options analysis |
| [02_COMPONENT_DESIGN.md](./02_COMPONENT_DESIGN.md) | Core component specifications |
| [03_WORKFLOW_ENGINE.md](./03_WORKFLOW_ENGINE.md) | Workflow/pipeline design |
| [04_PORT_CONTRACTS.md](./04_PORT_CONTRACTS.md) | Port behavior definitions |
| [05_TELEMETRY_DESIGN.md](./05_TELEMETRY_DESIGN.md) | Observability architecture |
| [06_API_SURFACE.md](./06_API_SURFACE.md) | Public API design |
| [07_MIGRATION_GUIDE.md](./07_MIGRATION_GUIDE.md) | tinkex_cookbook transformation |
| [08_IMPLEMENTATION_ROADMAP.md](./08_IMPLEMENTATION_ROADMAP.md) | Phased delivery plan |

---

## Success Metrics

| Metric | Target |
|--------|--------|
| tinkex_cookbook LOC after migration | < 2,000 |
| New backend implementation time | < 1 day |
| Recipe implementation time | < 2 hours |
| Telemetry coverage | 100% of operations |
| Test coverage | > 90% |

---

## Key Innovation

**What makes this different from existing ML frameworks:**

1. **Elixir-native**: Leverages OTP for fault tolerance, supervision, and concurrency
2. **Composition over configuration**: Pipelines are composed, not configured
3. **First-class observability**: Telemetry is not an afterthought
4. **Backend-agnostic by design**: Not "portable" as a feature, but agnostic as the core design
5. **Research-grade reproducibility**: Full audit trails, deterministic seeds, artifact versioning

This is the "industrial kitchen" that will make ML training in Elixir genuinely competitive with Python ecosystems while being more reliable and observable.
