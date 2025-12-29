# Canonical Architecture: Corrected Ownership Model

**Date:** 2025-12-28 (Revised)
**Status:** CANONICAL - This supersedes all prior architecture documents
**Purpose:** Define the authoritative ownership of components across the crucible ecosystem

---

## Executive Summary

Previous designs placed too much functionality in crucible_kitchen, creating duplication with crucible_train and crucible_telemetry. This document establishes the **corrected ownership model** where:

1. **crucible_train** owns all training-related ports (behaviours/interfaces)
2. **crucible_telemetry** owns MetricsStore port and metrics storage
3. **crucible_kitchen** is orchestration + adapters that WRAP external SDKs
4. **SDK repos** (tinkex, hf_*_ex) remain INDEPENDENT with no crucible dependencies

---

## CRITICAL: Dependency Direction

```
                     CORRECT                          WRONG

crucible_kitchen ──depends on──► tinkex      tinkex ──depends on──► crucible_train
       │                                            │
       └── wraps tinkex in adapter                  └── THIS INVERTS DEPENDENCIES
           implementing crucible port                   SDK repos must stay independent!
```

**Adapters WRAP external SDKs. They don't live IN external SDKs.**

SDKs (tinkex, hf_datasets_ex, hf_hub_ex) are independent libraries. They have NO knowledge of crucible. The crucible ecosystem adapts TO them, not vice versa.

---

## The Problem: Port Duplication

**Current state:**

```
crucible_train/ports/          crucible_kitchen/ports/
├── blob_store.ex              └── completer.ex
├── dataset_store.ex
├── embedding_client.ex
├── hub_client.ex
├── llm_client.ex
├── training_client.ex
└── vector_store.ex
```

**Duplicate kitchen ports removed; only Completer remains.**

This violates the principle: "If crucible_train already does it, delegate don't duplicate."

---

## Corrected Ownership Model

### Layer Diagram

```
┌────────────────────────────────────────────────────────────────────┐
│                        tinkex_cookbook                              │
│                   (Recipes + Config ONLY)                           │
│            - Recipe definitions (name, workflow, config)            │
│            - NO adapters, NO ports, NO domain logic                 │
└─────────────────────────────┬──────────────────────────────────────┘
                              │
┌─────────────────────────────▼──────────────────────────────────────┐
│                      crucible_kitchen                               │
│              (Orchestration + SDK Adapters)                         │
│                                                                     │
│  OWNS:                                                              │
│  - Workflow DSL (stage, loop, conditional)                          │
│  - Stage behaviour (execute, validate, rollback)                    │
│  - Context management (config, state, adapters)                     │
│  - Recipe behaviour (name, workflow, validate_config)               │
│  - Event emission ([:crucible_kitchen, :*])                         │
│  - Completer port (workflow-specific)                               │
│  - ADAPTERS that wrap external SDKs (tinkex, hf_*, etc.)            │
│                                                                     │
│  DOES NOT OWN:                                                      │
│  - Training ports (use crucible_train's)                            │
│  - Metrics storage (use crucible_telemetry's)                       │
│  - Domain logic (delegate to crucible_train)                        │
└─────────────────────────────┬──────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│ crucible_train│    │crucible_telem │    │  crucible_ir  │
│               │    │               │    │               │
│ OWNS:         │    │ OWNS:         │    │ OWNS:         │
│ - All ports   │    │ - MetricsStore│    │ - Experiment  │
│   (behaviours)│    │   port        │    │   specs       │
│ - Types/Datum │    │ - EventStore  │    │ - Config      │
│ - Renderers   │    │ - JSONL export│    │   schemas     │
│ - Noop adptrs │    │ - Analysis    │    │               │
└───────────────┘    └───────────────┘    └───────────────┘

        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│    tinkex     │    │hf_datasets_ex │    │  hf_hub_ex    │
│  (INDEPENDENT)│    │  (INDEPENDENT)│    │  (INDEPENDENT)│
│               │    │               │    │               │
│ Pure SDK for  │    │ Pure SDK for  │    │ Pure SDK for  │
│ Tinker API    │    │ HF Datasets   │    │ HF Hub        │
│               │    │               │    │               │
│ NO crucible   │    │ NO crucible   │    │ NO crucible   │
│ dependencies! │    │ dependencies! │    │ dependencies! │
└───────────────┘    └───────────────┘    └───────────────┘
```

---

## Port Ownership (CANONICAL)

### crucible_train Owns These Ports

| Port | Purpose | Location |
|------|---------|----------|
| TrainingClient | ML training operations | `crucible_train/ports/training_client.ex` |
| DatasetStore | Dataset loading/streaming | `crucible_train/ports/dataset_store.ex` |
| BlobStore | Artifact storage | `crucible_train/ports/blob_store.ex` |
| HubClient | Model hub operations | `crucible_train/ports/hub_client.ex` |
| LLMClient | LLM inference | `crucible_train/ports/llm_client.ex` |
| EmbeddingClient | Embedding generation | `crucible_train/ports/embedding_client.ex` |
| VectorStore | Vector search | `crucible_train/ports/vector_store.ex` |

### crucible_telemetry Owns These Ports

| Port | Purpose | Location |
|------|---------|----------|
| MetricsStore | Training metrics storage | `crucible_telemetry/ports/metrics_store.ex` |
| EventStore | Telemetry event storage | `crucible_telemetry/store.ex` (existing) |

### crucible_kitchen Owns These

| Component | Purpose | Location |
|-----------|---------|----------|
| Completer | Workflow completion tracking | `crucible_kitchen/ports/completer.ex` |

---

## Adapter Ownership (CANONICAL)

**Adapters that wrap external SDKs stay in crucible_kitchen.**
**They implement crucible_train ports by delegating to SDK functions.**

| Adapter | Implements Port | Wraps SDK | Location |
|---------|-----------------|-----------|----------|
| Tinkex.TrainingClient | CrucibleTrain.Ports.TrainingClient | tinkex | `crucible_kitchen/adapters/tinkex/` |
| HfDatasets.DatasetStore | CrucibleTrain.Ports.DatasetStore | hf_datasets_ex | `crucible_kitchen/adapters/hf_datasets/` |
| HfHub.HubClient | CrucibleTrain.Ports.HubClient | hf_hub_ex | `crucible_kitchen/adapters/hf_hub/` |
| JSONLMetrics | CrucibleTelemetry.Ports.MetricsStore | (none) | `crucible_telemetry/adapters/` |
| Noop adapters | Various | (none) | `crucible_kitchen/adapters/noop/` |

---

## Stage Ownership

Stages live in crucible_kitchen but DELEGATE to domain libraries:

```elixir
# Stage in crucible_kitchen (OWNS orchestration)
defmodule CrucibleKitchen.Stages.ForwardBackward do
  use CrucibleKitchen.Stage

  def execute(context) do
    # Kitchen handles: context, state, error handling
    session = get_state(context, :session)
    batch = get_state(context, :current_batch)

    # DELEGATE to crucible_train port (train owns the port)
    ports = get_train_ports(context)
    future = CrucibleTrain.Ports.TrainingClient.forward_backward(ports, session, batch)

    case CrucibleTrain.Ports.TrainingClient.await(ports, future) do
      {:ok, result} ->
        context = put_state(context, :fb_result, result)
        {:ok, context}
      {:error, reason} ->
        {:error, {:forward_backward_failed, reason}}
    end
  end
end
```

---

## Telemetry Architecture

### Event Emission (crucible_kitchen owns)

```elixir
# Kitchen defines and emits its events
:telemetry.execute(
  [:crucible_kitchen, :training, :step],
  %{loss: loss, lr: lr},
  %{step: step, run_id: run_id}
)
```

### Event Storage (crucible_telemetry owns)

```elixir
# Telemetry handles storage
CrucibleTelemetry.Store.record(run_id, event, measurements, metadata)
```

### Metrics Storage (crucible_telemetry owns)

```elixir
# Metrics port defined in telemetry
CrucibleTelemetry.Ports.MetricsStore.record(
  adapter,
  run_id,
  :loss,
  value,
  step: step
)
```

---

## What Changes Are Required

### Phase 1: MetricsStore in crucible_telemetry (DONE)

1. **CREATE** MetricsStore port in crucible_telemetry
2. **CREATE** JSONL adapter in crucible_telemetry
3. **DELETE** MetricsStore port from crucible_kitchen (after stages updated)

### Phase 2: Port Consolidation in crucible_kitchen

1. **DELETE** duplicate ports from crucible_kitchen:
   - `crucible_kitchen/ports/training_client.ex` → use `CrucibleTrain.Ports.TrainingClient`
   - `crucible_kitchen/ports/dataset_store.ex` → use `CrucibleTrain.Ports.DatasetStore`
   - `crucible_kitchen/ports/blob_store.ex` → use `CrucibleTrain.Ports.BlobStore`
   - `crucible_kitchen/ports/hub_client.ex` → use `CrucibleTrain.Ports.HubClient`
   - `crucible_kitchen/ports/tokenizer_client.ex` → no port; use adapter-specific helpers

2. **KEEP** only Completer port in kitchen

3. **UPDATE** adapters to implement `CrucibleTrain.Ports.*` behaviours instead of kitchen ports

### Phase 3: Stage Updates

1. Update all stages to import ports from crucible_train
2. Update LogStepMetrics to use crucible_telemetry.Ports.MetricsStore
3. Ensure stages delegate domain logic to crucible_train

### Phase 4: Recipe Updates

1. Update tinkex_cookbook recipes to:
   - Use adapters from crucible_kitchen (they stay there)
   - Provide MetricsStore adapter from crucible_telemetry
   - Attach telemetry handlers for observability

---

## Dependency Graph (Corrected)

```
tinkex_cookbook
    │
    └── crucible_kitchen (orchestration + adapters)
            │
            ├── crucible_train (ports/behaviours, types, domain logic)
            │
            ├── crucible_telemetry (metrics storage)
            │
            ├── crucible_ir (experiment specs)
            │
            ├── tinkex (SDK - INDEPENDENT, no crucible deps)
            │
            ├── hf_datasets_ex (SDK - INDEPENDENT, no crucible deps)
            │
            └── hf_hub_ex (SDK - INDEPENDENT, no crucible deps)
```

---

## Success Criteria

1. **Zero duplicate ports** - Each port defined in exactly one place
2. **Adapters wrap SDKs** - Adapters in kitchen implement train/telemetry ports by wrapping SDKs
3. **SDK repos independent** - tinkex, hf_*_ex have NO crucible dependencies
4. **Kitchen stays thin** - Only workflow DSL, stages, context, adapters
5. **All tests pass** - No regressions
6. **No warnings/errors** - Clean compilation
7. **Dialyzer clean** - No type errors
8. **Credo clean** - No style issues

---

## File Changes Summary

### DELETE from crucible_kitchen

```
lib/crucible_kitchen/ports/training_client.ex
lib/crucible_kitchen/ports/dataset_store.ex
lib/crucible_kitchen/ports/blob_store.ex
lib/crucible_kitchen/ports/hub_client.ex
lib/crucible_kitchen/ports/metrics_store.ex
```

### CREATE in crucible_telemetry (DONE)

```
lib/crucible_telemetry/ports/metrics_store.ex
lib/crucible_telemetry/adapters/jsonl_metrics.ex
```

### UPDATE in crucible_kitchen

```
lib/crucible_kitchen/adapters/tinkex/*.ex  (implement CrucibleTrain.Ports.*)
lib/crucible_kitchen/adapters/hf_datasets/*.ex  (implement CrucibleTrain.Ports.*)
lib/crucible_kitchen/adapters/hf_hub/*.ex  (implement CrucibleTrain.Ports.*)
lib/crucible_kitchen/stages/*.ex  (import from crucible_train)
lib/crucible_kitchen/ports.ex  (remove re-exports of train ports)
```

### UPDATE in tinkex_cookbook

```
lib/tinkex_cookbook/recipes/sl_basic_v2.ex  (select adapters from crucible_kitchen/telemetry)
```

### DO NOT TOUCH (SDK repos)

```
tinkex/*  ← INDEPENDENT, no changes needed
hf_datasets_ex/*  ← INDEPENDENT, no changes needed
hf_hub_ex/*  ← INDEPENDENT, no changes needed
```
