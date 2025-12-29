# Architectural Analysis: What Belongs Where

This document critically analyzes both the proposed implementation plan and the existing crucible_kitchen codebase to identify components that may belong in other crucible_* libraries.

---

## Section 1: Analysis of Proposed Plan

### Components Proposed for crucible_kitchen

| Component | Proposed Location | Should Be In | Reasoning |
|-----------|------------------|--------------|-----------|
| JSONL MetricsStore adapter | crucible_kitchen/adapters/jsonl/ | **crucible_telemetry** | Metrics storage is telemetry's domain |
| JSONL checkpoint metadata helper | crucible_kitchen/adapters/jsonl/ | **crucible_kitchen** | No port; helper only |
| MetricsStore port | crucible_kitchen/ports/ | **crucible_telemetry** | Port lives in telemetry |
| CheckpointStore port | crucible_kitchen/ports/ | **N/A** | Use TrainingClient/BlobStore instead |

### Critical Issues with Proposed Plan

#### 1. MetricsStore Belongs in crucible_telemetry

**Current situation:**
- `crucible_telemetry/ports/metrics_store.ex` defines the port
- `crucible_kitchen/ports/metrics_store.ex` has been removed
- `crucible_telemetry/adapters/jsonl_metrics.ex` implements JSONL storage

**Problem:** We're duplicating functionality. crucible_telemetry already provides:
- ETS-backed event storage
- JSONL export capability
- Analysis tools for stored metrics

**Recommendation:**
MetricsStore should live in crucible_telemetry (now done) with kitchen using the telemetry port.

```elixir
# Instead of this (in kitchen):
context = record_metric(context, :loss, 1.5, step: 0)

# Do this (delegating to telemetry):
CrucibleTelemetry.Store.record(run_id, :loss, 1.5, %{step: 0})
```

#### 2. Checkpoint Management Belongs in crucible_train

**Current situation:**
- Checkpoints are training artifacts (model weights, optimizer state)
- crucible_train already has `CrucibleTrain.Ports.BlobStore`
- crucible_train has training stages that understand checkpoints

**Problem:** Kitchen is trying to manage training-specific lifecycle events.

**Recommendation:**
1. Keep checkpoint metadata tracking in crucible_train
2. Kitchen orchestrates but doesn't own checkpoint semantics
3. Add `CrucibleTrain.CheckpointManager` module

```elixir
# Kitchen stage calls into train:
CrucibleTrain.CheckpointManager.save(session, step, %{
  epoch: epoch,
  global_step: step
})
```

#### 3. Training-Specific Stages May Belong in crucible_train

The following stages are tightly coupled to training semantics:

| Stage | Current Location | Consider Moving To |
|-------|-----------------|-------------------|
| ForwardBackward | crucible_kitchen | crucible_train |
| OptimStep | crucible_kitchen | crucible_train |
| SaveCheckpoint | crucible_kitchen | crucible_train |
| SaveFinalWeights | crucible_kitchen | crucible_train |
| BuildSupervisedDataset | crucible_kitchen | crucible_train |

**Counter-argument:** Kitchen's value is composing these operations. If stages live in crucible_train, kitchen becomes just a thin orchestrator.

**Recommendation:** Keep stages in kitchen but have them delegate to crucible_train for domain logic:

```elixir
# Kitchen stage (orchestration)
defmodule CrucibleKitchen.Stages.ForwardBackward do
  def execute(context) do
    # Kitchen handles context, adapters, state management
    batch = get_state(context, :current_batch)
    session = get_state(context, :session)

    # Delegate to train for domain logic
    case CrucibleTrain.Operations.forward_backward(session, batch) do
      {:ok, result} ->
        # Kitchen handles state updates
        context = put_state(context, :fb_result, result)
        context = put_state(context, :current_loss, result.loss)
        {:ok, context}
      error -> error
    end
  end
end
```

---

## Section 2: Analysis of Existing crucible_kitchen

### Port Duplication Problem

**crucible_train ports:**
```
crucible_train/lib/crucible_train/ports/
├── blob_store.ex
├── dataset_store.ex
├── embedding_client.ex
├── hub_client.ex
├── llm_client.ex
├── training_client.ex
└── vector_store.ex
```

**crucible_kitchen ports:**
```
crucible_kitchen/lib/crucible_kitchen/ports/
└── completer.ex
```

**All duplicate kitchen ports removed; only Completer remains.**

### Recommended Port Ownership

| Port | Should Own | Reasoning |
|------|------------|-----------|
| TrainingClient | crucible_train | Training-specific operations |
| DatasetStore | crucible_train | Training data management |
| BlobStore | crucible_train | Model/checkpoint storage |
| HubClient | crucible_train | Model hub operations |
| MetricsStore | **crucible_telemetry** | Telemetry/observability domain |
| Completer | crucible_kitchen | Workflow completion (orchestration) |
| VectorStore | crucible_train | RAG/embedding workflows |
| LLMClient | crucible_train | Inference operations |
| EmbeddingClient | crucible_train | Embedding generation |

### Adapters Stay in crucible_kitchen

**Current adapters:**
```
crucible_kitchen/lib/crucible_kitchen/adapters/
├── hf_datasets/dataset_store.ex    → Implements CrucibleTrain.Ports.DatasetStore
├── hf_hub/hub_client.ex            → Implements CrucibleTrain.Ports.HubClient
├── noop/                           → Testing infrastructure
└── tinkex/training_client.ex       → Implements CrucibleTrain.Ports.TrainingClient
```

**Recommendation:**
1. Keep adapters in crucible_kitchen to avoid SDK dependencies on crucible
2. crucible_train owns port behaviours; adapters implement them
3. SDK repos (tinkex, hf_*_ex) remain standalone

### Telemetry System Analysis

**Currently in crucible_kitchen:**
```elixir
# crucible_kitchen/lib/crucible_kitchen/telemetry.ex
defmodule CrucibleKitchen.Telemetry do
  @events [
    [:crucible_kitchen, :workflow, :run, :start],
    [:crucible_kitchen, :workflow, :run, :stop],
    [:crucible_kitchen, :stage, :run, :start],
    [:crucible_kitchen, :stage, :run, :stop],
    [:crucible_kitchen, :training, :step],
    [:crucible_kitchen, :training, :epoch],
    # ...
  ]

  def attach(:console, opts), do: # ...
  def attach(:jsonl, opts), do: # ...
end
```

**Already in crucible_telemetry:**
```elixir
# crucible_telemetry/lib/telemetry_research/handler.ex
# crucible_telemetry/lib/telemetry_research/store.ex
# crucible_telemetry/lib/telemetry_research/export/jsonl.ex
```

**Duplication analysis:**
- Kitchen has its own event definitions → OK (domain-specific)
- Kitchen has its own JSONL handler → DUPLICATE of telemetry's export
- Kitchen's console handler → Could use telemetry's handler

**Recommendation:**
1. Kitchen defines its OWN events (workflow, stage, training) - this is correct
2. Kitchen should USE crucible_telemetry for storage/export
3. Remove kitchen's built-in handlers, delegate to telemetry

```elixir
# Instead of kitchen having its own handlers:
CrucibleKitchen.Telemetry.attach(:jsonl, path: "...")

# Do this:
CrucibleTelemetry.attach_handler(
  "kitchen-metrics",
  CrucibleKitchen.Telemetry.events(),
  CrucibleTelemetry.Handlers.JSONL,
  %{path: "..."}
)
```

### Workflow DSL - Correctly Placed

The workflow DSL is appropriately in kitchen:
- `workflow do ... end` macro
- `stage/2`, `loop/3`, `conditional/2` constructs
- Stage execution and error handling
- Context management

This is kitchen's core value proposition and should stay.

### Stages Analysis

| Stage | Kitchen Responsibility | Train Responsibility |
|-------|----------------------|---------------------|
| LoadDataset | Context management, adapter dispatch | Dataset semantics |
| InitSession | State management | Session lifecycle |
| ForwardBackward | State updates, error handling | Loss computation |
| OptimStep | LR scheduling, state | Optimizer math |
| LogStepMetrics | When to log (throttling) | What to log |
| SaveCheckpoint | Conditional logic | Checkpoint format |

**Current problem:** Stages in kitchen directly call Tinkex/HfDatasets APIs instead of going through crucible_train abstractions.

**Example - BuildSupervisedDataset:**
```elixir
# Current (in kitchen):
CrucibleTrain.Supervised.Dataset.create(samples, opts)  # Direct call!

# Should be:
DatasetStore.build_supervised(context, samples, opts)  # Through port
```

---

## Section 3: Recommended Architecture

### Layer Diagram

```
┌────────────────────────────────────────────────────────────────────┐
│                        tinkex_cookbook                              │
│                   (Recipes + Config only)                           │
└─────────────────────────────┬──────────────────────────────────────┘
                              │
┌─────────────────────────────▼──────────────────────────────────────┐
│                      crucible_kitchen                               │
│  Workflow DSL │ Stage Orchestration │ Context Management           │
│  Recipe Behavior │ Adapter Dispatch │ Error Handling               │
└─────────────────────────────┬──────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│ crucible_train│    │crucible_telem │    │  crucible_ir  │
│               │    │               │    │               │
│ Ports:        │    │ MetricsStore  │    │ Experiment    │
│ - Training    │    │ EventStore    │    │ Definitions   │
│ - Dataset     │    │ Export (JSONL)│    │ StageSpecs    │
│ - Blob        │    │ Analysis      │    │               │
│ - Tokenizer   │    │               │    │               │
│               │    │               │    │               │
│ Domain Logic: │    │               │    │               │
│ - Supervised  │    │               │    │               │
│ - RL          │    │               │    │               │
│ - DPO         │    │               │    │               │
└───────┬───────┘    └───────────────┘    └───────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────────────┐
│              External Services & Standalone Libs                   │
│  ┌─────────┐  ┌─────────────┐  ┌─────────┐  ┌─────────────────┐  │
│  │ tinkex  │  │hf_datasets_ex│  │ hf_hub  │  │ snakebridge     │  │
│  │ (SDK)   │  │             │  │         │  │ (Python bridge) │  │
│  └─────────┘  └─────────────┘  └─────────┘  └─────────────────┘  │
└───────────────────────────────────────────────────────────────────┘
```

### Migration Plan

#### Phase 1: Port Consolidation (Prerequisite)

1. **Audit crucible_train ports** - ensure they're complete
2. **Remove duplicate ports from kitchen** - use train's
3. **Move MetricsStore port to crucible_telemetry**

```
# After consolidation:
crucible_train/ports/
├── training_client.ex
├── dataset_store.ex
├── blob_store.ex
├── hub_client.ex
├── llm_client.ex
├── embedding_client.ex
└── vector_store.ex

crucible_telemetry/ports/
└── metrics_store.ex

crucible_kitchen/ports/
└── completer.ex  # Only kitchen-specific port
```

#### Phase 2: Adapter Alignment

1. **Keep adapters in crucible_kitchen** to avoid SDK dependencies
2. **Ensure adapters implement CrucibleTrain ports**
3. **Keep SDK repos (tinkex, hf_*_ex) standalone**
4. **Kitchen imports these adapters**, doesn't define them

#### Phase 3: Telemetry Integration

1. **Remove kitchen's JSONL handler**
2. **Use crucible_telemetry for all storage/export**
3. **Kitchen emits events, telemetry handles them**

#### Phase 4: Stage Refactoring

1. **Stages remain in kitchen** (orchestration)
2. **Stages delegate to crucible_train** for domain logic
3. **LogStepMetrics calls crucible_telemetry** for storage

---

## Section 4: Revised Implementation Plan

Given the architectural analysis, here's the corrected plan:

### What to Build in Each Repo

#### crucible_telemetry (NEW WORK)

1. **Define MetricsStore port** (move from kitchen)
2. **Create JSONL MetricsStore adapter**
3. **Integrate with existing Store/Export**

```elixir
# New file: crucible_telemetry/lib/crucible_telemetry/ports/metrics_store.ex
defmodule CrucibleTelemetry.Ports.MetricsStore do
  @callback record(run_id, metric_name, value, opts) :: :ok | {:error, term()}
  @callback get_history(run_id, metric_name, opts) :: {:ok, list()} | {:error, term()}
  # ...
end

# New file: crucible_telemetry/lib/crucible_telemetry/adapters/jsonl_metrics.ex
defmodule CrucibleTelemetry.Adapters.JSONLMetrics do
  @behaviour CrucibleTelemetry.Ports.MetricsStore
  # Implementation (from our plan)
end
```

#### crucible_train (NEW WORK)

1. **Add CheckpointManager module**
2. **Ensure ports are complete and canonical**
3. **Adapters live here OR in their SDK repos**

```elixir
# New file: crucible_train/lib/crucible_train/checkpoint_manager.ex
defmodule CrucibleTrain.CheckpointManager do
  def save(session, name, metadata), do: # ...
  def load(session, name), do: # ...
  def list(run_id), do: # ...
  def get_latest(run_id, required_key), do: # ...
end
```

#### crucible_kitchen (REFACTOR)

1. **Remove duplicate ports** (use crucible_train's)
2. **Remove built-in JSONL handler** (use crucible_telemetry)
3. **Update stages to call crucible_telemetry for metrics**
4. **Keep: Workflow DSL, Stage behavior, Context, Recipe behavior**

```elixir
# Updated LogStepMetrics
def execute(context) do
  # ... collect metrics ...

  # Call telemetry instead of local storage
  CrucibleTelemetry.Adapters.JSONLMetrics.record(
    run_id,
    :loss,
    current_loss,
    step: global_step
  )

  # Emit event (kitchen's responsibility)
  :telemetry.execute([:crucible_kitchen, :training, :step], measurements, metadata)

  {:ok, context}
end
```

#### tinkex_cookbook (MINOR CHANGES)

1. **Update sl_basic_v2** to use new architecture
2. **Configure telemetry handlers in run/2**
3. **No need to provide MetricsStore adapter** (comes from telemetry)

```elixir
def run(config, opts \\ []) do
  log_path = Keyword.get(opts, :log_path, @default_log_path)

  # Attach telemetry handler from crucible_telemetry
  CrucibleTelemetry.attach_jsonl_handler(
    "sl_basic_v2",
    path: Path.join(log_path, "metrics.jsonl")
  )

  # Adapters only need training-related ones
  adapters = %{
    training_client: {Tinkex.Adapters.TrainingClient, opts},
    dataset_store: {HfDatasetsEx.Adapters.DatasetStore, []}
  }

  CrucibleKitchen.run(__MODULE__, config, adapters: adapters)
end
```

---

## Section 5: Summary

### What's Wrong Today

1. **Port duplication** - kitchen redefines train's ports
2. **Adapter location** - SDK adapters in kitchen instead of their repos
3. **Telemetry duplication** - kitchen has handlers that telemetry provides
4. **Tight coupling** - stages call external APIs directly

### Corrected Ownership

| Concern | Owner | Kitchen's Role |
|---------|-------|----------------|
| Port definitions (training) | crucible_train | Import & use |
| Port definitions (metrics) | crucible_telemetry | Import & use |
| SDK adapters | Individual SDKs | Import & use |
| Metrics storage | crucible_telemetry | Emit events |
| Checkpoint tracking | crucible_train | Orchestrate |
| Workflow DSL | crucible_kitchen | Own |
| Stage orchestration | crucible_kitchen | Own |
| Context management | crucible_kitchen | Own |

### Effort Estimate (Revised)

| Task | Repo | Effort |
|------|------|--------|
| Move MetricsStore port | crucible_telemetry | 1h |
| Create JSONL MetricsStore adapter | crucible_telemetry | 2h |
| Add CheckpointManager | crucible_train | 2h |
| Remove duplicate ports from kitchen | crucible_kitchen | 1h |
| Update LogStepMetrics to use telemetry | crucible_kitchen | 1h |
| Move/relocate adapters to SDK repos | tinkex, hf_*_ex | 3h |
| Update sl_basic_v2 | tinkex_cookbook | 1h |
| **Total** | | **11h** |

This is slightly more work than the original plan but results in a cleaner architecture with proper separation of concerns.
