# Architecture Patterns Analysis

**Decision**: Which architectural pattern best serves crucible_kitchen?

---

## Options Evaluated

### Option A: Pure Hexagonal Architecture

```
        ┌─────────────────────────────────────┐
        │           APPLICATION               │
        │  (Use Cases / Business Logic)       │
        └─────────────────────────────────────┘
                 ▲                 ▲
    ┌────────────┴───┐       ┌────┴────────────┐
    │  DRIVING PORTS │       │  DRIVEN PORTS   │
    │  (Primary)     │       │  (Secondary)    │
    │  - CLI         │       │  - TrainingAPI  │
    │  - HTTP        │       │  - Storage      │
    │  - gRPC        │       │  - Datasets     │
    └────────────────┘       └─────────────────┘
```

**Pros:**
- Clean port/adapter separation
- Well-understood pattern
- Easy to test (mock ports)

**Cons:**
- Doesn't address workflow composition
- No built-in lifecycle management
- Telemetry is an afterthought

**Verdict:** Necessary but insufficient.

---

### Option B: Clean Architecture (Onion)

```
┌─────────────────────────────────────────────────┐
│                  FRAMEWORKS                      │
│  (Phoenix, Ecto, Telemetry, HTTP clients)       │
├─────────────────────────────────────────────────┤
│              INTERFACE ADAPTERS                  │
│  (Controllers, Gateways, Presenters)            │
├─────────────────────────────────────────────────┤
│              APPLICATION LAYER                   │
│  (Use Cases, Orchestrators)                     │
├─────────────────────────────────────────────────┤
│                   DOMAIN                         │
│  (Entities, Value Objects, Domain Services)     │
└─────────────────────────────────────────────────┘
```

**Pros:**
- Dependency rule (inner layers don't know outer)
- Domain isolation
- Framework independence

**Cons:**
- Heavy ceremony for our use case
- Domain model for ML training is thin
- Over-engineering for workflow orchestration

**Verdict:** Too heavy; we're not building a complex domain.

---

### Option C: Pipeline Architecture

```
Input → Stage 1 → Stage 2 → Stage 3 → Output
           ↓          ↓          ↓
        Metrics    Metrics    Metrics
           ↓          ↓          ↓
        ┌──────────────────────────────┐
        │       Telemetry Sink         │
        └──────────────────────────────┘
```

**Pros:**
- Natural fit for ML training workflows
- CrucibleFramework already provides this
- Easy to reason about

**Cons:**
- Doesn't address the port/adapter pattern
- Linear pipelines don't capture all workflows (e.g., RL with rollouts)

**Verdict:** Core pattern, but needs augmentation.

---

### Option D: Actor-Based (OTP Native)

```
┌─────────────────────────────────────────────────┐
│              Supervisor Tree                     │
├─────────────────────────────────────────────────┤
│  Kitchen.Supervisor                              │
│    ├── Workflow.Supervisor                       │
│    │     ├── Workflow.Runner (GenServer)        │
│    │     └── Stage workers (Tasks)              │
│    ├── Telemetry.Aggregator (GenServer)         │
│    ├── Registry (Registry)                       │
│    └── Adapter.Pool (Poolboy/NimblePool)        │
└─────────────────────────────────────────────────┘
```

**Pros:**
- Fault tolerance built-in
- Concurrent execution natural
- OTP patterns are Elixir-native

**Cons:**
- Can over-complicate simple workflows
- Debugging distributed state is harder
- May be overkill for single-node training

**Verdict:** Use for concurrency/supervision, not as primary architecture.

---

## Recommended: Hybrid "Kitchen Architecture"

Combine the best of each pattern:

```
┌─────────────────────────────────────────────────────────────────┐
│                        RECIPE LAYER                              │
│  Configuration-driven recipe definitions                         │
│  (What to cook, not how)                                        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      WORKFLOW LAYER                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │ Supervised  │  │     RL      │  │    DPO      │             │
│  │  Workflow   │  │  Workflow   │  │  Workflow   │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
│  Composable workflow templates with lifecycle hooks             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       STAGE LAYER                                │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐  │
│  │  Load   │ │ Render  │ │  Train  │ │  Eval   │ │  Save   │  │
│  │ Dataset │ │ Messages│ │  Step   │ │  Step   │ │Checkpoint│  │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘  │
│  Reusable stage implementations with telemetry                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        PORT LAYER                                │
│  ┌───────────────┐ ┌───────────────┐ ┌───────────────┐         │
│  │TrainingClient │ │ DatasetStore  │ │  BlobStore    │         │
│  └───────────────┘ └───────────────┘ └───────────────┘         │
│  ┌───────────────┐ ┌───────────────┐ ┌───────────────┐         │
│  │  HubClient    │ │ VectorStore   │ │ MetricsStore  │         │
│  └───────────────┘ └───────────────┘ └───────────────┘         │
│  Behaviour contracts for external integrations                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      ADAPTER LAYER                               │
│  (Provided by cookbooks, not crucible_kitchen)                  │
│  TinkexAdapter, FireworksAdapter, LocalNxAdapter, NoopAdapter   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Kitchen Architecture Principles

### 1. Layered with Clear Boundaries

Each layer has a single responsibility:
- **Recipe**: What to do (configuration)
- **Workflow**: How to orchestrate (composition)
- **Stage**: Individual operations (implementation)
- **Port**: External contracts (abstraction)
- **Adapter**: External implementations (provided by cookbooks)

### 2. Inversion at the Adapter Boundary

CrucibleTrain and CrucibleTelemetry define ports; crucible_kitchen provides adapters. Cookbooks provide recipes/config only.
```elixir
# Port (in crucible_train):
defmodule CrucibleTrain.Ports.TrainingClient do
  @callback start_session(adapter_opts, config :: map()) :: {:ok, session} | {:error, term()}
  @callback forward_backward(adapter_opts, session, datums) :: future
  @callback optim_step(adapter_opts, session, lr :: float()) :: future
  @callback await(adapter_opts, future) :: {:ok, map()} | {:error, term()}
  # ...
end

# Kitchen provides:
defmodule CrucibleKitchen.Adapters.Tinkex.TrainingClient do
  @behaviour CrucibleTrain.Ports.TrainingClient
  # Implementation using Tinkex SDK
end
```

### 3. Workflow Composition

Workflows are composed of stages, not hardcoded:
```elixir
defmodule CrucibleKitchen.Workflows.SupervisedTraining do
  use CrucibleKitchen.Workflow

  workflow do
    stage :load_dataset, LoadDatasetStage
    stage :init_session, InitSessionStage

    loop :epochs, over: :num_epochs do
      loop :batches, over: :dataset do
        stage :render_batch, RenderBatchStage
        stage :forward_backward, ForwardBackwardStage
        stage :optim_step, OptimStepStage
        stage :log_metrics, LogMetricsStage
      end
      stage :maybe_checkpoint, CheckpointStage, when: :should_checkpoint
      stage :maybe_eval, EvalStage, when: :should_eval
    end

    stage :save_final, SaveFinalStage
  end
end
```

### 4. Telemetry is Structural

Every stage automatically emits telemetry:
```elixir
defmodule CrucibleKitchen.Stage do
  defmacro __using__(_opts) do
    quote do
      def run(context) do
        :telemetry.span(
          [:crucible_kitchen, :stage, stage_name()],
          %{context: context},
          fn ->
            result = execute(context)
            {result, %{}}
          end
        )
      end
    end
  end
end
```

### 5. Context Flows Through

A single context map flows through the workflow, accumulating state:
```elixir
%Context{
  config: %{model: "...", epochs: 3},
  adapters: %{training_client: TinkexAdapter},
  state: %{
    session: nil,
    dataset: nil,
    current_epoch: 0,
    current_step: 0,
    metrics: []
  }
}
```

---

## Decision Matrix

| Criterion | Hexagonal | Clean | Pipeline | Actor | Kitchen |
|-----------|-----------|-------|----------|-------|---------|
| Port/Adapter separation | ★★★ | ★★☆ | ☆☆☆ | ★☆☆ | ★★★ |
| Workflow composition | ☆☆☆ | ☆☆☆ | ★★★ | ★★☆ | ★★★ |
| Telemetry integration | ☆☆☆ | ☆☆☆ | ★★☆ | ★☆☆ | ★★★ |
| Elixir-native feel | ★☆☆ | ☆☆☆ | ★★☆ | ★★★ | ★★★ |
| ML training fit | ★☆☆ | ☆☆☆ | ★★★ | ★★☆ | ★★★ |
| Simplicity | ★★☆ | ☆☆☆ | ★★★ | ★☆☆ | ★★☆ |

**Decision:** Kitchen Architecture - a purpose-built hybrid optimized for ML training orchestration.

---

## Implementation Implications

1. **CrucibleKitchen.Workflow** module - DSL for workflow definition
2. **CrucibleKitchen.Stage** behaviour - Contract for stage implementations
3. **CrucibleTrain.Ports.*** and **CrucibleTelemetry.Ports.MetricsStore** - Shared port behaviours
4. **CrucibleKitchen.Context** - Flowing state container
5. **CrucibleKitchen.Runner** - Workflow executor with telemetry
6. **CrucibleKitchen.Telemetry** - Event definitions and handlers

The next document covers component design in detail.
