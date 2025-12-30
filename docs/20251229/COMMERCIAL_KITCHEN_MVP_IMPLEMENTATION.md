# Commercial Kitchen MVP: Complete Implementation Guide

**Date:** 2025-12-29
**Type:** Comprehensive Implementation Prompt
**Goal:** Complete integration of crucible_kitchen with telemetry, model registry, and evaluation

---

## Table of Contents

1. [Mission](#1-mission)
2. [Required Reading](#2-required-reading)
3. [Current State Analysis](#3-current-state-analysis)
4. [Implementation Tasks](#4-implementation-tasks)
5. [TDD Approach](#5-tdd-approach)
6. [Quality Gates](#6-quality-gates)
7. [File-by-File Implementation](#7-file-by-file-implementation)
8. [Integration Tests](#8-integration-tests)
9. [Verification Checklist](#9-verification-checklist)

---

## 1. Mission

Create a complete integration that allows:

```bash
mix kitchen.run :supervised \
  --model meta-llama/Llama-3.1-8B \
  --dataset HuggingFaceH4/no_robots \
  --epochs 1
```

**Outcomes:**
1. Training executes via tinkex → Tinker API
2. All stages emit telemetry events
3. Trained model registered in crucible_model_registry
4. Evaluation metrics computed via eval_ex
5. All tests pass, no warnings, no dialyzer issues, no credo issues

---

## 2. Required Reading

### 2.1 Architecture Documents (READ FIRST)

```
tinkerer/brainstorm/20251229/commercial_kitchen_mvp/
├── README.md                      # Vision overview
└── docs/
    ├── 00_EXECUTIVE_SUMMARY.md    # Goals and success metrics
    ├── 01_INTEGRATION_MAP.md      # All 50+ libs and status
    ├── 02_MINIMUM_SLICE.md        # MVP scope definition
    ├── 03_DEMO_SCRIPT.md          # Expected demo flow
    ├── 04_IMPLEMENTATION_PLAN.md  # Task breakdown
    └── 05_SUCCESS_CRITERIA.md     # Definition of done
```

### 2.2 Prior Architecture Work

```
commercial_kitchen/docs/20251227/architecture_design/
├── 00_EXECUTIVE_SUMMARY.md        # Problem statement
├── 01_ARCHITECTURE_PATTERNS.md    # Hexagonal pattern decision
├── 02_COMPONENT_DESIGN.md         # Component specs
├── 03_WORKFLOW_ENGINE.md          # Workflow DSL design
├── 04_API_SURFACE.md              # Public API design
└── 05_IMPLEMENTATION_ROADMAP.md   # Original roadmap

crucible_kitchen/docs/20251228/thesis-driven-refactor/
├── 00-thesis.md                   # Core thesis
├── 01-tinker-cookbook-core-analysis.md
├── 02-chz-integration-analysis.md
├── 03-hf-libs-analysis.md
├── 04-crucible-ecosystem-analysis.md
├── 05-tinkex-cookbook-analysis.md
├── 06-implementation-plan.md
└── 07-implementation-summary.md   # What was completed
```

### 2.3 Source Files to Understand

**crucible_kitchen Core:**
```
crucible_kitchen/lib/crucible_kitchen.ex                    # Main API
crucible_kitchen/lib/crucible_kitchen/workflow.ex           # Workflow DSL
crucible_kitchen/lib/crucible_kitchen/workflow_runner.ex    # Execution engine
crucible_kitchen/lib/crucible_kitchen/stage.ex              # Stage behaviour
crucible_kitchen/lib/crucible_kitchen/context.ex            # State container
crucible_kitchen/lib/crucible_kitchen/recipe.ex             # Recipe behaviour
```

**Existing Workflows:**
```
crucible_kitchen/lib/crucible_kitchen/workflows/supervised.ex
crucible_kitchen/lib/crucible_kitchen/workflows/reinforcement.ex  # Stub
crucible_kitchen/lib/crucible_kitchen/workflows/preference.ex     # Stub
crucible_kitchen/lib/crucible_kitchen/workflows/distillation.ex   # Stub
```

**Existing Stages (15 implemented):**
```
crucible_kitchen/lib/crucible_kitchen/stages/
├── load_dataset.ex
├── init_session.ex
├── init_tokenizer.ex
├── build_supervised_dataset.ex
├── set_epoch.ex
├── get_batch.ex
├── forward_backward.ex
├── optim_step.ex
├── await_future.ex
├── log_step_metrics.ex
├── log_epoch_metrics.ex
├── save_checkpoint.ex
├── save_final_weights.ex
├── evaluate.ex                    # PLACEHOLDER - needs enhancement
└── cleanup.ex
```

**Existing Adapters:**
```
crucible_kitchen/lib/crucible_kitchen/adapters/
├── tinkex/training_client.ex      # Tinkex backend
├── hf_datasets/dataset_store.ex   # HuggingFace datasets
├── hf_hub/hub_client.ex           # HuggingFace Hub
└── noop/                          # Noop adapters for testing
    ├── training_client.ex
    ├── dataset_store.ex
    ├── hub_client.ex
    └── blob_store.ex
```

**Ports (in crucible_train):**
```
crucible_train/lib/crucible_train/ports/
├── training_client.ex
├── dataset_store.ex
├── blob_store.ex
├── hub_client.ex
└── metrics_store.ex
```

**Integration Libraries:**
```
crucible_telemetry/lib/crucible_telemetry.ex
crucible_telemetry/lib/crucible_telemetry/handler.ex
crucible_telemetry/lib/crucible_telemetry/events.ex

crucible_model_registry/lib/crucible_model_registry.ex
crucible_model_registry/lib/crucible_model_registry/models.ex
crucible_model_registry/lib/crucible_model_registry/artifacts.ex

eval_ex/lib/eval_ex.ex
eval_ex/lib/eval_ex/runner.ex
eval_ex/lib/eval_ex/metrics.ex
```

---

## 3. Current State Analysis

### 3.1 What EXISTS and WORKS

| Component | Status | Location |
|-----------|--------|----------|
| Workflow DSL | ✅ Working | `workflow.ex`, `workflow_runner.ex` |
| Stage behaviour | ✅ Working | `stage.ex` |
| Context management | ✅ Working | `context.ex` |
| Supervised workflow | ✅ Working | `workflows/supervised.ex` |
| 15 training stages | ✅ Working | `stages/*.ex` |
| Tinkex adapter | ✅ Working | `adapters/tinkex/training_client.ex` |
| HfDatasets adapter | ✅ Working | `adapters/hf_datasets/dataset_store.ex` |
| HfHub adapter | ✅ Working | `adapters/hf_hub/hub_client.ex` |
| Noop adapters | ✅ Working | `adapters/noop/*.ex` |
| Basic telemetry events | ⚠️ Partial | Events exist, handler not wired |

### 3.2 What NEEDS Implementation

| Component | Status | Priority |
|-----------|--------|----------|
| Telemetry handler wiring | ❌ Missing | P0 |
| ModelRegistry port | ❌ Missing | P0 |
| ModelRegistry adapter | ❌ Missing | P0 |
| RegisterModel stage | ❌ Missing | P0 |
| Evaluator port | ❌ Missing | P0 |
| Evaluator adapter | ❌ Missing | P0 |
| Enhanced Evaluate stage | ⚠️ Placeholder | P0 |
| Updated Supervised workflow | ⚠️ Needs stages | P0 |
| Mix task for CLI | ❌ Missing | P1 |
| Integration tests | ❌ Missing | P0 |

### 3.3 Test Status

Run these commands to verify current state:

```bash
cd crucible_kitchen
mix deps.get
mix compile --warnings-as-errors
mix test
mix credo --strict
mix dialyzer
```

**Expected current state:** Tests pass, but integration tests for new features don't exist yet.

---

## 4. Implementation Tasks

### Task 1: Telemetry Handler Integration

**Goal:** All crucible_kitchen events flow to crucible_telemetry handlers.

**Files to create/modify:**
- `lib/crucible_kitchen/telemetry.ex` (NEW)
- `lib/crucible_kitchen/application.ex` (MODIFY - attach handlers)

**Events to ensure are emitted:**
```elixir
[:crucible_kitchen, :workflow, :start]
[:crucible_kitchen, :workflow, :stop]
[:crucible_kitchen, :workflow, :exception]
[:crucible_kitchen, :stage, :start]
[:crucible_kitchen, :stage, :stop]
[:crucible_kitchen, :stage, :exception]
[:crucible_kitchen, :training, :step]
[:crucible_kitchen, :training, :epoch]
[:crucible_kitchen, :training, :checkpoint]
[:crucible_kitchen, :eval, :complete]
[:crucible_kitchen, :model, :registered]
```

---

### Task 2: Model Registry Port & Adapter

**Goal:** Define interface for model registration and implement crucible_model_registry adapter.

**Files to create:**
- `lib/crucible_kitchen/ports/model_registry.ex` (NEW)
- `lib/crucible_kitchen/adapters/model_registry.ex` (NEW)
- `lib/crucible_kitchen/adapters/noop/model_registry.ex` (NEW)

**Port interface:**
```elixir
defmodule CrucibleKitchen.Ports.ModelRegistry do
  @callback register(opts :: keyword(), artifact :: map()) ::
    {:ok, model :: map()} | {:error, term()}

  @callback get(opts :: keyword(), model_id :: String.t()) ::
    {:ok, model :: map()} | {:error, :not_found}

  @callback list(opts :: keyword()) ::
    {:ok, [model :: map()]}
end
```

---

### Task 3: RegisterModel Stage

**Goal:** Stage that registers trained model in crucible_model_registry.

**File to create:**
- `lib/crucible_kitchen/stages/register_model.ex` (NEW)

**Stage implementation:**
```elixir
defmodule CrucibleKitchen.Stages.RegisterModel do
  use CrucibleKitchen.Stage

  @impl true
  def name, do: :register_model

  @impl true
  def execute(context) do
    adapter = CrucibleKitchen.Context.get_adapter(context, :model_registry)

    artifact = %{
      name: context.config.model,
      version: generate_version(context),
      artifact_uri: context.state.final_checkpoint,
      metadata: %{
        training_config: Map.take(context.config, [:model, :dataset, :epochs, :learning_rate, :batch_size]),
        final_metrics: context.state[:epoch_metrics] || %{}
      },
      lineage: %{
        dataset: context.config.dataset,
        workflow: :supervised,
        stages: context.state[:executed_stages] || [],
        started_at: context.state[:started_at],
        completed_at: DateTime.utc_now()
      }
    }

    case adapter.register([], artifact) do
      {:ok, model} ->
        :telemetry.execute(
          [:crucible_kitchen, :model, :registered],
          %{},
          %{model_id: model.id, name: artifact.name}
        )
        {:ok, CrucibleKitchen.Context.put_state(context, :registered_model, model)}

      {:error, reason} ->
        {:error, {:registration_failed, reason}}
    end
  end

  defp generate_version(context) do
    timestamp = DateTime.utc_now() |> DateTime.to_iso8601(:basic)
    run_id = context.state[:run_id] || "unknown"
    "#{timestamp}-#{run_id}"
  end
end
```

---

### Task 4: Evaluator Port & Adapter

**Goal:** Define interface for model evaluation and implement eval_ex adapter.

**Files to create:**
- `lib/crucible_kitchen/ports/evaluator.ex` (NEW)
- `lib/crucible_kitchen/adapters/evaluator.ex` (NEW)
- `lib/crucible_kitchen/adapters/noop/evaluator.ex` (NEW)

**Port interface:**
```elixir
defmodule CrucibleKitchen.Ports.Evaluator do
  @callback evaluate(opts :: keyword(), model :: term(), dataset :: term()) ::
    {:ok, results :: map()} | {:error, term()}

  @callback generate_report(opts :: keyword(), results :: map()) ::
    {:ok, report :: String.t()} | {:error, term()}
end
```

---

### Task 5: Enhanced Evaluate Stage

**Goal:** Replace placeholder with real evaluation logic.

**File to modify:**
- `lib/crucible_kitchen/stages/evaluate.ex` (MODIFY)

**Enhanced implementation:**
```elixir
defmodule CrucibleKitchen.Stages.Evaluate do
  use CrucibleKitchen.Stage

  @impl true
  def name, do: :evaluate

  @impl true
  def execute(context) do
    evaluator = CrucibleKitchen.Context.get_adapter(context, :evaluator)
    dataset_store = CrucibleKitchen.Context.get_adapter(context, :dataset_store)

    # Load evaluation dataset (test split)
    with {:ok, eval_dataset} <- load_eval_dataset(dataset_store, context),
         {:ok, results} <- run_evaluation(evaluator, context, eval_dataset),
         {:ok, report} <- generate_report(evaluator, results) do

      :telemetry.execute(
        [:crucible_kitchen, :eval, :complete],
        Map.take(results, [:accuracy, :f1, :precision, :recall]),
        %{model: context.config.model, dataset: context.config.dataset}
      )

      context
      |> CrucibleKitchen.Context.put_state(:eval_results, results)
      |> CrucibleKitchen.Context.put_state(:eval_report, report)
      |> then(&{:ok, &1})
    end
  end

  defp load_eval_dataset(dataset_store, context) do
    dataset_store.load(
      [split: "test", limit: context.config[:eval_samples] || 500],
      context.config.dataset
    )
  end

  defp run_evaluation(evaluator, context, eval_dataset) do
    evaluator.evaluate(
      [metrics: [:accuracy, :f1, :precision, :recall]],
      context.state.trained_model,
      eval_dataset
    )
  end

  defp generate_report(evaluator, results) do
    evaluator.generate_report([format: :markdown], results)
  end
end
```

---

### Task 6: Update Supervised Workflow

**Goal:** Add evaluate and register_model stages to workflow.

**File to modify:**
- `lib/crucible_kitchen/workflows/supervised.ex` (MODIFY)

**Updated workflow:**
```elixir
defmodule CrucibleKitchen.Workflows.Supervised do
  use CrucibleKitchen.Workflow

  alias CrucibleKitchen.Stages.{
    LoadDataset,
    InitSession,
    InitTokenizer,
    BuildSupervisedDataset,
    SetEpoch,
    GetBatch,
    ForwardBackward,
    AwaitFuture,
    OptimStep,
    LogStepMetrics,
    LogEpochMetrics,
    SaveCheckpoint,
    SaveFinalWeights,
    Cleanup,
    Evaluate,
    RegisterModel
  }

  workflow do
    # Setup
    stage :load_dataset, LoadDataset
    stage :init_session, InitSession
    stage :init_tokenizer, InitTokenizer
    stage :build_supervised_dataset, BuildSupervisedDataset

    # Training loop
    loop :epochs, over: :epochs_range do
      stage :set_epoch, SetEpoch

      loop :batches, over: :batches_range do
        stage :get_batch, GetBatch
        stage :forward_backward, ForwardBackward
        stage :await_forward, AwaitFuture
        stage :optim_step, OptimStep
        stage :await_optim, AwaitFuture
        stage :log_step_metrics, LogStepMetrics
      end

      stage :log_epoch_metrics, LogEpochMetrics

      conditional :should_checkpoint? do
        stage :save_checkpoint, SaveCheckpoint
      end
    end

    # Finalization
    stage :save_final_weights, SaveFinalWeights
    stage :cleanup, Cleanup

    # Evaluation & Registration (NEW)
    stage :evaluate, Evaluate
    stage :register_model, RegisterModel
  end
end
```

---

### Task 7: Mix Task for CLI

**Goal:** Create `mix kitchen.run` task for executing workflows.

**File to create:**
- `lib/mix/tasks/kitchen.run.ex` (NEW) - in tinkex_cookbook

```elixir
defmodule Mix.Tasks.Kitchen.Run do
  use Mix.Task

  @shortdoc "Run a crucible_kitchen workflow"

  @moduledoc """
  Run a crucible_kitchen workflow.

  ## Usage

      mix kitchen.run :supervised --model MODEL --dataset DATASET [options]

  ## Options

    * `--model` - Model name (required)
    * `--dataset` - Dataset name (required)
    * `--epochs` - Number of epochs (default: 1)
    * `--batch-size` - Batch size (default: 32)
    * `--learning-rate` - Learning rate (default: 2.0e-4)
    * `--lora-rank` - LoRA rank (default: 32)

  ## Examples

      mix kitchen.run :supervised \\
        --model meta-llama/Llama-3.1-8B \\
        --dataset HuggingFaceH4/no_robots \\
        --epochs 1
  """

  def run(args) do
    Application.ensure_all_started(:tinkex_cookbook)

    {opts, [workflow], _} = OptionParser.parse(args,
      switches: [
        model: :string,
        dataset: :string,
        epochs: :integer,
        batch_size: :integer,
        learning_rate: :float,
        lora_rank: :integer
      ],
      aliases: [
        m: :model,
        d: :dataset,
        e: :epochs,
        b: :batch_size,
        l: :learning_rate,
        r: :lora_rank
      ]
    )

    workflow = String.to_existing_atom(workflow)
    config = build_config(opts)
    adapters = build_adapters()

    IO.puts("Starting #{workflow} workflow...")
    IO.puts("Model: #{config.model}")
    IO.puts("Dataset: #{config.dataset}")
    IO.puts("")

    case CrucibleKitchen.run(workflow, config, adapters: adapters) do
      {:ok, result} ->
        IO.puts("\nWorkflow complete!")
        IO.puts("Model ID: #{result.registered_model.id}")
        IO.puts("Accuracy: #{result.eval_results.accuracy}")
        IO.puts("F1 Score: #{result.eval_results.f1}")

      {:error, reason} ->
        IO.puts("\nWorkflow failed: #{inspect(reason)}")
        System.halt(1)
    end
  end

  defp build_config(opts) do
    %{
      model: opts[:model] || raise("--model is required"),
      dataset: opts[:dataset] || raise("--dataset is required"),
      epochs: opts[:epochs] || 1,
      batch_size: opts[:batch_size] || 32,
      learning_rate: opts[:learning_rate] || 2.0e-4,
      lora_rank: opts[:lora_rank] || 32
    }
  end

  defp build_adapters do
    %{
      training_client: CrucibleKitchen.Adapters.Tinkex.TrainingClient,
      dataset_store: CrucibleKitchen.Adapters.HfDatasets.DatasetStore,
      hub_client: CrucibleKitchen.Adapters.HfHub.HubClient,
      blob_store: CrucibleKitchen.Adapters.Noop.BlobStore,
      evaluator: CrucibleKitchen.Adapters.Evaluator,
      model_registry: CrucibleKitchen.Adapters.ModelRegistry
    }
  end
end
```

---

## 5. TDD Approach

### 5.1 Test Structure

For each new component, write tests BEFORE implementation:

```
test/crucible_kitchen/
├── ports/
│   ├── model_registry_test.exs      # Port contract tests
│   └── evaluator_test.exs
├── adapters/
│   ├── model_registry_test.exs      # Adapter unit tests
│   ├── evaluator_test.exs
│   └── noop/
│       ├── model_registry_test.exs
│       └── evaluator_test.exs
├── stages/
│   ├── register_model_test.exs      # Stage unit tests
│   └── evaluate_test.exs
├── telemetry_test.exs               # Telemetry emission tests
└── integration/
    ├── supervised_noop_test.exs     # Fast CI test with noop adapters
    └── supervised_full_test.exs     # Full integration (manual)
```

### 5.2 Test First Pattern

**Example: RegisterModel Stage**

```elixir
# test/crucible_kitchen/stages/register_model_test.exs
defmodule CrucibleKitchen.Stages.RegisterModelTest do
  use ExUnit.Case, async: true

  alias CrucibleKitchen.Stages.RegisterModel
  alias CrucibleKitchen.Context

  describe "execute/1" do
    test "registers model with correct artifact structure" do
      # Setup
      context = build_context_with_training_complete()

      # Execute
      {:ok, result_context} = RegisterModel.execute(context)

      # Assert
      assert result_context.state.registered_model != nil
      assert result_context.state.registered_model.id != nil
    end

    test "includes lineage in artifact" do
      context = build_context_with_training_complete()

      {:ok, result_context} = RegisterModel.execute(context)

      model = result_context.state.registered_model
      assert model.lineage.dataset == context.config.dataset
      assert model.lineage.workflow == :supervised
    end

    test "emits telemetry event on success" do
      :telemetry.attach(
        "test-handler",
        [:crucible_kitchen, :model, :registered],
        fn event, measurements, metadata, _ ->
          send(self(), {:telemetry, event, measurements, metadata})
        end,
        nil
      )

      context = build_context_with_training_complete()
      {:ok, _} = RegisterModel.execute(context)

      assert_receive {:telemetry, [:crucible_kitchen, :model, :registered], _, metadata}
      assert metadata.model_id != nil
    end

    test "returns error when registration fails" do
      context = build_context_with_failing_adapter()

      assert {:error, {:registration_failed, _}} = RegisterModel.execute(context)
    end
  end

  defp build_context_with_training_complete do
    Context.new(%{
      config: %{
        model: "test-model",
        dataset: "test-dataset",
        epochs: 1
      },
      state: %{
        final_checkpoint: "tinker://checkpoints/abc123",
        epoch_metrics: %{loss: 1.23, accuracy: 0.82},
        run_id: "test-run-123"
      },
      adapters: %{
        model_registry: CrucibleKitchen.Adapters.Noop.ModelRegistry
      }
    })
  end

  defp build_context_with_failing_adapter do
    Context.new(%{
      config: %{model: "test-model", dataset: "test-dataset"},
      state: %{final_checkpoint: "path"},
      adapters: %{
        model_registry: CrucibleKitchen.Adapters.Failing.ModelRegistry
      }
    })
  end
end
```

### 5.3 Run Tests Continuously

```bash
# Run all tests
mix test

# Run specific test file
mix test test/crucible_kitchen/stages/register_model_test.exs

# Run with coverage
mix test --cover

# Watch mode (if using mix_test_watch)
mix test.watch
```

---

## 6. Quality Gates

### 6.1 Compiler Warnings

```bash
mix compile --warnings-as-errors
```

**Must produce:** Zero warnings

### 6.2 Tests

```bash
mix test
```

**Must produce:** All tests passing

### 6.3 Credo

```bash
mix credo --strict
```

**Must produce:** No issues

### 6.4 Dialyzer

```bash
mix dialyzer
```

**Must produce:** No errors

### 6.5 Documentation

```bash
mix docs
```

**Must produce:** No warnings about missing @doc or @moduledoc

---

## 7. File-by-File Implementation

### 7.1 New Files to Create

```
crucible_kitchen/lib/crucible_kitchen/
├── telemetry.ex                           # Telemetry handler setup
├── ports/
│   ├── model_registry.ex                  # Model registry port
│   └── evaluator.ex                       # Evaluator port
├── adapters/
│   ├── model_registry.ex                  # Real model registry adapter
│   ├── evaluator.ex                       # Real evaluator adapter
│   └── noop/
│       ├── model_registry.ex              # Noop model registry
│       └── evaluator.ex                   # Noop evaluator
└── stages/
    └── register_model.ex                  # Register model stage

crucible_kitchen/test/crucible_kitchen/
├── telemetry_test.exs
├── ports/
│   ├── model_registry_test.exs
│   └── evaluator_test.exs
├── adapters/
│   ├── model_registry_test.exs
│   ├── evaluator_test.exs
│   └── noop/
│       ├── model_registry_test.exs
│       └── evaluator_test.exs
├── stages/
│   ├── register_model_test.exs
│   └── evaluate_test.exs
└── integration/
    └── supervised_noop_test.exs
```

### 7.2 Files to Modify

```
crucible_kitchen/lib/crucible_kitchen/
├── application.ex                         # Add telemetry handler attach
├── stages/evaluate.ex                     # Enhance with real evaluation
└── workflows/supervised.ex                # Add evaluate + register stages

crucible_kitchen/lib/crucible_kitchen/stages.ex  # Export new stages
```

---

## 8. Integration Tests

### 8.1 Noop Integration Test (CI-friendly)

```elixir
# test/crucible_kitchen/integration/supervised_noop_test.exs
defmodule CrucibleKitchen.Integration.SupervisedNoopTest do
  use ExUnit.Case, async: false

  @moduletag :integration

  setup do
    # Start telemetry collector
    collector = start_supervised!({TelemetryCollector, []})
    {:ok, collector: collector}
  end

  test "supervised workflow completes with noop adapters", %{collector: collector} do
    config = %{
      model: "test-model",
      dataset: "test-dataset",
      epochs: 1,
      batch_size: 2,
      learning_rate: 1.0e-4
    }

    adapters = %{
      training_client: CrucibleKitchen.Adapters.Noop.TrainingClient,
      dataset_store: CrucibleKitchen.Adapters.Noop.DatasetStore,
      hub_client: CrucibleKitchen.Adapters.Noop.HubClient,
      blob_store: CrucibleKitchen.Adapters.Noop.BlobStore,
      evaluator: CrucibleKitchen.Adapters.Noop.Evaluator,
      model_registry: CrucibleKitchen.Adapters.Noop.ModelRegistry
    }

    # Execute workflow
    {:ok, result} = CrucibleKitchen.run(:supervised, config, adapters: adapters)

    # Verify result structure
    assert result.registered_model != nil
    assert result.eval_results != nil
    assert result.eval_results.accuracy >= 0
    assert result.eval_results.f1 >= 0

    # Verify telemetry events
    events = TelemetryCollector.get_events(collector)

    assert has_event?(events, [:crucible_kitchen, :workflow, :start])
    assert has_event?(events, [:crucible_kitchen, :workflow, :stop])
    assert has_event?(events, [:crucible_kitchen, :stage, :start], stage: :load_dataset)
    assert has_event?(events, [:crucible_kitchen, :stage, :stop], stage: :load_dataset)
    assert has_event?(events, [:crucible_kitchen, :eval, :complete])
    assert has_event?(events, [:crucible_kitchen, :model, :registered])
  end

  test "workflow emits step metrics during training", %{collector: collector} do
    config = %{model: "test", dataset: "test", epochs: 1, batch_size: 2}
    adapters = noop_adapters()

    {:ok, _} = CrucibleKitchen.run(:supervised, config, adapters: adapters)

    events = TelemetryCollector.get_events(collector)
    step_events = Enum.filter(events, &match?({[:crucible_kitchen, :training, :step], _, _}, &1))

    assert length(step_events) > 0
  end

  defp noop_adapters do
    %{
      training_client: CrucibleKitchen.Adapters.Noop.TrainingClient,
      dataset_store: CrucibleKitchen.Adapters.Noop.DatasetStore,
      hub_client: CrucibleKitchen.Adapters.Noop.HubClient,
      blob_store: CrucibleKitchen.Adapters.Noop.BlobStore,
      evaluator: CrucibleKitchen.Adapters.Noop.Evaluator,
      model_registry: CrucibleKitchen.Adapters.Noop.ModelRegistry
    }
  end

  defp has_event?(events, event_name, metadata \\ []) do
    Enum.any?(events, fn {name, _measurements, meta} ->
      name == event_name and
        Enum.all?(metadata, fn {k, v} -> Map.get(meta, k) == v end)
    end)
  end
end
```

### 8.2 Telemetry Collector Helper

```elixir
# test/support/telemetry_collector.ex
defmodule TelemetryCollector do
  use GenServer

  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  def get_events(pid \\ __MODULE__) do
    GenServer.call(pid, :get_events)
  end

  @impl true
  def init(_opts) do
    events = [
      [:crucible_kitchen, :workflow, :start],
      [:crucible_kitchen, :workflow, :stop],
      [:crucible_kitchen, :stage, :start],
      [:crucible_kitchen, :stage, :stop],
      [:crucible_kitchen, :training, :step],
      [:crucible_kitchen, :training, :epoch],
      [:crucible_kitchen, :eval, :complete],
      [:crucible_kitchen, :model, :registered]
    ]

    :telemetry.attach_many(
      "telemetry-collector-#{inspect(self())}",
      events,
      &__MODULE__.handle_event/4,
      %{pid: self()}
    )

    {:ok, []}
  end

  @impl true
  def handle_call(:get_events, _from, events) do
    {:reply, Enum.reverse(events), events}
  end

  @impl true
  def handle_cast({:event, event}, events) do
    {:noreply, [event | events]}
  end

  def handle_event(event_name, measurements, metadata, %{pid: pid}) do
    GenServer.cast(pid, {:event, {event_name, measurements, metadata}})
  end
end
```

---

## 9. Verification Checklist

Run these commands in order to verify complete implementation:

### 9.1 Compile Check
```bash
cd crucible_kitchen
mix deps.get
mix compile --warnings-as-errors
```
**Expected:** Clean compile, zero warnings

### 9.2 Test Suite
```bash
mix test
```
**Expected:** All tests pass

### 9.3 Integration Test
```bash
mix test test/crucible_kitchen/integration/supervised_noop_test.exs
```
**Expected:** Integration test passes

### 9.4 Credo Check
```bash
mix credo --strict
```
**Expected:** No issues

### 9.5 Dialyzer Check
```bash
mix dialyzer
```
**Expected:** No errors (warnings acceptable if documented)

### 9.6 Documentation
```bash
mix docs
```
**Expected:** Docs generate without warnings

### 9.7 Manual Demo (with Tinker API key)
```bash
cd tinkex_cookbook
TINKER_API_KEY=your-key mix kitchen.run :supervised \
  --model meta-llama/Llama-3.1-8B \
  --dataset HuggingFaceH4/no_robots \
  --epochs 1
```
**Expected:** Training completes, model registered, metrics printed

---

## Summary

This document provides everything needed to implement the Commercial Kitchen MVP:

1. **Required reading** - All architecture and prior work
2. **Current state** - What exists vs what's needed
3. **Implementation tasks** - Specific files and code
4. **TDD approach** - Test-first methodology
5. **Quality gates** - Zero warnings/errors requirement
6. **Integration tests** - End-to-end verification

**Success criteria:**
- All quality gates pass
- Integration test passes
- Demo executes successfully
- Model registered with lineage
- Evaluation metrics computed

**The implementation is complete when all 9 verification checks pass.**
