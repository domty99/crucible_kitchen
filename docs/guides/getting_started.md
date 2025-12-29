# Getting Started with CrucibleKitchen

CrucibleKitchen is the unified API surface for ML training orchestration in the North-Shore-AI ecosystem. It provides a clean, high-level interface for fine-tuning language models without needing to interact directly with underlying dependencies.

## Installation

Add `crucible_kitchen` to your dependencies in `mix.exs`:

```elixir
def deps do
  [
    {:crucible_kitchen, path: "../crucible_kitchen"}
  ]
end
```

## Core Concepts

### 1. Ports and Adapters

CrucibleKitchen uses the ports and adapters pattern. **Ports** define what operations are available (e.g., training, dataset loading). **Adapters** implement those operations for specific backends.

```
Port (Behaviour)                    Adapter (Implementation)
├─ TrainingClient                   ├─ CrucibleKitchen.Adapters.Tinkex.TrainingClient
├─ DatasetStore                     ├─ CrucibleKitchen.Adapters.HfDatasets.DatasetStore
├─ BlobStore                        ├─ CrucibleKitchen.Adapters.Noop.BlobStore
├─ HubClient                         ├─ CrucibleKitchen.Adapters.Noop.HubClient
├─ MetricsStore                      ├─ CrucibleTelemetry.Adapters.JSONLMetrics
└─ Completer                         └─ CrucibleKitchen.Adapters.Tinkex.Completer
```

### 2. Context

Context is the state that flows through a training run. It contains:
- `config` - User-provided configuration
- `adapters` - The implementations for each port
- `state` - Mutable state accumulated during execution
- `metrics` - Collected metrics for observability
- `metadata` - Run metadata (run_id, started_at, etc.)

### 3. Workflows

Workflows define the sequence of stages and control flow for training. They're built using a DSL:

```elixir
defmodule MyWorkflow do
  use CrucibleKitchen.Workflow

  workflow do
    stage :load_data, LoadDataStage
    stage :init, InitSessionStage

    loop :epochs, over: :epochs_range do
      stage :train, TrainEpochStage

      conditional :should_eval? do
        stage :eval, EvalStage
      end
    end

    stage :save, SaveModelStage
  end

  def epochs_range(ctx), do: 0..(ctx.config.epochs - 1)
  def should_eval?(ctx), do: rem(ctx.state.step, 100) == 0
end
```

### 4. Recipes

Recipes are pre-built workflow configurations for common training patterns:

- `SupervisedFineTuning` - Instruction tuning
- `RewardModeling` - RLHF reward model training
- `DPO` - Direct Preference Optimization

## Quick Start

### 1. Define Your Adapters

Create adapters for the ports you need. For development, use the noop adapters:

```elixir
adapters = %{
  training_client: CrucibleKitchen.Adapters.Noop.TrainingClient,
  dataset_store: CrucibleKitchen.Adapters.Noop.DatasetStore,
  metrics_store: CrucibleKitchen.Adapters.Noop.MetricsStore
}
```

For production, use adapters from crucible_kitchen or your app:

```elixir
adapters = %{
  training_client: {CrucibleKitchen.Adapters.Tinkex.TrainingClient, api_key: "..."},
  dataset_store: {CrucibleKitchen.Adapters.HfDatasets.DatasetStore, cache_dir: "/data"},
  metrics_store: {CrucibleTelemetry.Adapters.JSONLMetrics, path: "/var/log/metrics.jsonl"}
}
```

### 2. Configure Your Training Run

```elixir
config = %{
  model: "meta-llama/Llama-2-7b",
  dataset: "my_instructions",
  epochs: 3,
  batch_size: 8,
  learning_rate: 2.0e-5,
  lora_rank: 16
}
```

### 3. Run Training

Using a workflow module:

```elixir
{:ok, result} = CrucibleKitchen.run(MyWorkflow, config, adapters: adapters)
```

Using a recipe:

```elixir
alias CrucibleKitchen.Recipes.SupervisedFineTuning

{:ok, result} = CrucibleKitchen.run(SupervisedFineTuning, config, adapters: adapters)
```

## Using Ports in Stages

Stages interact with backends through the port facade functions:

```elixir
defmodule MyApp.Stages.TrainEpoch do
  @behaviour CrucibleKitchen.Stage

  alias CrucibleKitchen.Context
  alias CrucibleTrain.Ports.{TrainingClient, DatasetStore}
  alias CrucibleTrain.Types.Datum
  alias CrucibleTelemetry.Ports.MetricsStore

  @impl true
  def execute(context) do
    ports = Context.get_train_ports(context)
    session = Context.get_state(context, :session)
    dataset = Context.get_state(context, :dataset)
    metrics = Context.get_adapter(context, :metrics_store)
    run_id = Context.get_metadata(context, :run_id)

    # Stream batches from dataset
    {:ok, samples} = DatasetStore.to_list(ports, dataset)

    context =
      samples
      |> Stream.chunk_every(context.config.batch_size)
      |> Enum.reduce(context, fn batch, ctx ->
        # Convert to datums
        datums = Enum.map(batch, &to_datum/1)

        # Forward-backward pass
        future = TrainingClient.forward_backward(ports, session, datums)
        {:ok, result} = TrainingClient.await(ports, future)

        # Optimizer step
        optim_future = TrainingClient.optim_step(ports, session, ctx.config.learning_rate)
        {:ok, _} = TrainingClient.await(ports, optim_future)

        # Record metrics
        :ok = MetricsStore.record(metrics, run_id, :loss, result.loss, step: ctx.state.global_step)

        Context.put_state(ctx, :global_step, ctx.state.global_step + 1)
      end)

    {:ok, context}
  end

  defp to_datum(example) do
    # Convert raw example to Datum struct
    %Datum{
      model_input: %{chunks: [%{tokens: example.tokens}]},
      loss_fn_inputs: %{labels: example.labels}
    }
  end
end
```

## Type Safety

CrucibleKitchen re-exports types from the underlying training library:

```elixir
alias CrucibleKitchen.Types.{Datum, ModelInput, EncodedTextChunk, TensorData}

# Create a training datum
datum = %Datum{
  model_input: %ModelInput{
    chunks: [%EncodedTextChunk{tokens: [1, 2, 3, 4]}]
  },
  loss_fn_inputs: %{
    labels: %TensorData{data: <<...>>, shape: [4], dtype: :s32}
  }
}
```

## Metrics and Observability

Context automatically tracks metrics:

```elixir
# In a stage
context = Context.record_metric(context, :loss, 0.5, step: 100)
context = Context.record_metric(context, :accuracy, 0.85, step: 100, metadata: %{split: "eval"})

# Get metrics at the end
metrics = context.metrics
```

Use the MetricsStore port for persistent storage:

```elixir
MetricsStore.record(context, :loss, 0.5, step: 100)
{:ok, history} = MetricsStore.get_history(context, :loss)
```

## Next Steps

- Read the [Adapters Guide](adapters.md) to learn how to implement custom adapters
- Explore the built-in recipes in `CrucibleKitchen.Recipes`
- Check out the example workflows in the `examples/` directory
