# Custom Workflows

This guide covers how to create custom training workflows using the CrucibleKitchen DSL.

## Overview

Workflows define **how** training is orchestrated. They compose stages with control flow constructs like loops, conditionals, and parallel blocks. The workflow DSL compiles into an intermediate representation that the runner executes.

## Basic Structure

Every workflow module uses `CrucibleKitchen.Workflow` and defines a `workflow` block:

```elixir
defmodule MyApp.Workflows.CustomTraining do
  use CrucibleKitchen.Workflow

  workflow do
    stage :load_data, LoadDataStage
    stage :init_session, InitSessionStage
    stage :train, TrainStage
    stage :save, SaveStage
  end
end
```

## DSL Reference

### stage(name, module, opts \\ [])

Define a stage to execute. Stages are the atomic units of work.

```elixir
workflow do
  stage :load_dataset, LoadDatasetStage
  stage :init, InitSessionStage, timeout: 30_000
  stage :train, TrainEpochStage, retry: 3
end
```

- `name` - Stage identifier (atom), used in telemetry and logging
- `module` - Module implementing `CrucibleKitchen.Stage` behaviour
- `opts` - Options passed to the stage (available via `context.stage_opts`)

### loop(name, opts, do: block)

Iterate over a collection. The `:over` option names a function in your module that returns an enumerable.

```elixir
defmodule EpochLoopWorkflow do
  use CrucibleKitchen.Workflow

  workflow do
    stage :init, InitStage

    loop :epochs, over: :epochs_range do
      stage :train_epoch, TrainEpochStage

      loop :batches, over: :batch_iterator do
        stage :forward_backward, ForwardBackwardStage
        stage :optim_step, OptimStepStage
      end
    end

    stage :save, SaveStage
  end

  # These functions receive context and return enumerables
  def epochs_range(ctx), do: 0..(ctx.config.epochs - 1)
  def batch_iterator(ctx), do: ctx.state.dataset
end
```

The current loop item is stored in state as `:"#{name}_current"`:

```elixir
# Inside a stage within the :epochs loop
current_epoch = get_state(context, :epochs_current)
```

### conditional(predicate, do: block)

Execute block only if predicate returns true. The predicate is a function name in your module.

```elixir
defmodule ConditionalWorkflow do
  use CrucibleKitchen.Workflow

  workflow do
    loop :steps, over: :steps_range do
      stage :train_step, TrainStepStage

      conditional :should_eval? do
        stage :eval, EvalStage
      end

      conditional :should_checkpoint? do
        stage :checkpoint, CheckpointStage
      end
    end
  end

  def steps_range(ctx), do: 1..ctx.config.total_steps

  def should_eval?(ctx) do
    step = ctx.state.global_step
    rem(step, ctx.config.eval_every) == 0
  end

  def should_checkpoint?(ctx) do
    step = ctx.state.global_step
    rem(step, ctx.config.checkpoint_every) == 0
  end
end
```

### parallel(opts \\ [], do: block)

Execute stages in parallel. Useful for independent operations like running multiple evaluations.

```elixir
workflow do
  stage :train, TrainStage

  parallel max_concurrency: 4 do
    stage :eval_train, EvalTrainSetStage
    stage :eval_val, EvalValSetStage
    stage :eval_test, EvalTestSetStage
  end

  stage :save, SaveStage
end
```

Options:
- `:max_concurrency` - Maximum concurrent stages (default: `System.schedulers_online()`)

### stream_loop(name, opts, do: block)

Memory-efficient streaming loop. Unlike `loop` which materializes the full collection, `stream_loop` lazily evaluates items.

```elixir
defmodule StreamingWorkflow do
  use CrucibleKitchen.Workflow

  workflow do
    stage :init, InitStage

    stream_loop :batches, over: :stream_batches, prefetch: 4 do
      stage :train_batch, TrainBatchStage
    end

    stage :save, SaveStage
  end

  def stream_batches(ctx) do
    # Returns a Stream - items fetched lazily
    ctx.state.dataset
    |> Stream.chunk_every(ctx.config.batch_size)
    |> Stream.with_index()
  end
end
```

Options:
- `:over` - Function name returning a Stream
- `:prefetch` - Number of items to prefetch (default: 2)

Use cases:
- Large datasets that don't fit in memory
- Infinite streams
- Network-backed data sources

### async_loop(name, opts, do: block)

Producer-consumer pattern for off-policy training. The producer runs asynchronously while the consumer processes items.

```elixir
defmodule OffPolicyRLWorkflow do
  use CrucibleKitchen.Workflow

  workflow do
    stage :init, InitStage

    async_loop :off_policy,
      producer: :collect_rollouts,
      stop_when: :done_training?,
      buffer_size: 4 do
      stage :train_batch, PPOUpdateStage
    end

    stage :save, SaveStage
  end

  # Producer function - called repeatedly to produce items
  def collect_rollouts(ctx) do
    {:ok, trajectories} = TrainingClient.do_rollout(ctx, ctx.state.session, ctx.state.prompts)
    trajectories
  end

  # Stop predicate - checked after each consumer iteration
  def done_training?(ctx) do
    ctx.state.global_step >= ctx.config.total_steps
  end
end
```

Options:
- `:producer` - Function name that produces items (called repeatedly)
- `:stop_when` - Function name returning boolean to stop the loop
- `:buffer_size` - Max items to buffer before blocking producer (default: 4)

## Implementing Stages

Stages implement the `CrucibleKitchen.Stage` behaviour:

```elixir
defmodule TrainEpochStage do
  use CrucibleKitchen.Stage

  @impl true
  def name, do: :train_epoch

  @impl true
  def execute(context) do
    ports = get_train_ports(context)
    session = get_state(context, :session)
    epoch = get_state(context, :epochs_current)
    dataset = get_state(context, :dataset)

    context =
      dataset
      |> Enum.reduce(context, fn batch, ctx ->
        future = TrainingClient.forward_backward(ports, session, batch)
        {:ok, result} = TrainingClient.await(ports, future)

        optim_future = TrainingClient.optim_step(ports, session, ctx.config.learning_rate)
        {:ok, _} = TrainingClient.await(ports, optim_future)

        ctx
        |> record_metric(:loss, result.loss, step: ctx.state.global_step)
        |> put_state(:global_step, ctx.state.global_step + 1)
      end)

    {:ok, context}
  end

  # Optional: validate preconditions
  @impl true
  def validate(context) do
    if get_state(context, :session) do
      :ok
    else
      {:error, :session_not_initialized}
    end
  end

  # Optional: cleanup on failure
  @impl true
  def rollback(context, error) do
    Logger.error("TrainEpoch failed: #{inspect(error)}")
    context
  end
end
```

## Running Custom Workflows

```elixir
# Create context with adapters
adapters = %{
  training_client: MyApp.Adapters.TrainingClient,
  dataset_store: MyApp.Adapters.DatasetStore
}

config = %{
  model: "meta-llama/Llama-3.1-8B",
  epochs: 3,
  batch_size: 32
}

context = CrucibleKitchen.Context.new(config, adapters)

# Run the workflow
{:ok, result} = CrucibleKitchen.Workflow.Runner.run(CustomTrainingWorkflow, context)
```

## Composing Workflows

You can build complex workflows by composing simpler patterns:

```elixir
defmodule FullTrainingWorkflow do
  use CrucibleKitchen.Workflow

  workflow do
    # Setup phase
    stage :load_dataset, LoadDatasetStage
    stage :init_session, InitSessionStage

    # Main training loop
    loop :epochs, over: :epochs_range do
      stream_loop :batches, over: :stream_batches, prefetch: 4 do
        stage :forward_backward, ForwardBackwardStage
        stage :optim_step, OptimStepStage
      end

      # Periodic evaluation
      conditional :should_eval? do
        parallel max_concurrency: 2 do
          stage :eval_train, EvalTrainStage
          stage :eval_val, EvalValStage
        end
      end

      # Periodic checkpointing
      conditional :should_checkpoint? do
        stage :checkpoint, CheckpointStage
      end
    end

    # Cleanup phase
    stage :save_final, SaveFinalStage
    stage :cleanup, CleanupStage
  end

  def epochs_range(ctx), do: 0..(ctx.config.epochs - 1)
  def stream_batches(ctx), do: Stream.chunk_every(ctx.state.dataset, ctx.config.batch_size)
  def should_eval?(ctx), do: rem(ctx.state.epochs_current, ctx.config.eval_every_epochs) == 0
  def should_checkpoint?(ctx), do: rem(ctx.state.epochs_current + 1, ctx.config.checkpoint_every) == 0
end
```

## Next Steps

- See [Telemetry Guide](telemetry.md) for monitoring workflow execution
- See [Adapters Guide](adapters.md) for implementing backend integrations
- Check the built-in workflows in `CrucibleKitchen.Workflows` for reference implementations
