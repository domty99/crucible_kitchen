# Telemetry

CrucibleKitchen provides comprehensive telemetry for observability, using Erlang's `:telemetry` library. This guide covers the events emitted, built-in handlers, and how to write custom handlers.

## Overview

All CrucibleKitchen operations emit telemetry events, enabling:
- Real-time monitoring of training progress
- Logging to console or files
- Integration with metrics systems (Prometheus, StatsD)
- Custom dashboards and alerting

## Quick Start

Attach the console handler to see events as they happen:

```elixir
# Attach console handler
CrucibleKitchen.Telemetry.attach(:console)

# Run your workflow - events will be logged
{:ok, result} = CrucibleKitchen.run(MyWorkflow, config, adapters: adapters)
```

## Events

All events are prefixed with `[:crucible_kitchen, ...]`:

### Workflow Events

| Event | Description | Measurements | Metadata |
|-------|-------------|--------------|----------|
| `[:crucible_kitchen, :workflow, :run, :start]` | Workflow started | - | workflow, run_id |
| `[:crucible_kitchen, :workflow, :run, :stop]` | Workflow completed | duration | workflow, run_id |
| `[:crucible_kitchen, :workflow, :run, :exception]` | Workflow failed | duration | workflow, run_id, error |

### Stage Events

| Event | Description | Measurements | Metadata |
|-------|-------------|--------------|----------|
| `[:crucible_kitchen, :stage, :run, :start]` | Stage started | - | stage, module |
| `[:crucible_kitchen, :stage, :run, :stop]` | Stage completed | duration | stage, module, success |
| `[:crucible_kitchen, :stage, :run, :exception]` | Stage failed | duration | stage, module, error |

### Training Events

| Event | Description | Measurements | Metadata |
|-------|-------------|--------------|----------|
| `[:crucible_kitchen, :training, :step]` | Training step completed | loss, lr, grad_norm | step, total_steps |
| `[:crucible_kitchen, :training, :epoch]` | Epoch completed | epoch_loss | epoch, total_epochs |
| `[:crucible_kitchen, :training, :checkpoint]` | Checkpoint saved | size_bytes | checkpoint_name, path |
| `[:crucible_kitchen, :training, :eval]` | Evaluation completed | metrics | split, step |

### Dataset Events

| Event | Description | Measurements | Metadata |
|-------|-------------|--------------|----------|
| `[:crucible_kitchen, :dataset, :load]` | Dataset loaded | size, duration | name |
| `[:crucible_kitchen, :dataset, :batch]` | Batch processed | batch_size | batch_idx |

## Built-in Handlers

### Console Handler

Logs events to the console with formatting:

```elixir
CrucibleKitchen.Telemetry.attach(:console)
```

Output example:
```
[Kitchen] Workflow started: MyWorkflow
[Kitchen] Stage started: load_dataset
[Kitchen] Stage load_dataset completed in 1.2s
[Kitchen] Step 1/1000 | loss=2.5432 | lr=0.0002
[Kitchen] Step 2/1000 | loss=2.4123 | lr=0.0002
...
[Kitchen] Epoch 1/3 complete
[Kitchen] Checkpoint saved: epoch_1
[Kitchen] Workflow completed in 45.3m
```

### JSONL Handler

Writes events as newline-delimited JSON to a file:

```elixir
CrucibleKitchen.Telemetry.attach(:jsonl, path: "/var/log/training.jsonl")
```

Output format:
```json
{"timestamp":"2024-01-15T10:30:00Z","event":"crucible_kitchen.workflow.run.start","measurements":{},"metadata":{"workflow":"MyWorkflow","run_id":"run_123"}}
{"timestamp":"2024-01-15T10:30:01Z","event":"crucible_kitchen.training.step","measurements":{"loss":2.5432,"lr":0.0002},"metadata":{"step":1,"total_steps":1000}}
```

## Emitting Events from Stages

Use the `CrucibleKitchen.Telemetry` module to emit events from your stages:

```elixir
defmodule MyTrainStage do
  use CrucibleKitchen.Stage
  alias CrucibleKitchen.Telemetry

  @impl true
  def name, do: :train

  @impl true
  def execute(context) do
    # ... do training work ...

    # Emit step event
    Telemetry.emit_step(
      %{loss: result.loss, lr: config.learning_rate},
      %{step: context.state.global_step, total_steps: config.total_steps}
    )

    # Emit epoch event at end of epoch
    if end_of_epoch? do
      Telemetry.emit_epoch(
        %{epoch_loss: average_loss},
        %{epoch: current_epoch, total_epochs: config.epochs}
      )
    end

    {:ok, context}
  end
end
```

## Custom Handlers

Write custom handlers to integrate with your infrastructure:

```elixir
defmodule MyApp.Telemetry.PrometheusHandler do
  @events [
    [:crucible_kitchen, :training, :step],
    [:crucible_kitchen, :training, :epoch]
  ]

  def attach do
    CrucibleKitchen.Telemetry.attach_handler(
      "my-prometheus-handler",
      @events,
      &handle/4,
      nil
    )
  end

  def handle([:crucible_kitchen, :training, :step], measurements, metadata, _config) do
    # Push to Prometheus
    :prometheus_gauge.set(:training_loss, measurements.loss)
    :prometheus_counter.inc(:training_steps, 1)
    :prometheus_gauge.set(:learning_rate, measurements.lr)
  end

  def handle([:crucible_kitchen, :training, :epoch], _measurements, metadata, _config) do
    :prometheus_counter.inc(:training_epochs, 1)
  end

  def handle(_event, _measurements, _metadata, _config), do: :ok
end
```

### Slack/Discord Notifications

```elixir
defmodule MyApp.Telemetry.NotificationHandler do
  @events [
    [:crucible_kitchen, :workflow, :run, :stop],
    [:crucible_kitchen, :workflow, :run, :exception],
    [:crucible_kitchen, :training, :checkpoint]
  ]

  def attach(webhook_url) do
    CrucibleKitchen.Telemetry.attach_handler(
      "notifications",
      @events,
      &handle/4,
      %{webhook_url: webhook_url}
    )
  end

  def handle([:crucible_kitchen, :workflow, :run, :stop], measurements, metadata, config) do
    duration = format_duration(measurements.duration)
    send_notification(config.webhook_url, "Training completed in #{duration}")
  end

  def handle([:crucible_kitchen, :workflow, :run, :exception], _m, metadata, config) do
    send_notification(config.webhook_url, "Training FAILED: #{inspect(metadata.error)}")
  end

  def handle([:crucible_kitchen, :training, :checkpoint], _m, metadata, config) do
    send_notification(config.webhook_url, "Checkpoint saved: #{metadata.checkpoint_name}")
  end

  defp send_notification(url, message) do
    # HTTP POST to webhook
    HTTPoison.post(url, Jason.encode!(%{text: message}), [{"Content-Type", "application/json"}])
  end

  defp format_duration(ns) do
    minutes = ns / 60_000_000_000
    "#{Float.round(minutes, 1)} minutes"
  end
end
```

## Filtering Events

Filter events before processing:

```elixir
defmodule MyApp.Telemetry.FilteredHandler do
  def attach(opts) do
    filter = Keyword.get(opts, :filter, fn _, _, _ -> true end)

    CrucibleKitchen.Telemetry.attach_handler(
      "filtered-handler",
      CrucibleKitchen.Telemetry.events(),
      fn event, measurements, metadata, config ->
        if filter.(event, measurements, metadata) do
          handle(event, measurements, metadata, config)
        end
      end,
      nil
    )
  end

  defp handle(event, measurements, metadata, _config) do
    # Your handling logic
  end
end

# Usage - only log loss events above threshold
MyApp.Telemetry.FilteredHandler.attach(
  filter: fn event, measurements, _metadata ->
    case event do
      [:crucible_kitchen, :training, :step] ->
        measurements.loss > 1.0  # Only log high loss
      _ ->
        true
    end
  end
)
```

## Detaching Handlers

Remove handlers when no longer needed:

```elixir
# Detach by handler ID
CrucibleKitchen.Telemetry.detach("crucible-kitchen-console")
CrucibleKitchen.Telemetry.detach("my-prometheus-handler")
```

## Integration with External Systems

### W&B / MLflow

```elixir
defmodule MyApp.Telemetry.WandBHandler do
  @events [
    [:crucible_kitchen, :training, :step],
    [:crucible_kitchen, :training, :epoch]
  ]

  def attach(run_id) do
    CrucibleKitchen.Telemetry.attach_handler(
      "wandb",
      @events,
      &handle/4,
      %{run_id: run_id}
    )
  end

  def handle([:crucible_kitchen, :training, :step], measurements, metadata, config) do
    WandB.log(config.run_id, %{
      step: metadata.step,
      loss: measurements.loss,
      learning_rate: measurements.lr
    })
  end

  def handle([:crucible_kitchen, :training, :epoch], measurements, metadata, config) do
    WandB.log(config.run_id, %{
      epoch: metadata.epoch,
      epoch_loss: measurements.epoch_loss
    })
  end
end
```

## Best Practices

1. **Attach handlers before starting workflows** - Events emitted before attachment are lost
2. **Use unique handler IDs** - Prevents conflicts when attaching multiple handlers
3. **Handle errors gracefully** - Handler failures shouldn't crash your training
4. **Filter high-frequency events** - Step events can be very frequent; consider sampling
5. **Use async logging for I/O** - Don't block training on file writes or network calls

## Next Steps

- See [Custom Workflows](custom_workflows.md) for building workflows that emit telemetry
- See [Adapters Guide](adapters.md) for backend implementations
- Check `CrucibleKitchen.Telemetry` module docs for the full API
