# Metrics & Telemetry Evolution Plan

## Overview

This document outlines the implementation plan for evolving tinkex_cookbook's sl_basic recipe from a legacy file-based approach to a fully integrated CrucibleKitchen workflow with proper metrics storage, telemetry, and checkpointing.

## Current State Analysis

### sl_basic (Legacy)

The original recipe is a direct port from Python tinker-cookbook with several issues:

1. **Broken metrics logging** - `MlLog.log_metrics/3` is never called
2. **Direct file I/O** - Bypasses port/adapter abstraction
3. **No telemetry** - No events emitted for observability
4. **Tight coupling** - Hard-coded to Tinkex backend

```
sl_basic Flow (Current - Broken)
├── Config parsing (CLI args)
├── MlLog.setup_logging() → creates empty files
├── TinkexAdapter.start_session()
├── Training loop:
│   ├── forward_backward()
│   ├── optim_step()
│   └── Logger.info() → console only, no file write
├── save_checkpoint() → Tinkex backend only
└── save_final_weights()

Output: Empty metrics.jsonl, empty checkpoints.jsonl
```

### sl_basic_v2 (Partial Evolution)

Uses CrucibleKitchen.Recipe behavior with proper adapter injection but lacks storage adapters:

```elixir
# Current adapter configuration
def default_adapters(opts) do
  %{
    training_client: {TinkexAdapter, opts},
    dataset_store: {HfDatasetsAdapter, []}
    # MISSING: metrics_store, blob_store
  }
end
```

### CrucibleKitchen Capabilities

| Component | Status | Notes |
|-----------|--------|-------|
| Telemetry events | Complete | 12+ event types defined |
| Console handler | Complete | Colored output |
| JSONL handler | Complete | For telemetry events |
| MetricsStore port (crucible_telemetry) | Complete | Behavior defined |
| Noop MetricsStore (kitchen) | Complete | ETS-backed for testing |
| BlobStore port | Complete | Behavior defined |
| Noop BlobStore | Complete | Local file system |

---

## Implementation Plan

### Phase 1: JSONL MetricsStore Adapter

**File:** `crucible_telemetry/lib/crucible_telemetry/adapters/jsonl_metrics.ex`

**Purpose:** Persist training metrics (loss, learning rate, etc.) to JSONL file for analysis.

```elixir
defmodule CrucibleTelemetry.Adapters.JSONLMetrics do
  @moduledoc """
  JSONL file-based metrics storage adapter.

  Writes metrics as newline-delimited JSON for easy parsing
  and compatibility with analysis tools.

  ## Options

  - `:path` - Path to JSONL file (required)
  - `:buffer_size` - Number of records to buffer before flush (default: 1)

  ## Example

      adapters = %{
        metrics_store: {
          CrucibleTelemetry.Adapters.JSONLMetrics,
          [path: "/tmp/training/metrics.jsonl"]
        }
      }
  """

  @behaviour CrucibleTelemetry.Ports.MetricsStore

  @impl true
  def record(opts, run_id, metric_name, value, record_opts) do
    path = Keyword.fetch!(opts, :path)
    step = Keyword.get(record_opts, :step)
    metadata = Keyword.get(record_opts, :metadata, %{})

    entry = %{
      run_id: run_id,
      metric: to_string(metric_name),
      value: value,
      step: step,
      timestamp: DateTime.utc_now() |> DateTime.to_iso8601(),
      metadata: metadata
    }

    line = Jason.encode!(entry) <> "\n"
    File.write!(path, line, [:append, :utf8])
    :ok
  end

  @impl true
  def flush(_opts, _run_id), do: :ok

  @impl true
  def read(opts, run_id) do
    path = Keyword.fetch!(opts, :path)

    if File.exists?(path) do
      entries =
        path
        |> File.stream!()
        |> Stream.map(&Jason.decode!/1)
        |> Stream.filter(&(&1["run_id"] == run_id))
        |> Enum.to_list()

      {:ok, entries}
    else
      {:ok, []}
    end
  end

  @impl true
  def flush(_opts, _run_id), do: :ok

  defp maybe_filter_steps(stream, opts) do
    case {Keyword.get(opts, :from_step), Keyword.get(opts, :to_step)} do
      {nil, nil} -> stream
      {from, nil} -> Stream.filter(stream, &(&1["step"] >= from))
      {nil, to} -> Stream.filter(stream, &(&1["step"] <= to))
      {from, to} -> Stream.filter(stream, &(&1["step"] >= from and &1["step"] <= to))
    end
  end
end
```

**Estimated effort:** 2 hours

---

### Phase 2: JSONL Checkpoint Metadata Helper

**File:** `lib/crucible_kitchen/adapters/jsonl/checkpoint_store.ex`

**Purpose:** Track checkpoint metadata in JSONL while actual weights remain in backend (Tinkex).

**Note:** There is no CheckpointStore port in crucible_train. Treat this as a helper module or fold metadata into MetricsStore.

```elixir
defmodule CrucibleKitchen.Adapters.JSONL.CheckpointStore do
  @moduledoc """
  JSONL-based checkpoint metadata storage.

  Tracks checkpoint metadata (step, epoch, paths) while actual
  model weights are stored by the training backend.

  ## Options

  - `:path` - Path to checkpoints.jsonl file (required)
  """

  def save_metadata(opts, run_id, checkpoint_name, metadata) do
    path = Keyword.fetch!(opts, :path)

    entry = %{
      run_id: run_id,
      name: checkpoint_name,
      step: metadata[:step],
      epoch: metadata[:epoch],
      state_path: metadata[:state_path],
      sampler_path: metadata[:sampler_path],
      timestamp: DateTime.utc_now() |> DateTime.to_iso8601()
    }

    line = Jason.encode!(entry) <> "\n"
    File.write!(path, line, [:append, :utf8])
    {:ok, entry}
  end

  def list_checkpoints(opts, run_id) do
    path = Keyword.fetch!(opts, :path)

    if File.exists?(path) do
      checkpoints =
        path
        |> File.stream!()
        |> Stream.map(&Jason.decode!/1)
        |> Stream.filter(&(&1["run_id"] == run_id))
        |> Enum.to_list()

      {:ok, checkpoints}
    else
      {:ok, []}
    end
  end

  def get_latest(opts, run_id, required_key \\ nil) do
    case list_checkpoints(opts, run_id) do
      {:ok, []} -> {:ok, nil}
      {:ok, checkpoints} ->
        filtered =
          if required_key do
            Enum.filter(checkpoints, &(&1[required_key] != nil))
          else
            checkpoints
          end
        {:ok, List.last(filtered)}
    end
  end

  def get_by_name(opts, run_id, checkpoint_name) do
    case list_checkpoints(opts, run_id) do
      {:ok, checkpoints} ->
        checkpoint = Enum.find(checkpoints, &(&1["name"] == checkpoint_name))
        {:ok, checkpoint}
    end
  end
end
```

**Note:** No CheckpointStore port is planned; keep this as a helper module if needed.

**Estimated effort:** 3 hours (including port definition if needed)

---

### Phase 3: Fix ForwardBackward Stage

**File:** `lib/crucible_kitchen/stages/forward_backward.ex`

**Problem:** Loss value from forward-backward result is not extracted and stored in state.

**Current:**
```elixir
def execute(context) do
  # ... forward_backward call ...
  context = put_state(context, :fb_future, future)
  {:ok, context}
end
```

**Fixed:**
```elixir
def execute(context) do
  session = get_state(context, :session)
  batch = get_state(context, :current_batch)

  ports = get_train_ports(context)
  future = TrainingClient.forward_backward(ports, session, batch)
  context = put_state(context, :fb_future, future)
  {:ok, context}
end
```

**Additionally, fix AwaitFuture to extract loss:**
```elixir
# In AwaitFuture stage, after awaiting fb_future:
def execute(context) do
  key = get_stage_opt(context, :key)
  future = get_state(context, key)

  ports = get_train_ports(context)

  case TrainingClient.await(ports, future) do
    {:ok, result} ->
      context = put_state(context, :"#{key}_result", result)

      # Extract loss if this is fb_future
      context =
        if key == :fb_future and is_map(result) do
          loss = Map.get(result, :loss) || Map.get(result, "loss")
          if loss, do: put_state(context, :current_loss, loss), else: context
        else
          context
        end

      {:ok, context}
    {:error, reason} ->
      {:error, {:await_failed, key, reason}}
  end
end
```

**Estimated effort:** 1 hour

---

### Phase 4: Fix LogStepMetrics Stage

**File:** `lib/crucible_kitchen/stages/log_step_metrics.ex`

**Problem:** Stage logs to console but doesn't record to MetricsStore.

**Fixed implementation:**
```elixir
defmodule CrucibleKitchen.Stages.LogStepMetrics do
  use CrucibleKitchen.Stage

  alias CrucibleTelemetry.Ports.MetricsStore

  require Logger

  @impl true
  def name, do: :log_step_metrics

  @impl true
  def execute(context) do
    global_step = get_state(context, :global_step, 0)
    total_steps = get_state(context, :total_steps, 0)
    current_lr = get_state(context, :current_lr, 0.0)
    current_loss = get_state(context, :current_loss)
    log_every = get_config(context, :log_every, 10)

    # Console logging (throttled)
    if log_every > 0 and rem(global_step, log_every) == 0 do
      progress = if total_steps > 0, do: Float.round(global_step / total_steps * 100, 1), else: 0.0
      loss_str = if current_loss, do: " | loss=#{Float.round(current_loss, 6)}", else: ""
      Logger.info("[Step #{global_step}/#{total_steps}] #{progress}% | lr=#{format_lr(current_lr)}#{loss_str}")
    end

    # Always record metrics to storage
    context =
      context
      |> record_metric(:step, global_step, step: global_step)
      |> record_metric(:lr, current_lr, step: global_step)

    context =
      if current_loss do
        record_metric(context, :loss, current_loss, step: global_step)
      else
        context
      end

    # Emit telemetry event
    measurements = %{
      step: global_step,
      lr: current_lr
    }
    measurements = if current_loss, do: Map.put(measurements, :loss, current_loss), else: measurements

    :telemetry.execute(
      [:crucible_kitchen, :training, :step],
      measurements,
      %{
        total_steps: total_steps,
        run_id: get_metadata(context, :run_id)
      }
    )

    {:ok, context}
  end

  defp format_lr(lr) when is_float(lr), do: :io_lib.format("~.4e", [lr]) |> to_string()
  defp format_lr(lr), do: inspect(lr)
end
```

**Estimated effort:** 1 hour

---

### Phase 5: Update sl_basic_v2 Recipe

**File:** `lib/tinkex_cookbook/recipes/sl_basic_v2.ex`

**Changes:**

1. Add metrics_store adapter (optionally blob_store)
2. Attach telemetry handlers
3. Create output directory

```elixir
defmodule TinkexCookbook.Recipes.SlBasicV2 do
  use CrucibleKitchen.Recipe

  alias CrucibleKitchen.Adapters.Tinkex.TrainingClient, as: TinkexAdapter
  alias CrucibleKitchen.Adapters.HfDatasets.DatasetStore, as: HfDatasetsAdapter
  alias CrucibleTelemetry.Adapters.JSONLMetrics, as: JSONLMetrics

  require Logger

  @default_log_path "/tmp/tinkex-cookbook/sl_basic_v2"

  # ... existing callbacks ...

  @impl true
  def required_adapters do
    [:training_client, :dataset_store]
  end

  @impl true
  def optional_adapters do
    [:metrics_store, :blob_store, :hub_client]
  end

  @spec run(map(), keyword()) :: {:ok, map()} | {:error, term()}
  def run(config, opts \\ []) do
    log_path = Keyword.get(opts, :log_path, @default_log_path)

    # Ensure output directory exists
    File.mkdir_p!(log_path)

    # Attach telemetry handlers
    attach_telemetry_handlers(log_path)

    # Build adapters with storage
    adapters = build_adapters(opts, log_path)

    Logger.info("Starting sl_basic_v2 training")
    Logger.info("  Output: #{log_path}")

    result = CrucibleKitchen.run(__MODULE__, config, adapters: adapters)

    # Detach handlers after run
    detach_telemetry_handlers()

    result
  end

  defp build_adapters(opts, log_path) do
    api_key = Keyword.get(opts, :api_key)
    base_url = Keyword.get(opts, :base_url)

    training_opts =
      []
      |> maybe_add(:api_key, api_key)
      |> maybe_add(:base_url, base_url)

    %{
      training_client: {TinkexAdapter, training_opts},
      dataset_store: {HfDatasetsAdapter, []},
      metrics_store: {JSONLMetrics, [path: Path.join(log_path, "metrics.jsonl")]}
    }
  end

  defp attach_telemetry_handlers(log_path) do
    CrucibleKitchen.Telemetry.attach(:console, id: "sl_basic_v2_console")
    CrucibleKitchen.Telemetry.attach(:jsonl,
      id: "sl_basic_v2_jsonl",
      path: Path.join(log_path, "telemetry.jsonl")
    )
  end

  defp detach_telemetry_handlers do
    :telemetry.detach("sl_basic_v2_console")
    :telemetry.detach("sl_basic_v2_jsonl")
  end

  defp maybe_add(opts, _key, nil), do: opts
  defp maybe_add(opts, key, value), do: Keyword.put(opts, key, value)
end
```

**Estimated effort:** 2 hours

---

### Phase 6: Deprecate sl_basic

**File:** `lib/tinkex_cookbook/recipes/sl_basic.ex`

**Option A: Add deprecation warning**
```elixir
def run_training(config, opts \\ []) do
  Logger.warning("""
  TinkexCookbook.Recipes.SlBasic is deprecated.
  Use TinkexCookbook.Recipes.SlBasicV2 instead.
  """)

  # Continue with existing implementation for backwards compatibility
  do_run_training(config, opts)
end
```

**Option B: Delegate to v2**
```elixir
def run_training(config, opts \\ []) do
  Logger.warning("SlBasic is deprecated, delegating to SlBasicV2")
  SlBasicV2.run(config, opts)
end
```

**Estimated effort:** 30 minutes

---

## Output Structure (Evolved)

After implementation, running sl_basic_v2 produces:

```
/tmp/tinkex-cookbook/sl_basic_v2/
├── metrics.jsonl          # Per-step metrics
│   {"run_id":"abc123","metric":"loss","value":2.34,"step":0,"timestamp":"..."}
│   {"run_id":"abc123","metric":"lr","value":0.0002,"step":0,"timestamp":"..."}
│   {"run_id":"abc123","metric":"loss","value":2.12,"step":1,"timestamp":"..."}
│   ...
│
├── checkpoints.jsonl      # Checkpoint metadata
│   {"run_id":"abc123","name":"step_000020","step":20,"epoch":0,"state_path":"..."}
│   {"run_id":"abc123","name":"step_000040","step":40,"epoch":0,"state_path":"..."}
│   ...
│
├── telemetry.jsonl        # Workflow/stage events
│   {"event":"crucible_kitchen.workflow.run.start","timestamp":"...","metadata":{...}}
│   {"event":"crucible_kitchen.stage.run.stop","timestamp":"...","measurements":{"duration":123}}
│   ...
│
└── config.json            # (Optional) Run configuration snapshot
```

---

## Timeline Summary

| Phase | Component | Location | Effort |
|-------|-----------|----------|--------|
| 1 | JSONL MetricsStore | crucible_telemetry | 2h |
| 2 | JSONL checkpoint metadata helper | crucible_kitchen | 3h |
| 3 | Fix ForwardBackward/AwaitFuture | crucible_kitchen | 1h |
| 4 | Fix LogStepMetrics | crucible_kitchen | 1h |
| 5 | Update sl_basic_v2 | tinkex_cookbook | 2h |
| 6 | Deprecate sl_basic | tinkex_cookbook | 0.5h |
| **Total** | | | **9.5h** |

---

## Testing Strategy

### Unit Tests (per adapter)

```elixir
# test/crucible_telemetry/adapters/jsonl_metrics_test.exs
defmodule CrucibleTelemetry.Adapters.JSONLMetricsTest do
  use ExUnit.Case, async: false

  alias CrucibleTelemetry.Adapters.JSONLMetrics

  setup do
    path = Path.join(System.tmp_dir!(), "test_metrics_#{:rand.uniform(100_000)}.jsonl")
    on_exit(fn -> File.rm(path) end)
    {:ok, path: path}
  end

  test "record/5 appends to file", %{path: path} do
    opts = [path: path]

    assert :ok = JSONLMetrics.record(opts, "run1", :loss, 1.5, step: 0)
    assert :ok = JSONLMetrics.record(opts, "run1", :loss, 1.2, step: 1)

    lines = File.read!(path) |> String.split("\n", trim: true)
    assert length(lines) == 2
  end

  test "read/2 returns run metrics", %{path: path} do
    opts = [path: path]

    JSONLMetrics.record(opts, "run1", :loss, 1.5, step: 0)
    JSONLMetrics.record(opts, "run1", :loss, 1.2, step: 1)
    JSONLMetrics.record(opts, "run2", :loss, 2.0, step: 0)

    {:ok, entries} = JSONLMetrics.read(opts, "run1")
    assert length(entries) == 2
  end
end
```

### Integration Tests

```elixir
# test/tinkex_cookbook/recipes/sl_basic_v2_integration_test.exs
@tag :integration
test "full training run with metrics" do
  config = %{
    model: "meta-llama/Llama-3.1-8B",
    epochs: 1,
    batch_size: 2,
    n_train_samples: 4
  }

  log_path = Path.join(System.tmp_dir!(), "sl_basic_v2_test")

  {:ok, _result} = SlBasicV2.run(config, log_path: log_path)

  # Verify metrics file created
  metrics_path = Path.join(log_path, "metrics.jsonl")
  assert File.exists?(metrics_path)

  # Verify metrics content
  lines = File.read!(metrics_path) |> String.split("\n", trim: true)
  assert length(lines) >= 2  # At least step 0 and step 1
end
```

---

## Future Enhancements

### PostgreSQL Adapters (Future)
- MetricsStore backed by timeseries table
- Checkpoint metadata helper backed by checkpoints table
- Enable multi-run queries and analysis

### Prometheus Handler (Future)
- Real-time metrics export
- Integration with Grafana dashboards

### CrucibleTelemetry Integration (Future)
- Multi-run experiment tracking
- Statistical analysis across runs
- A/B test comparisons
