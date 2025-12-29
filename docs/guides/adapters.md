# Adapters Guide

Adapters implement CrucibleTrain and CrucibleTelemetry ports. They connect your training workflows to specific backends - whether that's a local development environment, cloud training services, or custom infrastructure. CrucibleKitchen keeps only the Completer port for inference.

## Available Ports

| Port | Purpose | Required Callbacks |
|------|---------|-------------------|
| `TrainingClient` | Training platform (forward/backward, optim) | `start_session`, `forward_backward`, `optim_step`, `await`, `save_checkpoint`, `load_checkpoint`, `close_session` |
| `DatasetStore` | Dataset loading and ops | `load_dataset`, `get_split`, `shuffle`, `take`, `skip`, `select`, `to_list` |
| `BlobStore` | Artifact storage (checkpoints, weights) | `read`, `stream`, `write`, `exists?` |
| `HubClient` | Model hub operations (HuggingFace) | `download`, `snapshot`, `list_files` |
| `MetricsStore` | Metrics persistence | `record`, `flush`, `read` |
| `Completer` | Text completion / sampling | `complete`, `complete_batch` (`stream_complete` optional) |

Tokenizer access is adapter-specific. TrainingClient adapters can expose helper functions, but there is no TokenizerClient port.

## Implementing an Adapter

### 1. Choose a Port

```elixir
defmodule MyApp.Adapters.TrainingClient do
  @behaviour CrucibleTrain.Ports.TrainingClient

  # ...
end
```

### 2. Implement Required Callbacks

Each port defines callbacks that adapters must implement. Callbacks receive:
- `opts` - Adapter-specific options (API keys, URLs, etc.)
- Additional arguments specific to the operation

```elixir
defmodule MyApp.Adapters.TrainingClient do
  @behaviour CrucibleTrain.Ports.TrainingClient

  @impl true
  def start_session(opts, config) do
    api_key = Keyword.fetch!(opts, :api_key)
    base_url = Keyword.get(opts, :base_url, "https://api.example.com")

    # Initialize connection to training platform
    case MyClient.connect(base_url, api_key) do
      {:ok, client} ->
        session = %{
          client: client,
          model: config[:model],
          started_at: DateTime.utc_now()
        }
        {:ok, session}

      {:error, reason} ->
        {:error, {:connection_failed, reason}}
    end
  end

  @impl true
  def forward_backward(_opts, session, datums) do
    # Submit batch for forward-backward pass
    batch_data = serialize_datums(datums)
    Task.async(fn -> MyClient.submit_batch(session.client, batch_data) end)
  end

  @impl true
  def optim_step(_opts, session, learning_rate) do
    Task.async(fn -> MyClient.apply_gradients(session.client, learning_rate) end)
  end

  @impl true
  def await(_opts, future) do
    case Task.await(future, 60_000) do
      {:ok, job_id} ->
        case MyClient.wait_for_job(job_id, timeout: 60_000) do
          {:ok, result} ->
            {:ok, %{loss: result["loss"], grad_norm: result["grad_norm"]}}

          {:error, :timeout} ->
            {:error, :timeout}

          {:error, reason} ->
            {:error, reason}
        end

      {:error, reason} ->
        {:error, reason}
    end
  end

  @impl true
  def save_checkpoint(_opts, session, path) do
    MyClient.save_checkpoint(session.client, path)
  end

  @impl true
  def load_checkpoint(_opts, session, path) do
    case MyClient.load_checkpoint(session.client, path) do
      :ok -> :ok
      {:error, reason} -> {:error, reason}
    end
  end

  @impl true
  def close_session(_opts, session) do
    MyClient.disconnect(session.client)
    :ok
  end

  defp serialize_datums(datums) do
    # Convert Datum structs to format expected by backend
    Enum.map(datums, fn datum ->
      %{
        "model_input" => serialize_model_input(datum.model_input),
        "loss_fn_inputs" => serialize_tensors(datum.loss_fn_inputs)
      }
    end)
  end
end
```

### 3. Register with Adapters Map

```elixir
adapters = %{
  training_client: {MyApp.Adapters.TrainingClient, api_key: "sk-..."},
  dataset_store: MyApp.Adapters.DatasetStore
}
```

## Adapter Options Pattern

Adapters receive options in two ways:

1. **At registration time** - Options passed when creating the adapters map
2. **At runtime** - These options are passed as the first argument to callbacks

```elixir
# At registration
adapters = %{
  training_client: {MyAdapter, api_key: "...", timeout: 30_000}
}

# In adapter callback
def start_session(opts, config) do
  api_key = Keyword.fetch!(opts, :api_key)      # "..."
  timeout = Keyword.get(opts, :timeout, 10_000)  # 30_000
  # ...
end
```

## Built-in Noop Adapters

CrucibleKitchen provides noop adapters for development and testing:

```elixir
alias CrucibleKitchen.Adapters.Noop

adapters = %{
  training_client: Noop.TrainingClient,
  dataset_store: Noop.DatasetStore,
  blob_store: Noop.BlobStore,
  hub_client: Noop.HubClient,
  metrics_store: Noop.MetricsStore,
  completer: Noop.Completer
}
```

Noop adapters:
- Return successful responses with synthetic data
- Use in-memory storage (ETS) where persistence is needed
- Simulate realistic latencies and metrics
- Are safe for concurrent use

Tokenizer helpers are adapter-specific; `Noop.TokenizerClient` exists as an optional extension, not a port.

## DatasetStore Adapter Example

```elixir
defmodule MyApp.Adapters.HFDatasets do
  @behaviour CrucibleTrain.Ports.DatasetStore

  @impl true
  def load_dataset(opts, dataset_id, load_opts) do
    cache_dir = Keyword.get(opts, :cache_dir, "/tmp/datasets")
    split = Keyword.get(load_opts, :split, "train")

    # Download or load from cache
    case HFDatasets.load(dataset_id, split: split, cache_dir: cache_dir) do
      {:ok, dataset} ->
        handle = %{
          id: dataset_id,
          split: split,
          dataset: dataset
        }
        {:ok, handle}

      {:error, reason} ->
        {:error, reason}
    end
  end

  @impl true
  def get_split(_opts, handle, split) do
    case HFDatasets.get_split(handle.dataset, split) do
      {:ok, dataset} -> {:ok, %{handle | split: split, dataset: dataset}}
      {:error, reason} -> {:error, reason}
    end
  end

  @impl true
  def shuffle(_opts, handle, shuffle_opts) do
    {:ok, %{handle | dataset: HFDatasets.shuffle(handle.dataset, shuffle_opts)}}
  end

  @impl true
  def take(_opts, handle, count) do
    {:ok, %{handle | dataset: HFDatasets.take(handle.dataset, count)}}
  end

  @impl true
  def skip(_opts, handle, count) do
    {:ok, %{handle | dataset: HFDatasets.skip(handle.dataset, count)}}
  end

  @impl true
  def select(_opts, handle, selection) do
    {:ok, %{handle | dataset: HFDatasets.select(handle.dataset, selection)}}
  end

  @impl true
  def to_list(_opts, handle) do
    {:ok, HFDatasets.to_list(handle.dataset)}
  end
end
```

## Testing Adapters

Use ExUnit to test adapter implementations:

```elixir
defmodule MyApp.Adapters.TrainingClientTest do
  use ExUnit.Case, async: true

  alias MyApp.Adapters.TrainingClient

  describe "start_session/2" do
    test "connects successfully with valid credentials" do
      opts = [api_key: "test-key"]
      config = %{model: "test-model"}

      assert {:ok, session} = TrainingClient.start_session(opts, config)
      assert session.model == "test-model"
    end

    test "returns error with invalid credentials" do
      opts = [api_key: "invalid"]
      config = %{model: "test-model"}

      assert {:error, _} = TrainingClient.start_session(opts, config)
    end
  end

  describe "forward_backward/3" do
    setup do
      {:ok, session} = TrainingClient.start_session([api_key: "test"], %{model: "m"})
      {:ok, session: session}
    end

    test "submits batch and returns future", %{session: session} do
      datums = [build_datum(), build_datum()]

      future = TrainingClient.forward_backward([], session, datums)
      assert is_map(future)
    end
  end

  defp build_datum do
    %CrucibleTrain.Types.Datum{
      model_input: %CrucibleTrain.Types.ModelInput{chunks: []},
      loss_fn_inputs: %{}
    }
  end
end
```

## Adapter Composition

Create adapters that wrap other adapters for cross-cutting concerns:

```elixir
defmodule MyApp.Adapters.RetryingTrainingClient do
  @behaviour CrucibleTrain.Ports.TrainingClient

  @impl true
  def forward_backward(opts, session, datums) do
    inner = Keyword.fetch!(opts, :inner_adapter)
    inner_opts = Keyword.get(opts, :inner_opts, [])
    inner.forward_backward(inner_opts, session, datums)
  end

  @impl true
  def await(opts, future) do
    inner = Keyword.fetch!(opts, :inner_adapter)
    inner_opts = Keyword.get(opts, :inner_opts, [])
    max_retries = Keyword.get(opts, :max_retries, 3)

    retry(max_retries, fn ->
      inner.await(inner_opts, future)
    end)
  end

  defp retry(0, _fun), do: {:error, :max_retries_exceeded}
  defp retry(n, fun) do
    case fun.() do
      {:ok, result} -> {:ok, result}
      {:error, :temporary_error} -> retry(n - 1, fun)
      {:error, reason} -> {:error, reason}
    end
  end

  # Delegate other callbacks...
end

# Usage
adapters = %{
  training_client: {MyApp.Adapters.RetryingTrainingClient,
    inner_adapter: MyApp.Adapters.TrainingClient,
    inner_opts: [api_key: "..."],
    max_retries: 5
  }
}
```

## Best Practices

1. **Handle errors gracefully** - Return `{:error, reason}` tuples, never raise
2. **Make adapters stateless** - State should live in session/handle structs
3. **Document options** - Use `@moduledoc` to list available options
4. **Implement optional callbacks** - They provide better user experience
5. **Test with noop adapters** - Develop stages using noops, swap for real adapters in production
6. **Use telemetry** - Emit events for observability
