defmodule CrucibleKitchen.Adapters.HfDatasets.DatasetStore do
  @moduledoc """
  DatasetStore adapter for HuggingFace Datasets.

  Implements CrucibleTrain.Ports.DatasetStore using the hf_datasets_ex library.
  Supports loading datasets from HuggingFace Hub with streaming and caching.

  ## Configuration

  Environment variables:
  - `HF_TOKEN` - HuggingFace API token for private datasets

  ## Adapter Options

  - `:token` - Override HuggingFace token
  - `:cache` - Enable/disable caching (default: true)

  ## Usage

      config :my_app,
        adapters: %{
          dataset_store: {CrucibleKitchen.Adapters.HfDatasets.DatasetStore, [
            token: System.get_env("HF_TOKEN")
          ]}
        }

  ## Dataset ID Format

  Supports multiple formats:
  - HuggingFace repo: "openai/gsm8k"
  - Named datasets: :mmlu, :humaneval, :gsm8k
  - Local paths: "/path/to/data.jsonl"
  """

  @behaviour CrucibleTrain.Ports.DatasetStore

  require Logger

  # ============================================================================
  # CrucibleTrain.Ports.DatasetStore Callbacks
  # ============================================================================

  @impl true
  @spec load_dataset(keyword(), String.t(), keyword()) :: {:ok, term()} | {:error, term()}
  def load_dataset(opts, repo_id, load_opts \\ []) do
    load(opts, repo_id, load_opts)
  end

  @impl true
  @spec get_split(keyword(), term(), String.t() | atom()) :: {:ok, term()} | {:error, term()}
  def get_split(_opts, dataset_handle, split) do
    case unwrap_dataset(dataset_handle) do
      %HfDatasetsEx.DatasetDict{} = dd ->
        split_key = if is_atom(split), do: Atom.to_string(split), else: split

        case Map.get(dd, split_key) do
          nil -> {:error, {:split_not_found, split}}
          split_ds -> {:ok, wrap_dataset(split_ds, dataset_handle.id, dataset_handle.load_opts)}
        end

      _ ->
        # Already a single split dataset
        {:ok, dataset_handle}
    end
  end

  @impl true
  @spec shuffle(keyword(), term(), keyword()) :: {:ok, term()} | {:error, term()}
  def shuffle(_opts, dataset_handle, shuffle_opts \\ []) do
    case unwrap_dataset(dataset_handle) do
      %HfDatasetsEx.Dataset{} = ds ->
        shuffled = HfDatasetsEx.Dataset.shuffle(ds, shuffle_opts)
        {:ok, wrap_dataset(shuffled, dataset_handle.id, dataset_handle.load_opts)}

      _ ->
        {:error, :shuffle_not_supported}
    end
  end

  @impl true
  @spec take(keyword(), term(), non_neg_integer()) :: {:ok, term()} | {:error, term()}
  def take(_opts, dataset_handle, count) do
    case unwrap_dataset(dataset_handle) do
      %HfDatasetsEx.Dataset{items: items} = ds ->
        taken_items = Enum.take(items, count)

        {:ok,
         wrap_dataset(%{ds | items: taken_items}, dataset_handle.id, dataset_handle.load_opts)}

      _ ->
        {:error, :take_not_supported}
    end
  end

  @impl true
  @spec skip(keyword(), term(), non_neg_integer()) :: {:ok, term()} | {:error, term()}
  def skip(_opts, dataset_handle, count) do
    case unwrap_dataset(dataset_handle) do
      %HfDatasetsEx.Dataset{items: items} = ds ->
        skipped_items = Enum.drop(items, count)

        {:ok,
         wrap_dataset(%{ds | items: skipped_items}, dataset_handle.id, dataset_handle.load_opts)}

      _ ->
        {:error, :skip_not_supported}
    end
  end

  @impl true
  @spec select(keyword(), term(), Range.t() | [non_neg_integer()]) ::
          {:ok, term()} | {:error, term()}
  def select(_opts, dataset_handle, selection) do
    case unwrap_dataset(dataset_handle) do
      %HfDatasetsEx.Dataset{items: items} = ds ->
        selected_items =
          case selection do
            %Range{} = range -> Enum.slice(items, range)
            indices when is_list(indices) -> Enum.map(indices, &Enum.at(items, &1))
          end

        {:ok,
         wrap_dataset(%{ds | items: selected_items}, dataset_handle.id, dataset_handle.load_opts)}

      _ ->
        {:error, :select_not_supported}
    end
  end

  @impl true
  @spec to_list(keyword(), term()) :: {:ok, [map()]} | {:error, term()}
  def to_list(_opts, dataset_handle) do
    case unwrap_dataset(dataset_handle) do
      %HfDatasetsEx.Dataset{items: items} -> {:ok, items}
      %HfDatasetsEx.IterableDataset{stream: stream} -> {:ok, Enum.to_list(stream)}
      _ -> {:error, :to_list_not_supported}
    end
  end

  # ============================================================================
  # Extended Functionality (not part of CrucibleTrain.Ports.DatasetStore)
  # ============================================================================

  @doc """
  Load a dataset from HuggingFace Hub or local path.

  This is the main loading function that supports multiple formats.
  """
  @spec load(keyword(), String.t() | atom(), keyword()) :: {:ok, term()} | {:error, term()}
  def load(opts, dataset_id, load_opts \\ []) do
    token = Keyword.get(opts, :token) || get_token()
    cache = Keyword.get(opts, :cache, true)
    split = Keyword.get(load_opts, :split)
    config = Keyword.get(load_opts, :config)
    streaming = Keyword.get(load_opts, :streaming, false)

    hf_opts =
      [
        token: token,
        split: split,
        config: config,
        streaming: streaming
      ]
      |> Keyword.merge(Keyword.take(load_opts, [:revision, :sample_size]))

    Logger.debug("Loading dataset: #{inspect(dataset_id)} (split: #{inspect(split)})")

    result =
      case dataset_id do
        id when is_atom(id) ->
          # Named dataset
          HfDatasetsEx.load(id, Keyword.put(hf_opts, :cache, cache))

        id when is_binary(id) ->
          if File.exists?(id) do
            # Local file
            HfDatasetsEx.load(Path.basename(id, ".jsonl"), source: id, cache: cache)
          else
            # HuggingFace repo
            HfDatasetsEx.load_dataset(id, hf_opts)
          end
      end

    case result do
      {:ok, dataset} ->
        Logger.debug("Dataset loaded: #{inspect(dataset_id)}")
        {:ok, wrap_dataset(dataset, dataset_id, load_opts)}

      {:error, reason} ->
        Logger.error("Failed to load dataset #{inspect(dataset_id)}: #{inspect(reason)}")
        {:error, {:load_failed, reason}}
    end
  end

  @doc """
  Stream dataset items.
  """
  @spec stream(keyword(), term(), keyword()) :: Enumerable.t()
  def stream(_opts, dataset_handle, stream_opts \\ []) do
    shuffle = Keyword.get(stream_opts, :shuffle, false)
    seed = Keyword.get(stream_opts, :seed)
    batch_size = Keyword.get(stream_opts, :batch_size)

    stream = get_stream(dataset_handle)

    stream =
      if shuffle do
        buffer_size = Keyword.get(stream_opts, :buffer_size, 1000)
        shuffle_stream(stream, buffer_size, seed)
      else
        stream
      end

    if batch_size do
      Stream.chunk_every(stream, batch_size)
    else
      stream
    end
  end

  @doc """
  Get a batch of items from a dataset.
  """
  @spec get_batch(keyword(), term(), non_neg_integer(), pos_integer()) ::
          {:ok, [map()]} | {:error, term()}
  def get_batch(_opts, dataset_handle, offset, batch_size) do
    items =
      dataset_handle
      |> get_stream()
      |> Stream.drop(offset)
      |> Enum.take(batch_size)

    {:ok, items}
  rescue
    e -> {:error, {:batch_error, Exception.message(e)}}
  end

  @doc """
  Get information about a dataset.
  """
  @spec info(keyword(), term()) :: {:ok, map()} | {:error, term()}
  def info(_opts, dataset_handle) do
    {:ok, extract_info(dataset_handle)}
  end

  @doc """
  Close a dataset handle and release resources.
  """
  @spec close(keyword(), term()) :: :ok
  def close(_opts, _dataset_handle) do
    # No resources to release for HfDatasets
    :ok
  end

  # ============================================================================
  # Extended Functionality
  # ============================================================================

  @doc """
  List available built-in datasets.
  """
  @spec list_available() :: [atom()]
  def list_available do
    HfDatasetsEx.list_available()
  end

  @doc """
  Get metadata for a named dataset.
  """
  @spec get_metadata(atom()) :: {:ok, map()} | {:error, term()}
  def get_metadata(dataset_name) when is_atom(dataset_name) do
    case HfDatasetsEx.get_metadata(dataset_name) do
      nil -> {:error, :not_found}
      metadata -> {:ok, metadata}
    end
  end

  @doc """
  Create a random sample from a dataset.
  """
  @spec random_sample(term(), keyword()) :: {:ok, term()} | {:error, term()}
  def random_sample(dataset_handle, opts \\ []) do
    case unwrap_dataset(dataset_handle) do
      %HfDatasetsEx.Dataset{} = ds ->
        sampled = HfDatasetsEx.random_sample(ds, opts)
        {:ok, wrap_dataset(sampled, dataset_handle.id, dataset_handle.load_opts)}

      _ ->
        {:error, :sampling_not_supported_for_streaming}
    end
  end

  @doc """
  Split dataset into train/test sets.
  """
  @spec train_test_split(term(), keyword()) :: {:ok, map()} | {:error, term()}
  def train_test_split(dataset_handle, opts \\ []) do
    case unwrap_dataset(dataset_handle) do
      %HfDatasetsEx.Dataset{} = ds ->
        splits = HfDatasetsEx.train_test_split(ds, opts)
        {:ok, splits}

      _ ->
        {:error, :split_not_supported_for_streaming}
    end
  end

  # ============================================================================
  # Private Functions
  # ============================================================================

  defp get_token do
    case HfHub.Auth.get_token() do
      {:ok, token} -> token
      _ -> System.get_env("HF_TOKEN")
    end
  end

  defp wrap_dataset(dataset, id, load_opts) do
    %{
      id: id,
      dataset: dataset,
      load_opts: load_opts,
      type: dataset_type(dataset)
    }
  end

  defp unwrap_dataset(%{dataset: dataset}), do: dataset
  defp unwrap_dataset(dataset), do: dataset

  defp dataset_type(%HfDatasetsEx.Dataset{}), do: :dataset
  defp dataset_type(%HfDatasetsEx.DatasetDict{}), do: :dataset_dict
  defp dataset_type(%HfDatasetsEx.IterableDataset{}), do: :iterable
  defp dataset_type(_), do: :unknown

  defp get_stream(%{dataset: %HfDatasetsEx.Dataset{items: items}}), do: Stream.map(items, & &1)

  defp get_stream(%{dataset: %HfDatasetsEx.IterableDataset{} = iterable}),
    do: iterable.stream

  defp get_stream(%{dataset: %HfDatasetsEx.DatasetDict{} = dd}) do
    # For DatasetDict, concatenate all splits
    dd
    |> Map.values()
    |> Enum.flat_map(& &1.items)
    |> Stream.map(& &1)
  end

  defp get_stream(%{dataset: items}) when is_list(items), do: Stream.map(items, & &1)
  defp get_stream(items) when is_list(items), do: Stream.map(items, & &1)

  defp shuffle_stream(stream, buffer_size, seed) do
    Stream.resource(
      fn ->
        if seed, do: :rand.seed(:exsss, {seed, seed, seed})
        {stream, []}
      end,
      fn {remaining, buffer} ->
        case Enum.take(remaining, max(0, buffer_size - length(buffer))) do
          [] when buffer == [] ->
            {:halt, nil}

          [] ->
            shuffled = Enum.shuffle(buffer)
            {shuffled, {[], []}}

          new_items ->
            new_buffer = buffer ++ new_items
            shuffled = Enum.shuffle(new_buffer)
            {emit, keep} = Enum.split(shuffled, 1)
            new_remaining = Stream.drop(remaining, length(new_items))
            {emit, {new_remaining, keep}}
        end
      end,
      fn _ -> :ok end
    )
  end

  defp extract_info(%{id: id, dataset: dataset, type: type, load_opts: load_opts}) do
    base_info = %{
      id: id,
      type: type,
      load_opts: load_opts
    }

    case dataset do
      %HfDatasetsEx.Dataset{} = ds ->
        Map.merge(base_info, %{
          size: length(ds.items),
          name: ds.name,
          metadata: ds.metadata
        })

      %HfDatasetsEx.IterableDataset{} = iterable ->
        Map.merge(base_info, %{
          size: :unknown,
          name: iterable.name,
          streaming: true,
          metadata: iterable.info
        })

      %HfDatasetsEx.DatasetDict{} = dd ->
        splits =
          dd
          |> Map.keys()
          |> Enum.map(fn split ->
            {split, length(dd[split].items)}
          end)
          |> Map.new()

        Map.merge(base_info, %{
          splits: splits,
          total_size: Enum.sum(Map.values(splits))
        })

      _ ->
        base_info
    end
  end
end
