defmodule CrucibleKitchen.Adapters.Noop.DatasetStore do
  @moduledoc """
  Noop adapter for DatasetStore port.

  Returns synthetic data. Useful for:
  - Testing stages in isolation
  - Benchmarking data pipeline throughput
  - Development without real datasets
  """

  @behaviour CrucibleTrain.Ports.DatasetStore

  @impl true
  def load_dataset(_opts, dataset_id, load_opts) do
    size = Keyword.get(load_opts, :size, 1000)
    split = Keyword.get(load_opts, :split, "train")

    items = build_items(dataset_id, split, size)

    dataset = %{
      id: dataset_id,
      size: size,
      loaded_at: DateTime.utc_now(),
      split: split,
      items: items
    }

    {:ok, dataset}
  end

  @impl true
  def get_split(_opts, %{splits: splits}, split) when is_map(splits) do
    split_key = if is_atom(split), do: Atom.to_string(split), else: split

    case Map.get(splits, split_key) do
      nil -> {:error, {:split_not_found, split}}
      split_dataset -> {:ok, split_dataset}
    end
  end

  def get_split(_opts, dataset, split) when is_map(dataset) do
    {:ok, Map.put(dataset, :split, split)}
  end

  def get_split(_opts, _dataset, split), do: {:error, {:split_not_found, split}}

  @impl true
  def shuffle(_opts, dataset, shuffle_opts \\ []) do
    case Map.get(dataset, :items) do
      items when is_list(items) ->
        seed = Keyword.get(shuffle_opts, :seed)
        if seed, do: :rand.seed(:exsss, {seed, seed, seed})
        {:ok, update_items(dataset, Enum.shuffle(items))}

      _ ->
        {:error, :shuffle_not_supported}
    end
  end

  @impl true
  def take(_opts, dataset, count) do
    case Map.get(dataset, :items) do
      items when is_list(items) ->
        {:ok, update_items(dataset, Enum.take(items, count))}

      _ ->
        {:error, :take_not_supported}
    end
  end

  @impl true
  def skip(_opts, dataset, count) do
    case Map.get(dataset, :items) do
      items when is_list(items) ->
        {:ok, update_items(dataset, Enum.drop(items, count))}

      _ ->
        {:error, :skip_not_supported}
    end
  end

  @impl true
  def select(_opts, dataset, selection) do
    case Map.get(dataset, :items) do
      items when is_list(items) ->
        selected_items =
          case selection do
            %Range{} = range -> Enum.slice(items, range)
            indices when is_list(indices) -> Enum.map(indices, &Enum.at(items, &1))
          end

        {:ok, update_items(dataset, Enum.reject(selected_items, &is_nil/1))}

      _ ->
        {:error, :select_not_supported}
    end
  end

  @impl true
  def to_list(_opts, dataset) do
    case Map.get(dataset, :items) do
      items when is_list(items) -> {:ok, items}
      _ -> {:error, :to_list_not_supported}
    end
  end

  # ==========================================================================
  # Extended Functionality (not part of CrucibleTrain.Ports.DatasetStore)
  # ==========================================================================

  def load(opts, dataset_id, load_opts) do
    load_dataset(opts, dataset_id, load_opts)
  end

  def stream(_opts, dataset, stream_opts) do
    shuffle = Keyword.get(stream_opts, :shuffle, false)
    seed = Keyword.get(stream_opts, :seed, 42)
    batch_size = Keyword.get(stream_opts, :batch_size)

    items = Map.get(dataset, :items, [])

    items =
      if shuffle do
        :rand.seed(:exsss, {seed, seed, seed})
        Enum.shuffle(items)
      else
        items
      end

    stream = Stream.map(items, & &1)

    if batch_size do
      Stream.chunk_every(stream, batch_size)
    else
      stream
    end
  end

  def get_batch(_opts, dataset, offset, batch_size) do
    items = Map.get(dataset, :items, [])

    examples =
      items
      |> Enum.drop(offset)
      |> Enum.take(batch_size)

    {:ok, examples}
  end

  def info(_opts, dataset) do
    info = %{
      size: Map.get(dataset, :size, 0),
      split: Map.get(dataset, :split, "train"),
      columns: [:id, :text, :label, :metadata],
      schema: %{
        id: :integer,
        text: :string,
        label: :integer,
        metadata: :map
      }
    }

    {:ok, info}
  end

  def close(_opts, _dataset) do
    :ok
  end

  defp build_items(_dataset_id, _split, size) when size <= 0, do: []

  defp build_items(dataset_id, split, size) do
    Enum.map(0..(size - 1), fn i ->
      %{
        id: i,
        text: "Example #{i} from dataset #{dataset_id}",
        label: rem(i, 2),
        metadata: %{split: split}
      }
    end)
  end

  defp update_items(dataset, items) do
    dataset
    |> Map.put(:items, items)
    |> Map.put(:size, length(items))
  end
end
