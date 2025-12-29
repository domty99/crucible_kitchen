defmodule CrucibleKitchen.Stages.LoadDataset do
  @moduledoc """
  Stage for loading a dataset from the dataset store.

  Reads dataset configuration and loads it using the dataset_store adapter.
  Stores the result in context state as `:dataset_handle`.

  ## Configuration

  - `:dataset` - Dataset identifier (string or atom)
  - `:split` - Dataset split to load (e.g., "train")
  - `:sample_size` - Optional limit on samples
  """

  use CrucibleKitchen.Stage

  alias CrucibleTrain.Ports.DatasetStore

  require Logger

  @impl true
  def name, do: :load_dataset

  @impl true
  def execute(context) do
    dataset_id = get_config(context, :dataset)
    split = get_config(context, :split, "train")
    sample_size = get_config(context, :sample_size, nil)

    Logger.debug("[LoadDataset] Loading dataset: #{inspect(dataset_id)}, split: #{split}")

    ports = get_train_ports(context)

    load_opts =
      [split: split]
      |> maybe_add(:sample_size, sample_size)

    case DatasetStore.load_dataset(ports, dataset_id, load_opts) do
      {:ok, dataset_handle} ->
        Logger.info("[LoadDataset] Dataset loaded successfully")

        context =
          context
          |> put_state(:dataset_handle, dataset_handle)
          |> record_metric(:dataset_loaded, 1)

        {:ok, context}

      {:error, reason} ->
        Logger.error("[LoadDataset] Failed to load dataset: #{inspect(reason)}")
        {:error, {:dataset_load_failed, reason}}
    end
  end

  defp maybe_add(opts, _key, nil), do: opts
  defp maybe_add(opts, key, value), do: Keyword.put(opts, key, value)
end
