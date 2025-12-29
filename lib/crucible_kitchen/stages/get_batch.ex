defmodule CrucibleKitchen.Stages.GetBatch do
  @moduledoc """
  Stage for getting the current batch from the dataset.

  Retrieves a batch by index from the dataset and stores it for processing.

  ## State Requirements

  - `:dataset` - The built supervised dataset
  - `:batch_index` - Current batch index (set by loop)

  ## State Updates

  - `:current_batch` - The batch of datums to process
  """

  use CrucibleKitchen.Stage

  alias CrucibleTrain.Supervised.Dataset

  require Logger

  @impl true
  def name, do: :get_batch

  @impl true
  def execute(context) do
    dataset = get_state(context, :dataset)
    batch_index = get_state(context, :batch_index, 0)

    batch = get_batch_from_dataset(dataset, batch_index)

    context =
      context
      |> put_state(:current_batch, batch)
      |> record_metric(:batch_size, length(batch))

    {:ok, context}
  end

  defp get_batch_from_dataset(dataset, index) when is_map(dataset) do
    # Support for CrucibleTrain.Supervised.Dataset
    if match?(%{__struct__: _}, dataset) and function_exported?(Dataset, :get_batch, 2) do
      Dataset.get_batch(dataset, index)
    else
      Map.get(dataset, :batches, []) |> Enum.at(index, [])
    end
  end

  defp get_batch_from_dataset(dataset, index) when is_list(dataset) do
    Enum.at(dataset, index, [])
  end

  defp get_batch_from_dataset(_dataset, _index), do: []
end
