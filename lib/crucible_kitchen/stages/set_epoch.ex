defmodule CrucibleKitchen.Stages.SetEpoch do
  @moduledoc """
  Stage for setting the current epoch on the dataset.

  Updates the dataset state for the new epoch (may trigger shuffling, etc.)

  ## State Requirements

  - `:dataset` - The supervised dataset
  - `:epoch_index` - Current epoch index (set by loop)

  ## State Updates

  - `:dataset` - Updated dataset with new epoch
  - `:current_epoch` - Current epoch number
  """

  use CrucibleKitchen.Stage

  alias CrucibleTrain.Supervised.Dataset

  require Logger

  @impl true
  def name, do: :set_epoch

  @impl true
  def execute(context) do
    dataset = get_state(context, :dataset)
    epoch_index = get_state(context, :epoch_index, 0)

    Logger.info("[SetEpoch] Starting epoch #{epoch_index + 1}")

    # Update dataset for new epoch if supported
    updated_dataset =
      if match?(%{__struct__: _}, dataset) and function_exported?(Dataset, :set_epoch, 2) do
        Dataset.set_epoch(dataset, epoch_index)
      else
        dataset
      end

    context =
      context
      |> put_state(:dataset, updated_dataset)
      |> put_state(:current_epoch, epoch_index)
      |> record_metric(:epoch, epoch_index + 1)

    {:ok, context}
  end
end
