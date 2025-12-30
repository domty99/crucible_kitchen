defmodule CrucibleKitchen.Stages.GetPreferenceBatch do
  @moduledoc """
  Gets next preference batch from dataset for DPO training.

  Fetches the next batch of (prompt, chosen, rejected) triplets
  for the current training step.

  ## Context Requirements

  **Input:**
  - State: `:preference_dataset` - Preference dataset from BuildPreferenceDataset

  **Output:**
  - State: `:preference_batch` - Current batch of preference pairs
  - State: `:batch_index` - Current batch index

  ## Example

      loop :pref_batches, over: :pref_batches_range do
        stage(:get_preference_batch, GetPreferenceBatch)
        # ... training stages
      end
  """

  use CrucibleKitchen.Stage

  alias CrucibleKitchen.Context

  require Logger

  @impl true
  def name, do: :get_preference_batch

  @impl true
  def execute(context) do
    dataset = Context.get_state(context, :preference_dataset)
    batch_index = Context.get_state(context, :pref_batches_index, 0)

    batch = get_batch(dataset, batch_index)

    Logger.debug(
      "Got preference batch #{batch_index + 1}/#{dataset.num_batches} " <>
        "with #{length(batch)} pairs"
    )

    context
    |> Context.put_state(:preference_batch, batch)
    |> Context.put_state(:batch_index, batch_index)
    |> then(&{:ok, &1})
  end

  @impl true
  def validate(context) do
    case Context.get_state(context, :preference_dataset) do
      nil -> {:error, "preference_dataset is required in state"}
      _ -> :ok
    end
  end

  defp get_batch(dataset, batch_index) do
    start_idx = batch_index * dataset.batch_size
    end_idx = min(start_idx + dataset.batch_size, dataset.num_pairs)

    dataset.pairs
    |> Enum.slice(start_idx, end_idx - start_idx)
  end
end
