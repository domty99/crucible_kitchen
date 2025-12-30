defmodule CrucibleKitchen.Stages.BuildPreferenceDataset do
  @moduledoc """
  Builds preference dataset from raw comparison data for DPO training.

  This stage converts loaded comparison data into preference pairs suitable
  for Direct Preference Optimization training.

  ## Context Requirements

  **Input:**
  - State: `:raw_dataset` - Raw dataset with preference comparisons
  - Config: `:batch_size` - Batch size for training

  **Output:**
  - State: `:preference_dataset` - Processed preference dataset with num_batches
  - State: `:num_preference_pairs` - Total number of preference pairs

  ## Dataset Format

  Input comparisons should have:
  - `prompt` - The input prompt
  - `chosen` - The preferred response
  - `rejected` - The rejected response

  ## Example

      stage(:build_preference_dataset, BuildPreferenceDataset)
  """

  use CrucibleKitchen.Stage

  alias CrucibleKitchen.Context

  require Logger

  @impl true
  def name, do: :build_preference_dataset

  @impl true
  def execute(context) do
    raw_dataset = Context.get_state(context, :raw_dataset)
    tokenizer = Context.get_state(context, :tokenizer)
    batch_size = Context.get_config(context, :batch_size, 256)
    max_length = Context.get_config(context, :max_length, 8192)

    Logger.info("Building preference dataset with batch_size=#{batch_size}")

    # Convert raw dataset to preference pairs
    pairs = build_preference_pairs(raw_dataset, tokenizer, max_length)
    num_pairs = length(pairs)
    num_batches = ceil(num_pairs / batch_size)

    Logger.info("Built #{num_pairs} preference pairs in #{num_batches} batches")

    preference_dataset = %{
      pairs: pairs,
      batch_size: batch_size,
      num_batches: num_batches,
      num_pairs: num_pairs,
      current_batch: 0,
      epoch: 0
    }

    emit_telemetry(num_pairs, num_batches)

    context
    |> Context.put_state(:preference_dataset, preference_dataset)
    |> Context.put_state(:num_preference_pairs, num_pairs)
    |> Context.record_metric(:preference_pairs, num_pairs)
    |> then(&{:ok, &1})
  end

  @impl true
  def validate(context) do
    case Context.get_state(context, :raw_dataset) do
      nil -> {:error, "raw_dataset is required in state for preference dataset building"}
      _ -> :ok
    end
  end

  defp build_preference_pairs(raw_dataset, tokenizer, max_length) do
    raw_dataset
    |> Enum.map(fn example ->
      %{
        prompt: example["prompt"] || example[:prompt],
        chosen: example["chosen"] || example[:chosen],
        rejected: example["rejected"] || example[:rejected],
        prompt_tokens:
          maybe_tokenize(tokenizer, example["prompt"] || example[:prompt], max_length),
        chosen_tokens:
          maybe_tokenize(tokenizer, example["chosen"] || example[:chosen], max_length),
        rejected_tokens:
          maybe_tokenize(tokenizer, example["rejected"] || example[:rejected], max_length)
      }
    end)
    |> Enum.filter(&valid_pair?/1)
  end

  defp maybe_tokenize(nil, _text, _max_length), do: nil

  defp maybe_tokenize(tokenizer, text, max_length) when is_binary(text) do
    tokens = tokenizer.encode(text)

    if length(tokens) > max_length do
      Enum.take(tokens, max_length)
    else
      tokens
    end
  end

  defp maybe_tokenize(_tokenizer, _text, _max_length), do: nil

  defp valid_pair?(%{prompt: p, chosen: c, rejected: r})
       when is_binary(p) and is_binary(c) and is_binary(r) do
    byte_size(p) > 0 and byte_size(c) > 0 and byte_size(r) > 0
  end

  defp valid_pair?(_), do: false

  defp emit_telemetry(num_pairs, num_batches) do
    :telemetry.execute(
      [:crucible_kitchen, :dpo, :dataset_built],
      %{num_pairs: num_pairs, num_batches: num_batches},
      %{}
    )
  end
end
