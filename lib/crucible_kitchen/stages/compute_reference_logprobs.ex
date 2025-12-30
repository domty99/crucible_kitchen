defmodule CrucibleKitchen.Stages.ComputeReferenceLogprobs do
  @moduledoc """
  Computes log probabilities from frozen reference model for DPO training.

  In DPO, the reference model provides a baseline for the KL divergence
  constraint. This stage computes logprobs for both chosen and rejected
  responses using the reference model (usually the initial model before training).

  ## Context Requirements

  **Input:**
  - State: `:preference_batch` - Current batch of preference pairs
  - State: `:reference_session` - Reference model session (or uses main session)
  - Config: `:reference_model` - Reference model name (optional, defaults to base model)

  **Output:**
  - State: `:ref_chosen_logprobs` - Reference logprobs for chosen responses
  - State: `:ref_rejected_logprobs` - Reference logprobs for rejected responses

  ## Example

      stage(:compute_reference_logprobs, ComputeReferenceLogprobs)
  """

  use CrucibleKitchen.Stage

  alias CrucibleKitchen.Context

  require Logger

  @impl true
  def name, do: :compute_reference_logprobs

  @impl true
  def execute(context) do
    batch = Context.get_state(context, :preference_batch)

    # Use reference session if available, otherwise use main session
    ref_session =
      Context.get_state(context, :reference_session) ||
        Context.get_state(context, :session)

    case get_adapter(context, :training_client) do
      nil ->
        # No adapter - use mock logprobs for testing
        mock_logprobs(context, batch)

      {adapter, opts} ->
        compute_logprobs(context, adapter, opts, ref_session, batch)
    end
  end

  @impl true
  def validate(context) do
    case Context.get_state(context, :preference_batch) do
      nil -> {:error, "preference_batch is required in state"}
      _ -> :ok
    end
  end

  defp compute_logprobs(context, adapter, opts, session, batch) do
    Logger.debug("Computing reference logprobs for #{length(batch)} pairs")

    # Compute logprobs for chosen responses
    chosen_inputs =
      Enum.map(batch, fn pair ->
        %{prompt: pair.prompt, response: pair.chosen}
      end)

    # Compute logprobs for rejected responses
    rejected_inputs =
      Enum.map(batch, fn pair ->
        %{prompt: pair.prompt, response: pair.rejected}
      end)

    with {:ok, chosen_logprobs} <- adapter.compute_logprobs(opts, session, chosen_inputs),
         {:ok, rejected_logprobs} <- adapter.compute_logprobs(opts, session, rejected_inputs) do
      Logger.debug("Computed reference logprobs successfully")

      context
      |> Context.put_state(:ref_chosen_logprobs, chosen_logprobs)
      |> Context.put_state(:ref_rejected_logprobs, rejected_logprobs)
      |> then(&{:ok, &1})
    else
      {:error, reason} ->
        Logger.error("Failed to compute reference logprobs: #{inspect(reason)}")
        {:error, {:ref_logprobs_failed, reason}}
    end
  end

  defp mock_logprobs(context, batch) do
    # Generate mock logprobs for testing without real adapter
    Logger.debug("Using mock reference logprobs (no training_client adapter)")

    num_pairs = length(batch)

    # Mock logprobs as random negative values (log probs are always <= 0)
    chosen_logprobs = Enum.map(1..num_pairs, fn _ -> -:rand.uniform() * 10 end)
    rejected_logprobs = Enum.map(1..num_pairs, fn _ -> -:rand.uniform() * 10 end)

    context
    |> Context.put_state(:ref_chosen_logprobs, chosen_logprobs)
    |> Context.put_state(:ref_rejected_logprobs, rejected_logprobs)
    |> then(&{:ok, &1})
  end
end
