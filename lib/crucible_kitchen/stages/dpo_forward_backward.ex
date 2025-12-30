defmodule CrucibleKitchen.Stages.DPOForwardBackward do
  @moduledoc """
  DPO forward-backward pass with beta-scaled preference loss.

  Implements the DPO loss function:

      L = -log(sigmoid(beta * (log_pi_chosen - log_pi_rejected - log_ref_chosen + log_ref_rejected)))

  This loss encourages the policy to prefer chosen responses over rejected ones,
  with the reference model providing a KL constraint via the log probability differences.

  ## Context Requirements

  **Input:**
  - State: `:preference_batch` - Current batch of preference pairs
  - State: `:ref_chosen_logprobs` - Reference model logprobs for chosen
  - State: `:ref_rejected_logprobs` - Reference model logprobs for rejected
  - State: `:session` - Training session
  - Config: `:dpo_beta` - DPO beta parameter (default: 0.1)

  **Output:**
  - State: `:dpo_future` - Async future for the DPO computation
  - State: `:dpo_loss` - DPO loss value (after await)
  - State: `:dpo_metrics` - DPO-specific metrics

  ## Example

      stage(:dpo_forward_backward, DPOForwardBackward)
      stage(:await_dpo, AwaitFuture, key: :dpo_future)
  """

  use CrucibleKitchen.Stage

  alias CrucibleKitchen.Context

  require Logger

  @impl true
  def name, do: :dpo_forward_backward

  @impl true
  def execute(context) do
    session = Context.get_state(context, :session)
    batch = Context.get_state(context, :preference_batch)
    ref_chosen = Context.get_state(context, :ref_chosen_logprobs)
    ref_rejected = Context.get_state(context, :ref_rejected_logprobs)
    beta = Context.get_config(context, :dpo_beta, 0.1)

    case get_adapter(context, :training_client) do
      nil ->
        # No adapter - mock DPO computation
        mock_dpo(context, batch, beta)

      {adapter, opts} ->
        run_dpo(context, adapter, opts, session, batch, ref_chosen, ref_rejected, beta)
    end
  end

  @impl true
  def validate(context) do
    cond do
      Context.get_state(context, :preference_batch) == nil ->
        {:error, "preference_batch is required in state"}

      Context.get_state(context, :ref_chosen_logprobs) == nil ->
        {:error, "ref_chosen_logprobs is required (run ComputeReferenceLogprobs first)"}

      Context.get_state(context, :ref_rejected_logprobs) == nil ->
        {:error, "ref_rejected_logprobs is required (run ComputeReferenceLogprobs first)"}

      true ->
        :ok
    end
  end

  defp run_dpo(context, adapter, opts, session, batch, ref_chosen, ref_rejected, beta) do
    Logger.debug("Running DPO forward-backward with beta=#{beta}")

    dpo_input = %{
      batch: batch,
      ref_chosen_logprobs: ref_chosen,
      ref_rejected_logprobs: ref_rejected,
      beta: beta
    }

    # Start async DPO computation
    future = adapter.dpo_forward_backward(opts, session, dpo_input)

    context
    |> Context.put_state(:dpo_future, future)
    |> then(&{:ok, &1})
  end

  defp mock_dpo(context, batch, beta) do
    Logger.debug("Using mock DPO computation (no training_client adapter)")

    # Simulate DPO metrics
    num_pairs = length(batch)

    mock_metrics = %{
      loss: :rand.uniform() * 0.5 + 0.3,
      accuracy: :rand.uniform() * 0.3 + 0.5,
      chosen_reward: :rand.uniform() * 2 - 1,
      rejected_reward: :rand.uniform() * 2 - 2,
      margin: :rand.uniform() * 0.5 + 0.1,
      beta: beta,
      num_pairs: num_pairs
    }

    # Return immediately (no async)
    context
    |> Context.put_state(:dpo_future, {:ok, mock_metrics})
    |> Context.put_state(:dpo_loss, mock_metrics.loss)
    |> Context.put_state(:dpo_metrics, mock_metrics)
    |> then(&{:ok, &1})
  end
end
