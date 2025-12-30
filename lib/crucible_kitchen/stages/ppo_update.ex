defmodule CrucibleKitchen.Stages.PPOUpdate do
  @moduledoc """
  PPO policy gradient update with clipping.

  Implements Proximal Policy Optimization (Schulman et al., 2017) with
  clipped surrogate objective:

      L = min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)

  Where:
  - ratio = π(a|s) / π_old(a|s)
  - A = advantage estimate
  - ε = clip_epsilon (typically 0.2)

  ## Context Requirements

  **Input:**
  - State: `:rl_batch` - Training batch from AssembleRLBatch
  - State: `:session` - Training session
  - Config: `:clip_epsilon` - Clipping parameter (default: 0.2)
  - Config: `:ppo_epochs` - Number of PPO epochs (default: 4)
  - Config: `:kl_penalty_coef` - KL penalty coefficient (default: 0.0)

  **Output:**
  - State: `:ppo_metrics` - PPO training metrics

  ## Example

      stage(:ppo_update, PPOUpdate)
  """

  use CrucibleKitchen.Stage

  alias CrucibleKitchen.Context

  require Logger

  @impl true
  def name, do: :ppo_update

  @impl true
  def execute(context) do
    session = Context.get_state(context, :session)
    batch = Context.get_state(context, :rl_batch)
    clip_epsilon = Context.get_config(context, :clip_epsilon, 0.2)
    ppo_epochs = Context.get_config(context, :ppo_epochs, 4)
    kl_penalty = Context.get_config(context, :kl_penalty_coef, 0.0)

    case get_adapter(context, :training_client) do
      nil ->
        mock_ppo(context, batch, clip_epsilon, ppo_epochs)

      {adapter, opts} ->
        run_ppo(context, adapter, opts, session, batch, clip_epsilon, ppo_epochs, kl_penalty)
    end
  end

  @impl true
  def validate(context) do
    case Context.get_state(context, :rl_batch) do
      nil -> {:error, "rl_batch is required (run AssembleRLBatch first)"}
      _ -> :ok
    end
  end

  defp run_ppo(context, adapter, opts, session, batch, clip_epsilon, ppo_epochs, kl_penalty) do
    Logger.debug(
      "Running PPO update: epochs=#{ppo_epochs} " <>
        "clip_epsilon=#{clip_epsilon} kl_penalty=#{kl_penalty}"
    )

    # Run multiple PPO epochs on the same batch
    {final_metrics, all_metrics} =
      Enum.reduce(1..ppo_epochs, {nil, []}, fn epoch, {_prev, acc} ->
        ppo_input = %{
          batch: batch,
          clip_epsilon: clip_epsilon,
          kl_penalty_coef: kl_penalty,
          epoch: epoch
        }

        case adapter.ppo_step(opts, session, ppo_input) do
          {:ok, metrics} ->
            Logger.debug(
              "PPO epoch #{epoch}: policy_loss=#{Float.round(metrics.policy_loss, 4)} " <>
                "clip_fraction=#{Float.round(metrics.clip_fraction, 4)}"
            )

            {metrics, [metrics | acc]}

          {:error, reason} ->
            Logger.error("PPO epoch #{epoch} failed: #{inspect(reason)}")
            {%{policy_loss: 0.0, clip_fraction: 0.0, entropy: 0.0}, acc}
        end
      end)

    # Aggregate metrics across epochs
    ppo_metrics = aggregate_metrics(final_metrics, Enum.reverse(all_metrics))

    context
    |> Context.put_state(:ppo_metrics, ppo_metrics)
    |> then(&{:ok, &1})
  end

  defp mock_ppo(context, batch, clip_epsilon, ppo_epochs) do
    Logger.debug("Using mock PPO update (no training_client adapter)")

    # Simulate PPO metrics
    ppo_metrics = %{
      policy_loss: :rand.uniform() * 0.1,
      value_loss: :rand.uniform() * 0.5,
      entropy: :rand.uniform() * 1.0,
      clip_fraction: :rand.uniform() * 0.3,
      kl_divergence: :rand.uniform() * 0.01,
      approx_kl: :rand.uniform() * 0.005,
      clip_epsilon: clip_epsilon,
      ppo_epochs: ppo_epochs,
      batch_size: batch.batch_size
    }

    context
    |> Context.put_state(:ppo_metrics, ppo_metrics)
    |> then(&{:ok, &1})
  end

  defp aggregate_metrics(nil, _all), do: %{}

  defp aggregate_metrics(final, all_metrics) do
    # Final epoch metrics plus aggregates
    Map.merge(final, %{
      mean_policy_loss: mean_field(all_metrics, :policy_loss),
      mean_clip_fraction: mean_field(all_metrics, :clip_fraction),
      num_epochs: length(all_metrics)
    })
  end

  defp mean_field(metrics, field) do
    values = Enum.map(metrics, &Map.get(&1, field, 0.0))

    case values do
      [] -> 0.0
      values -> Enum.sum(values) / length(values)
    end
  end
end
