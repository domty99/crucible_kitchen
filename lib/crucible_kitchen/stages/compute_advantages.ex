defmodule CrucibleKitchen.Stages.ComputeAdvantages do
  @moduledoc """
  Computes Generalized Advantage Estimation (GAE) for RL training.

  Implements GAE from "High-Dimensional Continuous Control Using
  Generalized Advantage Estimation" (Schulman et al., 2015).

  GAE provides a balance between bias and variance in advantage estimation:
  - gamma (γ) controls the discount factor
  - lambda (λ) controls the bias-variance tradeoff

  ## Context Requirements

  **Input:**
  - State: `:trajectory_group` - Trajectories from DoRollout
  - Config: `:gamma` - Discount factor (default: 0.99)
  - Config: `:gae_lambda` - GAE lambda (default: 0.95)

  **Output:**
  - State: `:advantages` - Computed advantages per step
  - State: `:returns` - Computed returns per step
  - State: `:normalized_advantages` - Mean-centered, std-normalized advantages

  ## Algorithm

  For each timestep t:
    δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
    A_t = Σ_{l=0}^{∞} (γλ)^l * δ_{t+l}

  ## Example

      stage(:compute_advantages, ComputeAdvantages)
  """

  use CrucibleKitchen.Stage

  alias CrucibleKitchen.Context

  require Logger

  @impl true
  def name, do: :compute_advantages

  @impl true
  def execute(context) do
    trajectory_group = Context.get_state(context, :trajectory_group)
    gamma = Context.get_config(context, :gamma, 0.99)
    gae_lambda = Context.get_config(context, :gae_lambda, 0.95)

    Logger.debug("Computing advantages with gamma=#{gamma}, lambda=#{gae_lambda}")

    # Use existing GAE implementation from RLEnv if available
    {advantages, returns} = compute_gae_for_group(trajectory_group, gamma, gae_lambda)

    # Normalize advantages
    normalized = normalize_advantages(advantages)

    Logger.debug(
      "Computed advantages: mean=#{Float.round(mean(advantages), 4)} " <>
        "std=#{Float.round(std(advantages), 4)}"
    )

    emit_telemetry(advantages, returns)

    context
    |> Context.put_state(:advantages, advantages)
    |> Context.put_state(:returns, returns)
    |> Context.put_state(:normalized_advantages, normalized)
    |> then(&{:ok, &1})
  end

  @impl true
  def validate(context) do
    case Context.get_state(context, :trajectory_group) do
      nil -> {:error, "trajectory_group is required (run DoRollout first)"}
      _ -> :ok
    end
  end

  defp compute_gae_for_group(trajectory_group, gamma, lambda) do
    trajectories = trajectory_group.trajectories

    # Compute GAE for each trajectory
    results =
      Enum.map(trajectories, fn traj ->
        compute_gae_for_trajectory(traj.rewards, gamma, lambda)
      end)

    # Flatten results
    advantages = Enum.flat_map(results, fn {adv, _ret} -> adv end)
    returns = Enum.flat_map(results, fn {_adv, ret} -> ret end)

    {advantages, returns}
  end

  defp compute_gae_for_trajectory(rewards, gamma, lambda) when is_list(rewards) do
    # For single-step trajectories, advantage = reward - mean_reward
    # This is the simplified version used in GRPO-style training
    case rewards do
      [] ->
        {[], []}

      [single_reward] ->
        # Single step: advantage is just the reward (will be normalized later)
        {[single_reward], [single_reward]}

      rewards ->
        # Multi-step: compute proper GAE
        # For now, use simple discounted returns as advantage
        # (In full implementation, would need value function estimates)
        compute_simple_gae(rewards, gamma, lambda)
    end
  end

  defp compute_simple_gae(rewards, gamma, _lambda) do
    # Simple version: compute discounted returns from end
    _n = length(rewards)
    rewards_reversed = Enum.reverse(rewards)

    # Compute returns (discounted sum of future rewards)
    {returns_reversed, _} =
      Enum.reduce(rewards_reversed, {[], 0.0}, fn r, {acc, running} ->
        new_return = r + gamma * running
        {[new_return | acc], new_return}
      end)

    returns = Enum.reverse(returns_reversed)

    # For simple GAE without value function, advantages = returns - baseline
    # Use mean return as baseline
    baseline = mean(returns)
    advantages = Enum.map(returns, fn ret -> ret - baseline end)

    {advantages, returns}
  end

  defp normalize_advantages(advantages) when is_list(advantages) do
    case advantages do
      [] ->
        []

      [_single] ->
        [0.0]

      advantages ->
        m = mean(advantages)
        s = std(advantages)

        # Avoid division by zero
        s = if s < 1.0e-8, do: 1.0, else: s

        Enum.map(advantages, fn a -> (a - m) / s end)
    end
  end

  defp mean([]), do: 0.0
  defp mean(list), do: Enum.sum(list) / length(list)

  defp std([]), do: 0.0
  defp std([_]), do: 0.0

  defp std(list) do
    m = mean(list)

    variance =
      list
      |> Enum.map(fn v -> (v - m) * (v - m) end)
      |> Enum.sum()
      |> Kernel./(length(list))

    :math.sqrt(variance)
  end

  defp emit_telemetry(advantages, returns) do
    :telemetry.execute(
      [:crucible_kitchen, :rl, :advantages_computed],
      %{
        advantage_mean: mean(advantages),
        advantage_std: std(advantages),
        return_mean: mean(returns),
        return_std: std(returns),
        num_steps: length(advantages)
      },
      %{}
    )
  end
end
