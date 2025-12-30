defmodule CrucibleKitchen.Stages.AssembleRLBatch do
  @moduledoc """
  Assembles trajectory data into training batch for PPO update.

  Converts the collected trajectories with computed advantages into
  a format suitable for policy gradient training.

  ## Context Requirements

  **Input:**
  - State: `:trajectory_group` - Trajectories from DoRollout
  - State: `:normalized_advantages` - Advantages from ComputeAdvantages
  - State: `:returns` - Returns from ComputeAdvantages

  **Output:**
  - State: `:rl_batch` - Assembled training batch

  ## Batch Structure

  The RL batch contains:
  - `observations` - Flattened observations across trajectories
  - `actions` - Flattened actions
  - `old_logprobs` - Log probabilities from rollout policy
  - `advantages` - Normalized advantages
  - `returns` - Discounted returns (for value function training)

  ## Example

      stage(:assemble_rl_batch, AssembleRLBatch)
  """

  use CrucibleKitchen.Stage

  alias CrucibleKitchen.Context

  require Logger

  @impl true
  def name, do: :assemble_rl_batch

  @impl true
  def execute(context) do
    trajectory_group = Context.get_state(context, :trajectory_group)
    advantages = Context.get_state(context, :normalized_advantages)
    returns = Context.get_state(context, :returns)

    Logger.debug("Assembling RL batch from #{trajectory_group.num_trajectories} trajectories")

    trajectories = trajectory_group.trajectories

    # Flatten all trajectory data
    observations = Enum.flat_map(trajectories, & &1.observations)
    actions = Enum.flat_map(trajectories, & &1.actions)
    old_logprobs = Enum.flat_map(trajectories, & &1.logprobs)

    batch_size = length(observations)

    rl_batch = %{
      observations: observations,
      actions: actions,
      old_logprobs: old_logprobs,
      advantages: advantages,
      returns: returns,
      batch_size: batch_size,
      num_trajectories: trajectory_group.num_trajectories
    }

    Logger.debug("Assembled batch with #{batch_size} steps")

    emit_telemetry(rl_batch)

    context
    |> Context.put_state(:rl_batch, rl_batch)
    |> then(&{:ok, &1})
  end

  @impl true
  def validate(context) do
    cond do
      Context.get_state(context, :trajectory_group) == nil ->
        {:error, "trajectory_group is required (run DoRollout first)"}

      Context.get_state(context, :normalized_advantages) == nil ->
        {:error, "normalized_advantages is required (run ComputeAdvantages first)"}

      true ->
        :ok
    end
  end

  defp emit_telemetry(batch) do
    :telemetry.execute(
      [:crucible_kitchen, :rl, :batch_assembled],
      %{
        batch_size: batch.batch_size,
        num_trajectories: batch.num_trajectories
      },
      %{}
    )
  end
end
