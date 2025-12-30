defmodule CrucibleKitchen.Stages.DoRollout do
  @moduledoc """
  Collects trajectories via parallel rollouts for RL training.

  Executes rollouts in parallel across the environment group, collecting
  trajectories with observations, actions, rewards, and done flags.

  ## Context Requirements

  **Input:**
  - State: `:env_group` - Environment group from BuildEnvGroup
  - State: `:session` - Training session for action sampling
  - Config: `:max_tokens` - Maximum tokens per response (default: 512)

  **Output:**
  - State: `:trajectory_group` - Collected trajectories with rewards
  - State: `:rollout_metrics` - Metrics from rollout collection

  ## Trajectory Structure

  Each trajectory contains:
  - `observations` - List of observations (prompts)
  - `actions` - List of actions (responses)
  - `rewards` - List of rewards
  - `dones` - List of done flags
  - `logprobs` - List of action log probabilities

  ## Example

      loop :rollouts, over: :rollouts_range do
        stage(:do_rollout, DoRollout)
        # ... advantage computation and training
      end
  """

  use CrucibleKitchen.Stage

  alias CrucibleKitchen.Context

  require Logger

  @impl true
  def name, do: :do_rollout

  @impl true
  def execute(context) do
    env_group = Context.get_state(context, :env_group)
    session = Context.get_state(context, :session)
    rollout_index = Context.get_state(context, :rollouts_index, 0)
    max_tokens = Context.get_config(context, :max_tokens, 512)

    case get_adapter(context, :training_client) do
      nil ->
        # No adapter - use mock rollout
        mock_rollout(context, env_group, rollout_index)

      {adapter, opts} ->
        run_rollout(context, adapter, opts, session, env_group, rollout_index, max_tokens)
    end
  end

  @impl true
  def validate(context) do
    case Context.get_state(context, :env_group) do
      nil -> {:error, "env_group is required (run BuildEnvGroup first)"}
      _ -> :ok
    end
  end

  defp run_rollout(context, adapter, opts, session, env_group, rollout_index, max_tokens) do
    Logger.debug("Starting rollout #{rollout_index + 1}")

    start_time = System.monotonic_time(:millisecond)

    # Create environments for this rollout
    envs = env_group.make_envs(%{rollout_index: rollout_index})

    # Collect trajectories in parallel
    trajectories =
      envs
      |> Task.async_stream(
        fn env ->
          collect_trajectory(adapter, opts, session, env, max_tokens)
        end,
        max_concurrency: length(envs),
        timeout: 60_000
      )
      |> Enum.map(fn {:ok, traj} -> traj end)

    duration_ms = System.monotonic_time(:millisecond) - start_time

    # Compute group-level metrics
    rewards = Enum.flat_map(trajectories, & &1.rewards)
    reward_mean = if rewards == [], do: 0.0, else: Enum.sum(rewards) / length(rewards)
    reward_std = compute_std(rewards, reward_mean)

    trajectory_group = %{
      trajectories: trajectories,
      rewards: rewards,
      reward_mean: reward_mean,
      reward_std: reward_std,
      num_trajectories: length(trajectories),
      rollout_index: rollout_index
    }

    rollout_metrics = %{
      duration_ms: duration_ms,
      num_trajectories: length(trajectories),
      total_steps: Enum.sum(Enum.map(trajectories, &length(&1.rewards))),
      reward_mean: reward_mean,
      reward_std: reward_std
    }

    Logger.debug(
      "Rollout #{rollout_index + 1} complete: " <>
        "#{length(trajectories)} trajectories, " <>
        "reward_mean=#{Float.round(reward_mean, 4)}"
    )

    emit_telemetry(rollout_index, rollout_metrics)

    context
    |> Context.put_state(:trajectory_group, trajectory_group)
    |> Context.put_state(:rollout_metrics, rollout_metrics)
    |> then(&{:ok, &1})
  end

  defp collect_trajectory(adapter, opts, session, env, max_tokens) do
    # Single trajectory collection
    initial_obs = env.observation

    # Sample action from policy
    case adapter.sample(opts, session, initial_obs, max_tokens: max_tokens) do
      {:ok, %{response: action, logprobs: logprobs}} ->
        # Get reward from environment (mock for now)
        reward = compute_reward(env, action)

        %{
          observations: [initial_obs],
          actions: [action],
          rewards: [reward],
          dones: [true],
          logprobs: [logprobs]
        }

      {:error, _reason} ->
        # Return empty trajectory on error
        %{
          observations: [initial_obs],
          actions: [""],
          rewards: [0.0],
          dones: [true],
          logprobs: [0.0]
        }
    end
  end

  defp compute_reward(env, _action) do
    # In real implementation, this would evaluate the action
    # For now, return mock reward
    Map.get(env, :mock_reward, :rand.uniform())
  end

  defp mock_rollout(context, env_group, rollout_index) do
    Logger.debug("Using mock rollout (no training_client adapter)")

    group_size = env_group.config.group_size

    # Generate mock trajectories
    trajectories =
      Enum.map(1..group_size, fn i ->
        reward = :rand.uniform()

        %{
          observations: ["Mock observation #{i}"],
          actions: ["Mock action #{i}"],
          rewards: [reward],
          dones: [true],
          logprobs: [-:rand.uniform() * 5]
        }
      end)

    rewards = Enum.flat_map(trajectories, & &1.rewards)
    reward_mean = Enum.sum(rewards) / length(rewards)

    trajectory_group = %{
      trajectories: trajectories,
      rewards: rewards,
      reward_mean: reward_mean,
      reward_std: compute_std(rewards, reward_mean),
      num_trajectories: length(trajectories),
      rollout_index: rollout_index
    }

    rollout_metrics = %{
      duration_ms: 0,
      num_trajectories: length(trajectories),
      total_steps: length(rewards),
      reward_mean: reward_mean,
      reward_std: compute_std(rewards, reward_mean)
    }

    context
    |> Context.put_state(:trajectory_group, trajectory_group)
    |> Context.put_state(:rollout_metrics, rollout_metrics)
    |> then(&{:ok, &1})
  end

  defp compute_std([], _mean), do: 0.0

  defp compute_std(values, mean) do
    variance =
      values
      |> Enum.map(fn v -> (v - mean) * (v - mean) end)
      |> Enum.sum()
      |> Kernel./(length(values))

    :math.sqrt(variance)
  end

  defp emit_telemetry(rollout_index, metrics) do
    :telemetry.execute(
      [:crucible_kitchen, :rl, :rollout_complete],
      %{
        duration_ms: metrics.duration_ms,
        num_trajectories: metrics.num_trajectories,
        total_steps: metrics.total_steps,
        reward_mean: metrics.reward_mean,
        reward_std: metrics.reward_std
      },
      %{rollout_index: rollout_index}
    )
  end
end
