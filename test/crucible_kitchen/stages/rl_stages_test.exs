defmodule CrucibleKitchen.Stages.RLStagesTest do
  use ExUnit.Case, async: true

  alias CrucibleKitchen.Context

  alias CrucibleKitchen.Stages.{
    AssembleRLBatch,
    BuildEnvGroup,
    ComputeAdvantages,
    DoRollout,
    LogRLMetrics,
    PPOUpdate
  }

  describe "BuildEnvGroup" do
    test "name returns :build_env_group" do
      assert BuildEnvGroup.name() == :build_env_group
    end

    test "builds noop env group" do
      context =
        build_context(%{
          env: :noop,
          group_size: 4,
          groups_per_batch: 10
        })

      assert {:ok, result} = BuildEnvGroup.execute(context)
      assert result.state.env_group != nil
      assert result.state.env_config.group_size == 4
    end

    test "emits telemetry event" do
      :telemetry.attach(
        "build-env-group-test",
        [:crucible_kitchen, :rl, :env_group_built],
        fn event, measurements, metadata, _ ->
          send(self(), {:telemetry, event, measurements, metadata})
        end,
        nil
      )

      context = build_context(%{env: :noop, group_size: 2, groups_per_batch: 5})
      {:ok, _} = BuildEnvGroup.execute(context)

      assert_receive {:telemetry, [:crucible_kitchen, :rl, :env_group_built], measurements, _}
      assert measurements.group_size == 2

      :telemetry.detach("build-env-group-test")
    end
  end

  describe "DoRollout" do
    test "name returns :do_rollout" do
      assert DoRollout.name() == :do_rollout
    end

    test "performs mock rollout when no adapter" do
      env_group = %{
        type: :noop,
        config: %{group_size: 2},
        make_envs: fn _opts ->
          [
            %{id: "env_1", observation: "Obs 1", done: false},
            %{id: "env_2", observation: "Obs 2", done: false}
          ]
        end
      }

      context =
        build_context_without_training(%{})
        |> Context.put_state(:env_group, env_group)
        |> Context.put_state(:rollouts_index, 0)

      assert {:ok, result} = DoRollout.execute(context)
      assert result.state.trajectory_group != nil
      assert result.state.trajectory_group.num_trajectories == 2
      assert result.state.rollout_metrics != nil
    end

    test "validation requires env_group" do
      context = build_context(%{})
      assert {:error, _} = DoRollout.validate(context)
    end
  end

  describe "ComputeAdvantages" do
    test "name returns :compute_advantages" do
      assert ComputeAdvantages.name() == :compute_advantages
    end

    test "computes advantages from trajectory group" do
      trajectory_group = %{
        trajectories: [
          %{rewards: [1.0]},
          %{rewards: [0.5]},
          %{rewards: [0.0]}
        ],
        num_trajectories: 3
      }

      context =
        build_context(%{gamma: 0.99, gae_lambda: 0.95})
        |> Context.put_state(:trajectory_group, trajectory_group)

      assert {:ok, result} = ComputeAdvantages.execute(context)
      assert length(result.state.advantages) == 3
      assert length(result.state.returns) == 3
      assert length(result.state.normalized_advantages) == 3
    end

    test "normalizes advantages to zero mean" do
      trajectory_group = %{
        trajectories: [
          %{rewards: [10.0]},
          %{rewards: [5.0]},
          %{rewards: [0.0]}
        ],
        num_trajectories: 3
      }

      context =
        build_context(%{gamma: 1.0, gae_lambda: 1.0})
        |> Context.put_state(:trajectory_group, trajectory_group)

      {:ok, result} = ComputeAdvantages.execute(context)

      # Normalized advantages should have mean close to 0
      mean = Enum.sum(result.state.normalized_advantages) / 3
      assert abs(mean) < 0.01
    end

    test "validation requires trajectory_group" do
      context = build_context(%{})
      assert {:error, _} = ComputeAdvantages.validate(context)
    end
  end

  describe "AssembleRLBatch" do
    test "name returns :assemble_rl_batch" do
      assert AssembleRLBatch.name() == :assemble_rl_batch
    end

    test "assembles batch from trajectories" do
      trajectory_group = %{
        trajectories: [
          %{observations: ["O1"], actions: ["A1"], logprobs: [-1.0]},
          %{observations: ["O2"], actions: ["A2"], logprobs: [-2.0]}
        ],
        num_trajectories: 2
      }

      context =
        build_context(%{})
        |> Context.put_state(:trajectory_group, trajectory_group)
        |> Context.put_state(:normalized_advantages, [0.5, -0.5])
        |> Context.put_state(:returns, [1.0, 0.5])

      assert {:ok, result} = AssembleRLBatch.execute(context)
      assert result.state.rl_batch.batch_size == 2
      assert result.state.rl_batch.observations == ["O1", "O2"]
      assert result.state.rl_batch.actions == ["A1", "A2"]
    end
  end

  describe "PPOUpdate" do
    test "name returns :ppo_update" do
      assert PPOUpdate.name() == :ppo_update
    end

    test "performs mock PPO update when no adapter" do
      batch = %{
        observations: ["O1", "O2"],
        actions: ["A1", "A2"],
        old_logprobs: [-1.0, -2.0],
        advantages: [0.5, -0.5],
        returns: [1.0, 0.5],
        batch_size: 2
      }

      context =
        build_context_without_training(%{
          clip_epsilon: 0.2,
          ppo_epochs: 4,
          kl_penalty_coef: 0.0
        })
        |> Context.put_state(:rl_batch, batch)

      assert {:ok, result} = PPOUpdate.execute(context)
      assert result.state.ppo_metrics != nil
      assert Map.has_key?(result.state.ppo_metrics, :policy_loss)
    end

    test "validation requires rl_batch" do
      context = build_context(%{})
      assert {:error, _} = PPOUpdate.validate(context)
    end
  end

  describe "LogRLMetrics" do
    test "name returns :log_rl_metrics" do
      assert LogRLMetrics.name() == :log_rl_metrics
    end

    test "logs metrics and increments step" do
      context =
        build_context(%{})
        |> Context.put_state(:rollout_metrics, %{
          reward_mean: 0.75,
          reward_std: 0.1,
          num_trajectories: 10
        })
        |> Context.put_state(:ppo_metrics, %{
          policy_loss: 0.1,
          entropy: 1.5,
          clip_fraction: 0.15
        })
        |> Context.put_state(:global_step, 5)

      assert {:ok, result} = LogRLMetrics.execute(context)
      assert result.state.global_step == 6

      # Check metrics recorded
      assert Enum.any?(result.metrics, &(&1.name == :rl_reward_mean))
      assert Enum.any?(result.metrics, &(&1.name == :rl_policy_loss))
    end

    test "emits telemetry event" do
      :telemetry.attach(
        "log-rl-metrics-test",
        [:crucible_kitchen, :rl, :step],
        fn event, measurements, metadata, _ ->
          send(self(), {:telemetry, event, measurements, metadata})
        end,
        nil
      )

      context =
        build_context(%{})
        |> Context.put_state(:rollout_metrics, %{reward_mean: 0.5})
        |> Context.put_state(:ppo_metrics, %{policy_loss: 0.2})
        |> Context.put_state(:global_step, 10)

      {:ok, _} = LogRLMetrics.execute(context)

      assert_receive {:telemetry, [:crucible_kitchen, :rl, :step], measurements, _}
      assert measurements.reward_mean == 0.5

      :telemetry.detach("log-rl-metrics-test")
    end
  end

  defp build_context(extra_config) do
    config = Map.merge(%{}, extra_config)

    Context.new(
      config,
      %{
        training_client: CrucibleKitchen.Adapters.Noop.TrainingClient,
        dataset_store: CrucibleKitchen.Adapters.Noop.DatasetStore
      }
    )
  end

  defp build_context_without_training(extra_config) do
    config = Map.merge(%{}, extra_config)

    Context.new(
      config,
      %{
        dataset_store: CrucibleKitchen.Adapters.Noop.DatasetStore
      }
    )
  end
end
