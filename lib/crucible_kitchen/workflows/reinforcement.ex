defmodule CrucibleKitchen.Workflows.Reinforcement do
  @moduledoc """
  Reinforcement learning workflow with rollouts and PPO.

  This workflow implements RL training with:
  - Environment-based rollout collection
  - Advantage estimation (GAE)
  - PPO optimization with KL penalty
  - Multiple optimization epochs per rollout batch

  ## Usage

      CrucibleKitchen.run(:reinforcement, %{
        model: "meta-llama/Llama-3.1-8B",
        num_rollouts: 100,
        batch_size: 32,
        learning_rate: 1.0e-4,
        gamma: 0.99,
        gae_lambda: 0.95,
        clip_epsilon: 0.2,
        ppo_epochs: 4
      }, adapters: my_adapters)

  ## Required Adapters

  - `:training_client` - Backend for training operations with RL support
  - `:dataset_store` - Dataset loading (prompts for rollouts)

  ## Optional Adapters

  - `:blob_store` - Checkpoint storage
  - `:metrics_store` - Metrics persistence
  """

  use CrucibleKitchen.Workflow

  alias CrucibleKitchen.Stages

  workflow do
    # Setup
    stage(:load_dataset, Stages.LoadDataset)
    stage(:init_session, Stages.InitSession)
    stage(:init_tokenizer, Stages.InitTokenizer)
    stage(:build_env_group, Stages.BuildEnvGroup)

    # RL training loop
    loop :rollouts, over: :rollouts_range do
      # Collect rollouts
      stage(:do_rollout, Stages.DoRollout)

      # Compute advantages
      stage(:compute_advantages, Stages.ComputeAdvantages)

      # Assemble training batch
      stage(:assemble_rl_batch, Stages.AssembleRLBatch)

      # PPO update (multiple epochs on same rollout batch)
      stage(:ppo_update, Stages.PPOUpdate)

      # Log metrics
      stage(:log_rl_metrics, Stages.LogRLMetrics)

      # Conditional checkpoint
      conditional :should_checkpoint? do
        stage(:checkpoint, Stages.SaveCheckpoint)
      end

      # Conditional evaluation
      conditional :should_evaluate? do
        stage(:evaluate, Stages.Evaluate)
      end
    end

    # Finalize
    stage(:save_final, Stages.SaveFinalWeights)
    stage(:cleanup, Stages.Cleanup)
  end

  @doc false
  def rollouts_range(ctx) do
    num_rollouts = CrucibleKitchen.Context.get_config(ctx, :num_rollouts, 100)
    0..(num_rollouts - 1)
  end

  @doc false
  def should_checkpoint?(ctx) do
    save_every = CrucibleKitchen.Context.get_config(ctx, :save_every, 0)
    global_step = CrucibleKitchen.Context.get_state(ctx, :global_step, 0)
    save_every > 0 and rem(global_step + 1, save_every) == 0
  end

  @doc false
  def should_evaluate?(ctx) do
    eval_every = CrucibleKitchen.Context.get_config(ctx, :eval_every, 0)
    global_step = CrucibleKitchen.Context.get_state(ctx, :global_step, 0)
    eval_every > 0 and rem(global_step + 1, eval_every) == 0
  end
end
