defmodule CrucibleKitchen.Workflows.Preference do
  @moduledoc """
  Direct Preference Optimization (DPO) workflow.

  This workflow implements DPO training with:
  - Preference pair dataset loading
  - Reference model for KL penalty
  - DPO loss computation
  - Preference-based evaluation

  ## Usage

      CrucibleKitchen.run(:preference, %{
        model: "meta-llama/Llama-3.1-8B",
        epochs: 1,
        batch_size: 32,
        learning_rate: 5.0e-7,
        dpo_beta: 0.1
      }, adapters: my_adapters)

  ## Required Adapters

  - `:training_client` - Backend for training operations with DPO support
  - `:dataset_store` - Dataset loading (preference pairs)

  ## Optional Adapters

  - `:blob_store` - Checkpoint storage
  - `:metrics_store` - Metrics persistence

  ## Dataset Format

  Expects preference pairs with:
  - `prompt` - The input prompt
  - `chosen` - The preferred response
  - `rejected` - The rejected response
  """

  use CrucibleKitchen.Workflow

  alias CrucibleKitchen.Stages

  workflow do
    # Setup
    stage(:load_dataset, Stages.LoadDataset)
    stage(:init_session, Stages.InitSession)
    stage(:init_tokenizer, Stages.InitTokenizer)
    stage(:build_preference_dataset, Stages.BuildPreferenceDataset)

    # DPO training loop
    loop :epochs, over: :epochs_range do
      stage(:set_epoch, Stages.SetEpoch)

      loop :pref_batches, over: :pref_batches_range do
        # Get preference batch
        stage(:get_preference_batch, Stages.GetPreferenceBatch)

        # Compute reference model log probs
        stage(:compute_reference_logprobs, Stages.ComputeReferenceLogprobs)

        # DPO forward-backward
        stage(:dpo_forward_backward, Stages.DPOForwardBackward)
        stage(:await_dpo, Stages.AwaitFuture, key: :dpo_future)

        # Optimizer step
        stage(:optim_step, Stages.OptimStep)
        stage(:await_optim, Stages.AwaitFuture, key: :optim_future)

        # Log metrics
        stage(:log_dpo_metrics, Stages.LogDPOMetrics)
      end

      stage(:log_epoch_metrics, Stages.LogEpochMetrics)

      conditional :should_checkpoint? do
        stage(:checkpoint, Stages.SaveCheckpoint)
      end

      conditional :should_evaluate? do
        stage(:evaluate, Stages.Evaluate)
      end
    end

    # Finalize
    stage(:save_final, Stages.SaveFinalWeights)
    stage(:cleanup, Stages.Cleanup)
  end

  @doc false
  def epochs_range(ctx) do
    num_epochs = CrucibleKitchen.Context.get_config(ctx, :epochs, 1)
    0..(num_epochs - 1)
  end

  @doc false
  def pref_batches_range(ctx) do
    dataset = CrucibleKitchen.Context.get_state(ctx, :preference_dataset)
    if dataset, do: 0..(dataset.num_batches - 1), else: []
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
