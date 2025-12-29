defmodule CrucibleKitchen.Workflows.Supervised do
  @moduledoc """
  Standard supervised learning workflow.

  This workflow implements the classic SFT (Supervised Fine-Tuning) pipeline:

  1. Load dataset
  2. Initialize training session
  3. For each epoch:
     a. For each batch:
        - Render messages to tokens
        - Forward-backward pass
        - Optimizer step
        - Log metrics
     b. Maybe checkpoint
     c. Maybe evaluate
  4. Save final weights

  ## Usage

      CrucibleKitchen.run(:supervised, %{
        model: "meta-llama/Llama-3.1-8B",
        epochs: 3,
        batch_size: 128,
        learning_rate: 2.0e-4
      }, adapters: my_adapters)

  ## Required Adapters

  - `:training_client` - Backend for training operations
  - `:dataset_store` - Dataset loading

  ## Optional Adapters

  - `:blob_store` - Checkpoint storage
  - `:metrics_store` - Metrics persistence
  """

  use CrucibleKitchen.Workflow

  alias CrucibleKitchen.Stages.{
    AwaitFuture,
    BuildSupervisedDataset,
    Cleanup,
    Evaluate,
    ForwardBackward,
    GetBatch,
    InitSession,
    InitTokenizer,
    LoadDataset,
    LogEpochMetrics,
    LogStepMetrics,
    OptimStep,
    SaveCheckpoint,
    SaveFinalWeights,
    SetEpoch
  }

  workflow do
    # Setup
    stage(:load_dataset, LoadDataset)
    stage(:init_session, InitSession)
    stage(:init_tokenizer, InitTokenizer)
    stage(:build_dataset, BuildSupervisedDataset)

    # Training loop
    loop :epochs, over: :epochs_range do
      stage(:set_epoch, SetEpoch)

      loop :batches, over: :batches_range do
        stage(:get_batch, GetBatch)
        stage(:forward_backward, ForwardBackward)
        stage(:await_fb, AwaitFuture, key: :fb_future)
        stage(:optim_step, OptimStep)
        stage(:await_optim, AwaitFuture, key: :optim_future)
        stage(:log_step_metrics, LogStepMetrics)
      end

      stage(:log_epoch_metrics, LogEpochMetrics)

      conditional :should_checkpoint? do
        stage(:checkpoint, SaveCheckpoint)
      end

      conditional :should_evaluate? do
        stage(:evaluate, Evaluate)
      end
    end

    # Finalize
    stage(:save_final, SaveFinalWeights)
    stage(:cleanup, Cleanup)
  end

  @doc false
  def epochs_range(ctx) do
    num_epochs = CrucibleKitchen.Context.get_config(ctx, :epochs, 1)
    0..(num_epochs - 1)
  end

  @doc false
  def batches_range(ctx) do
    dataset = CrucibleKitchen.Context.get_state(ctx, :dataset)
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
