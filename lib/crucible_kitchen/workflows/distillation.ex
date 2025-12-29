defmodule CrucibleKitchen.Workflows.Distillation do
  @moduledoc """
  Knowledge distillation workflow.

  This workflow implements distillation training with:
  - Teacher model inference
  - Student model training with KL loss
  - On-policy mode (generating from teacher)
  - Temperature-scaled soft labels

  ## Usage

      CrucibleKitchen.run(:distillation, %{
        teacher_model: "meta-llama/Llama-3.1-70B",
        model: "meta-llama/Llama-3.1-8B",  # student
        epochs: 3,
        batch_size: 32,
        learning_rate: 2.0e-4,
        distillation_temperature: 2.0,
        distillation_alpha: 0.5  # weight between KL and CE loss
      }, adapters: my_adapters)

  ## Required Adapters

  - `:training_client` - Backend for training operations with distillation support
  - `:dataset_store` - Dataset loading

  ## Optional Adapters

  - `:blob_store` - Checkpoint storage
  - `:metrics_store` - Metrics persistence

  ## Loss Computation

  Total loss = alpha * KL(student || teacher) + (1 - alpha) * CE(student, labels)

  Where:
  - KL loss uses temperature-scaled softmax
  - CE loss uses hard labels (ground truth)
  """

  use CrucibleKitchen.Workflow

  alias CrucibleKitchen.Stages

  workflow do
    # Setup
    stage(:load_dataset, Stages.LoadDataset)
    stage(:init_session, Stages.InitSession)
    stage(:init_tokenizer, Stages.InitTokenizer)
    stage(:init_teacher, Stages.InitTeacher)
    stage(:build_distillation_dataset, Stages.BuildDistillationDataset)

    # Distillation training loop
    loop :epochs, over: :epochs_range do
      stage(:set_epoch, Stages.SetEpoch)

      loop :dist_batches, over: :dist_batches_range do
        # Get batch
        stage(:get_distillation_batch, Stages.GetDistillationBatch)

        # Teacher inference (get soft labels)
        stage(:teacher_inference, Stages.TeacherInference)

        # Student forward-backward with distillation loss
        stage(:distillation_forward_backward, Stages.DistillationForwardBackward)
        stage(:await_distillation, Stages.AwaitFuture, key: :distillation_future)

        # Optimizer step
        stage(:optim_step, Stages.OptimStep)
        stage(:await_optim, Stages.AwaitFuture, key: :optim_future)

        # Log metrics
        stage(:log_distillation_metrics, Stages.LogDistillationMetrics)
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
    stage(:cleanup_teacher, Stages.CleanupTeacher)
    stage(:cleanup, Stages.Cleanup)
  end

  @doc false
  def epochs_range(ctx) do
    num_epochs = CrucibleKitchen.Context.get_config(ctx, :epochs, 1)
    0..(num_epochs - 1)
  end

  @doc false
  def dist_batches_range(ctx) do
    dataset = CrucibleKitchen.Context.get_state(ctx, :distillation_dataset)
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
