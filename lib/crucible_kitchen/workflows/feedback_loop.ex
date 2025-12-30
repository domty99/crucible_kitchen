defmodule CrucibleKitchen.Workflows.FeedbackLoop do
  @moduledoc """
  Feedback-driven retraining workflow.

  This workflow implements the production feedback loop pipeline:

  1. Check retraining triggers (drift, quality drop, data count)
  2. If triggered:
     a. Curate high-value examples from production
     b. Export curated data for training
     c. Run supervised training on feedback data
     d. Register the new model
     e. Update drift baseline

  ## Usage

      CrucibleKitchen.run(:feedback_loop, %{
        deployment_id: "my-deployment",
        model: "meta-llama/Llama-3.1-8B",
        epochs: 1,
        batch_size: 64
      }, adapters: my_adapters)

  ## Required Adapters

  - `:feedback_client` - Production feedback system integration
  - `:training_client` - Backend for training operations
  - `:dataset_store` - Dataset loading (for feedback data)

  ## Optional Adapters

  - `:model_registry` - Model versioning and registration
  - `:evaluator` - Model evaluation
  - `:blob_store` - Checkpoint storage
  - `:metrics_store` - Metrics persistence

  ## Trigger Types

  The workflow checks these trigger conditions:
  - `:drift_threshold` - Input/output distribution shift detected
  - `:quality_drop` - Rolling quality average below threshold
  - `:data_count` - Enough curated examples accumulated
  - `:scheduled` - Time-based trigger (e.g., weekly)

  ## Config Options

  | Option | Type | Description |
  |--------|------|-------------|
  | `:deployment_id` | string | **Required.** Production deployment to monitor |
  | `:drift_threshold` | float | Drift score threshold (default: 0.2) |
  | `:quality_threshold` | float | Quality average threshold (default: 0.7) |
  | `:data_count_threshold` | int | Minimum curated examples (default: 1000) |
  | `:curate_limit` | int | Max examples to curate (default: 1000) |
  | `:export_format` | atom | Format: :jsonl, :huggingface, :parquet |

  ## Example: Scheduled Retraining Pipeline

      # In your scheduler (Quantum, Oban, etc.)
      def perform_scheduled_retrain do
        CrucibleKitchen.run(:feedback_loop, %{
          deployment_id: "prod-chat-v2",
          model: "my-org/chat-model",
          epochs: 1,
          trigger_types: [:data_count, :drift_threshold]
        }, adapters: prod_adapters())
      end
  """

  use CrucibleKitchen.Workflow

  # Feedback stages: CheckTriggers, CurateData, ExportFeedbackData, UpdateBaseline
  # Training stages (reused from supervised): AwaitFuture, BuildSupervisedDataset, etc.
  alias CrucibleKitchen.Stages.{
    AwaitFuture,
    BuildSupervisedDataset,
    CheckTriggers,
    Cleanup,
    CurateData,
    Evaluate,
    ExportFeedbackData,
    ForwardBackward,
    GetBatch,
    InitSession,
    InitTokenizer,
    LogEpochMetrics,
    LogStepMetrics,
    OptimStep,
    RegisterModel,
    SaveFinalWeights,
    SetEpoch,
    UpdateBaseline
  }

  workflow do
    # Phase 1: Check if retraining is needed
    stage(:check_triggers, CheckTriggers)

    # Phase 2: Only retrain if triggers fired
    conditional :should_retrain? do
      # Curate and export feedback data
      stage(:curate_data, CurateData)
      stage(:export_feedback, ExportFeedbackData)

      # Setup training using feedback data as dataset
      stage(:init_session, InitSession)
      stage(:init_tokenizer, InitTokenizer)
      stage(:build_dataset, BuildSupervisedDataset)

      # Training loop (same as supervised workflow)
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
      end

      # Finalize
      stage(:save_final, SaveFinalWeights)

      # Evaluation and registration
      stage(:final_evaluate, Evaluate)
      stage(:register_model, RegisterModel)

      # Reset drift baseline for new model
      stage(:update_baseline, UpdateBaseline)

      # Cleanup
      stage(:cleanup, Cleanup)
    end
  end

  @doc false
  def should_retrain?(ctx) do
    CrucibleKitchen.Context.get_state(ctx, :should_retrain, false)
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
end
