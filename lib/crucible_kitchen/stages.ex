defmodule CrucibleKitchen.Stages do
  @moduledoc """
  Provides aliases for all built-in stages.

  This module exists purely for documentation and convenience.
  The actual stage modules are defined in `CrucibleKitchen.Stages.*`.

  ## Available Stages

  ### Setup Stages
  - `CrucibleKitchen.Stages.LoadDataset` - Load dataset from dataset store
  - `CrucibleKitchen.Stages.InitSession` - Initialize training session
  - `CrucibleKitchen.Stages.InitTokenizer` - Get tokenizer from session
  - `CrucibleKitchen.Stages.BuildSupervisedDataset` - Build supervised dataset

  ### Training Stages
  - `CrucibleKitchen.Stages.SetEpoch` - Set current epoch on dataset
  - `CrucibleKitchen.Stages.GetBatch` - Get current batch from dataset
  - `CrucibleKitchen.Stages.ForwardBackward` - Run forward-backward pass
  - `CrucibleKitchen.Stages.OptimStep` - Run optimizer step
  - `CrucibleKitchen.Stages.AwaitFuture` - Await async operation

  ### Logging Stages
  - `CrucibleKitchen.Stages.LogStepMetrics` - Log metrics per step
  - `CrucibleKitchen.Stages.LogEpochMetrics` - Log metrics per epoch

  ### Checkpoint Stages
  - `CrucibleKitchen.Stages.SaveCheckpoint` - Save training checkpoint
  - `CrucibleKitchen.Stages.SaveFinalWeights` - Save final weights
  - `CrucibleKitchen.Stages.Evaluate` - Run evaluation

  ### Cleanup Stages
  - `CrucibleKitchen.Stages.Cleanup` - Clean up resources

  ## Usage in Workflows

      defmodule MyWorkflow do
        use CrucibleKitchen.Workflow

        alias CrucibleKitchen.Stages

        workflow do
          stage :load, Stages.LoadDataset
          stage :init, Stages.InitSession
          # ...
        end
      end
  """

  @doc """
  Returns a list of all available stage modules.
  """
  def all do
    [
      CrucibleKitchen.Stages.LoadDataset,
      CrucibleKitchen.Stages.InitSession,
      CrucibleKitchen.Stages.InitTokenizer,
      CrucibleKitchen.Stages.BuildSupervisedDataset,
      CrucibleKitchen.Stages.SetEpoch,
      CrucibleKitchen.Stages.GetBatch,
      CrucibleKitchen.Stages.ForwardBackward,
      CrucibleKitchen.Stages.OptimStep,
      CrucibleKitchen.Stages.AwaitFuture,
      CrucibleKitchen.Stages.LogStepMetrics,
      CrucibleKitchen.Stages.LogEpochMetrics,
      CrucibleKitchen.Stages.SaveCheckpoint,
      CrucibleKitchen.Stages.SaveFinalWeights,
      CrucibleKitchen.Stages.Evaluate,
      CrucibleKitchen.Stages.Cleanup,
      CrucibleKitchen.Stages.Noop
    ]
  end
end
