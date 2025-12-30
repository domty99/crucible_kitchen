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

  ### Evaluation & Registration Stages
  - `CrucibleKitchen.Stages.Evaluate` - Run model evaluation
  - `CrucibleKitchen.Stages.RegisterModel` - Register trained model in registry

  ### Feedback Loop Stages
  - `CrucibleKitchen.Stages.CheckTriggers` - Check retraining triggers from production
  - `CrucibleKitchen.Stages.CurateData` - Curate high-value examples from feedback
  - `CrucibleKitchen.Stages.ExportFeedbackData` - Export curated data for training
  - `CrucibleKitchen.Stages.UpdateBaseline` - Update drift baseline after training

  ### DPO (Preference) Stages
  - `CrucibleKitchen.Stages.BuildPreferenceDataset` - Build preference dataset from comparisons
  - `CrucibleKitchen.Stages.GetPreferenceBatch` - Get next preference batch
  - `CrucibleKitchen.Stages.ComputeReferenceLogprobs` - Compute reference model logprobs
  - `CrucibleKitchen.Stages.DPOForwardBackward` - DPO loss computation
  - `CrucibleKitchen.Stages.LogDPOMetrics` - Log DPO-specific metrics

  ### RL (Reinforcement Learning) Stages
  - `CrucibleKitchen.Stages.BuildEnvGroup` - Build environment group for rollouts
  - `CrucibleKitchen.Stages.DoRollout` - Collect trajectories via rollouts
  - `CrucibleKitchen.Stages.ComputeAdvantages` - Compute GAE advantages
  - `CrucibleKitchen.Stages.AssembleRLBatch` - Assemble trajectories into training batch
  - `CrucibleKitchen.Stages.PPOUpdate` - PPO policy gradient update
  - `CrucibleKitchen.Stages.LogRLMetrics` - Log RL-specific metrics

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
      CrucibleKitchen.Stages.RegisterModel,
      CrucibleKitchen.Stages.Cleanup,
      CrucibleKitchen.Stages.Noop,
      # Feedback loop stages
      CrucibleKitchen.Stages.CheckTriggers,
      CrucibleKitchen.Stages.CurateData,
      CrucibleKitchen.Stages.ExportFeedbackData,
      CrucibleKitchen.Stages.UpdateBaseline,
      # DPO (Preference) stages
      CrucibleKitchen.Stages.BuildPreferenceDataset,
      CrucibleKitchen.Stages.GetPreferenceBatch,
      CrucibleKitchen.Stages.ComputeReferenceLogprobs,
      CrucibleKitchen.Stages.DPOForwardBackward,
      CrucibleKitchen.Stages.LogDPOMetrics,
      # RL stages
      CrucibleKitchen.Stages.BuildEnvGroup,
      CrucibleKitchen.Stages.DoRollout,
      CrucibleKitchen.Stages.ComputeAdvantages,
      CrucibleKitchen.Stages.AssembleRLBatch,
      CrucibleKitchen.Stages.PPOUpdate,
      CrucibleKitchen.Stages.LogRLMetrics
    ]
  end
end
