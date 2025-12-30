defmodule CrucibleKitchen.Ports.FeedbackClient do
  @moduledoc """
  Port for production feedback collection and analysis.

  Provides operations for:
  - Checking retraining triggers (drift, quality, data count, schedule)
  - Curating high-value training examples
  - Exporting curated data for training
  - Updating drift baselines after training
  """

  @type opts :: keyword()
  @type deployment_id :: String.t()
  @type trigger :: {:trigger, atom()}
  @type curated_example :: %{
          inference_event_id: String.t(),
          deployment_id: String.t(),
          curation_source: atom(),
          curation_score: float(),
          prompt: String.t(),
          response: String.t()
        }

  @doc """
  Check all retraining triggers for a deployment.

  Returns a list of triggered conditions (drift, quality drop, data count, scheduled).

  ## Options
    - `:drift_score` - Current drift score (for testing)
    - `:quality_average` - Current quality average (for testing)
    - `:event_count` - Current event count (for testing)
    - `:trigger_types` - List of trigger types to check (default: all)
  """
  @callback check_triggers(opts(), deployment_id()) :: [trigger()]

  @doc """
  Curate high-value examples for retraining.

  Selects examples from multiple strategies:
  - User edits (highest priority)
  - High quality examples
  - Hard examples (model struggled)
  - Diverse examples (spread across input space)

  ## Options
    - `:limit` - Maximum number of examples to curate (default: 1000)
    - `:persist` - Whether to persist curated examples (default: true)
  """
  @callback curate(opts(), deployment_id()) :: {:ok, [curated_example()]} | {:error, term()}

  @doc """
  Export curated examples for training.

  ## Options
    - `:format` - Export format: :jsonl, :huggingface, :parquet (default: :jsonl)
    - `:output_path` - Custom output path (auto-generated if not provided)
    - `:include_exported` - Include previously exported examples (default: false)
    - `:limit` - Maximum examples to export
  """
  @callback export(opts(), deployment_id()) :: {:ok, Path.t()} | {:error, term()}

  @doc """
  Export preference pairs from user edits for DPO training.

  Returns pairs of (original_response, edited_response) for preference learning.
  """
  @callback export_preference_pairs(opts(), deployment_id()) ::
              {:ok, Path.t()} | {:error, term()}

  @doc """
  Update the drift baseline after training a new model.

  Should be called after deploying a new model version to reset drift detection.
  """
  @callback update_baseline(opts(), deployment_id()) :: :ok | {:error, term()}

  @doc """
  Get current drift status for a deployment.
  """
  @callback get_drift_status(opts(), deployment_id()) ::
              {:ok, %{input_drift: float(), output_drift: float()}} | {:error, term()}

  @doc """
  Get current quality metrics for a deployment.
  """
  @callback get_quality_metrics(opts(), deployment_id()) ::
              {:ok, %{average: float(), count: integer()}} | {:error, term()}
end
