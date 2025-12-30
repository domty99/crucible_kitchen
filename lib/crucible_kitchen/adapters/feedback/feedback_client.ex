defmodule CrucibleKitchen.Adapters.Feedback.FeedbackClient do
  @moduledoc """
  CrucibleFeedback adapter for production feedback integration.

  Wraps the CrucibleFeedback package to provide:
  - Trigger checking (drift, quality, count, schedule)
  - Data curation (high quality, hard examples, diverse)
  - Export for training (JSONL, HuggingFace, Parquet)
  - Baseline updates after retraining
  """

  @behaviour CrucibleKitchen.Ports.FeedbackClient

  require Logger

  @impl true
  def check_triggers(opts, deployment_id) do
    if Code.ensure_loaded?(CrucibleFeedback) do
      CrucibleFeedback.check_triggers(deployment_id, opts)
    else
      Logger.warning("CrucibleFeedback not available, returning empty triggers")
      []
    end
  end

  @impl true
  def curate(opts, deployment_id) do
    if Code.ensure_loaded?(CrucibleFeedback) do
      examples = CrucibleFeedback.curate(deployment_id, opts)
      {:ok, examples}
    else
      Logger.warning("CrucibleFeedback not available, returning empty curation")
      {:ok, []}
    end
  end

  @impl true
  def export(opts, deployment_id) do
    if Code.ensure_loaded?(CrucibleFeedback) do
      CrucibleFeedback.export(deployment_id, opts)
    else
      Logger.warning("CrucibleFeedback not available, cannot export")
      {:error, :crucible_feedback_not_available}
    end
  end

  @impl true
  def export_preference_pairs(opts, deployment_id) do
    if Code.ensure_loaded?(CrucibleFeedback) do
      CrucibleFeedback.export_preference_pairs(deployment_id, opts)
    else
      Logger.warning("CrucibleFeedback not available, cannot export preference pairs")
      {:error, :crucible_feedback_not_available}
    end
  end

  @impl true
  def update_baseline(_opts, deployment_id) do
    # CrucibleFeedback uses sliding window drift detection, so there's no explicit
    # baseline to update. The baseline is automatically the older window of events.
    Logger.debug("[FeedbackClient] Baseline update acknowledged for deployment: #{deployment_id}")
    :ok
  end

  @impl true
  def get_drift_status(opts, deployment_id) do
    if Code.ensure_loaded?(CrucibleFeedback) do
      drift_results = CrucibleFeedback.detect_drift(deployment_id, opts)

      # Extract drift scores by type from the results
      input_drift =
        Enum.find_value(drift_results, 0.0, fn
          %{type: :statistical, score: score} -> score
          %{type: :embedding, score: score} -> score
          _ -> nil
        end)

      output_drift =
        Enum.find_value(drift_results, 0.0, fn
          %{type: :output, score: score} -> score
          _ -> nil
        end)

      {:ok, %{input_drift: input_drift, output_drift: output_drift}}
    else
      {:ok, %{input_drift: 0.0, output_drift: 0.0}}
    end
  end

  @impl true
  def get_quality_metrics(_opts, deployment_id) do
    if Code.ensure_loaded?(CrucibleFeedback) do
      stats = CrucibleFeedback.get_quality_stats(deployment_id)
      {:ok, %{average: stats.rolling_average, count: 0}}
    else
      {:ok, %{average: 1.0, count: 0}}
    end
  end
end
