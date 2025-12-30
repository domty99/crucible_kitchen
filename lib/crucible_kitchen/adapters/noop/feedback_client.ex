defmodule CrucibleKitchen.Adapters.Noop.FeedbackClient do
  @moduledoc """
  No-op feedback client for testing feedback loop workflows.

  Provides configurable trigger responses and mock data for testing
  the feedback-driven retraining workflow without real production data.

  ## Options

    - `:triggers` - List of triggers to return (default: [])
    - `:drift_score` - Mock drift score (default: 0.1)
    - `:quality_average` - Mock quality average (default: 0.85)
    - `:curated_count` - Number of mock curated examples (default: 100)
    - `:export_path` - Mock export path (default: "/tmp/noop_feedback_export.jsonl")
  """

  @behaviour CrucibleKitchen.Ports.FeedbackClient

  @impl true
  def check_triggers(opts, deployment_id) do
    # Return configured triggers or check against thresholds
    case Keyword.get(opts, :triggers) do
      triggers when is_list(triggers) ->
        triggers

      nil ->
        # Simulate threshold checking
        drift_score = Keyword.get(opts, :drift_score, 0.1)
        quality_avg = Keyword.get(opts, :quality_average, 0.85)
        event_count = Keyword.get(opts, :event_count, 500)

        drift_threshold = Keyword.get(opts, :drift_threshold, 0.2)
        quality_threshold = Keyword.get(opts, :quality_threshold, 0.7)
        count_threshold = Keyword.get(opts, :data_count_threshold, 1000)

        triggers = []

        triggers =
          if drift_score > drift_threshold,
            do: [{:trigger, :drift_threshold} | triggers],
            else: triggers

        triggers =
          if quality_avg < quality_threshold,
            do: [{:trigger, :quality_drop} | triggers],
            else: triggers

        triggers =
          if event_count >= count_threshold,
            do: [{:trigger, :data_count} | triggers],
            else: triggers

        emit_telemetry(:check_triggers, %{
          deployment_id: deployment_id,
          trigger_count: length(triggers)
        })

        triggers
    end
  end

  @impl true
  def curate(opts, deployment_id) do
    count = Keyword.get(opts, :curated_count, 100)

    examples =
      Enum.map(1..count, fn i ->
        source =
          Enum.random([:user_edit, :high_quality, :hard_example, :diverse])

        %{
          inference_event_id: "noop_event_#{deployment_id}_#{i}",
          deployment_id: deployment_id,
          curation_source: source,
          curation_score: :rand.uniform() * 10,
          prompt: "Mock prompt #{i} for deployment #{deployment_id}",
          response: "Mock response #{i} with useful content"
        }
      end)

    emit_telemetry(:curate, %{deployment_id: deployment_id, count: count})

    {:ok, examples}
  end

  @impl true
  def export(opts, deployment_id) do
    format = Keyword.get(opts, :format, :jsonl)
    base_path = Keyword.get(opts, :export_path, "/tmp/noop_feedback_export")

    path =
      case format do
        :jsonl -> "#{base_path}_#{deployment_id}.jsonl"
        :huggingface -> "#{base_path}_#{deployment_id}_hf"
        :parquet -> "#{base_path}_#{deployment_id}.parquet"
      end

    emit_telemetry(:export, %{deployment_id: deployment_id, format: format, path: path})

    {:ok, path}
  end

  @impl true
  def export_preference_pairs(opts, deployment_id) do
    base_path = Keyword.get(opts, :export_path, "/tmp/noop_preference_pairs")
    path = "#{base_path}_#{deployment_id}.jsonl"

    emit_telemetry(:export_preference_pairs, %{deployment_id: deployment_id, path: path})

    {:ok, path}
  end

  @impl true
  def update_baseline(_opts, deployment_id) do
    emit_telemetry(:update_baseline, %{deployment_id: deployment_id})
    :ok
  end

  @impl true
  def get_drift_status(opts, _deployment_id) do
    {:ok,
     %{
       input_drift: Keyword.get(opts, :drift_score, 0.1),
       output_drift: Keyword.get(opts, :drift_score, 0.1) * 0.8
     }}
  end

  @impl true
  def get_quality_metrics(opts, _deployment_id) do
    {:ok,
     %{
       average: Keyword.get(opts, :quality_average, 0.85),
       count: Keyword.get(opts, :event_count, 500)
     }}
  end

  defp emit_telemetry(action, metadata) do
    :telemetry.execute(
      [:crucible_kitchen, :feedback, action],
      %{},
      metadata
    )
  end
end
