defmodule CrucibleKitchen.Stages.CurateData do
  @moduledoc """
  Stage that curates high-value examples from production feedback.

  Selects examples from multiple strategies:
  - **User edits**: Highest priority - manual corrections
  - **High quality**: Positive signals + high quality scores
  - **Hard examples**: Model struggled but response was safe
  - **Diverse**: Spread across the input embedding space

  ## Context Requirements

  **Input:**
  - Config: `:deployment_id` - ID of the deployment to curate from

  **Output:**
  - State: `:curated_examples` - List of curated example maps
  - State: `:curated_count` - Number of examples curated
  - Metric: `:curated_count` - Same as state for telemetry

  ## Options

  - `:limit` - Maximum examples to curate (default: 1000)
  - `:persist` - Whether to persist to storage (default: true)
  - `:min_quality_score` - Min score for high quality (default: 0.8)
  - `:max_hard_quality` - Max score for hard examples (default: 0.5)

  ## Example

      stage(:curate_data, CurateData)

  The curated examples are stored in state for the export stage:

      stage(:curate_data, CurateData)
      stage(:export_feedback, ExportFeedbackData)
  """

  use CrucibleKitchen.Stage

  alias CrucibleKitchen.Context

  require Logger

  @impl true
  def name, do: :curate_data

  @impl true
  def execute(context) do
    deployment_id = get_deployment_id(context)

    case get_adapter(context, :feedback_client) do
      nil ->
        Logger.warning("No feedback_client adapter configured, skipping curation")

        context
        |> Context.put_state(:curated_examples, [])
        |> Context.put_state(:curated_count, 0)
        |> Context.record_metric(:curated_count, 0)
        |> then(&{:ok, &1})

      {adapter, opts} ->
        curate_examples(context, adapter, opts, deployment_id)
    end
  end

  @impl true
  def validate(context) do
    case get_deployment_id(context) do
      nil -> {:error, "deployment_id is required in config for data curation"}
      _ -> :ok
    end
  end

  defp curate_examples(context, adapter, opts, deployment_id) do
    limit = Keyword.get(opts, :limit, 1000)
    Logger.info("Curating up to #{limit} examples from deployment #{deployment_id}")

    case adapter.curate(opts, deployment_id) do
      {:ok, examples} ->
        count = length(examples)
        Logger.info("Curated #{count} examples from feedback data")

        # Group by source for logging
        by_source = Enum.group_by(examples, & &1.curation_source)

        for {source, items} <- by_source do
          Logger.debug("  #{source}: #{length(items)} examples")
        end

        emit_telemetry(deployment_id, examples)

        context
        |> Context.put_state(:curated_examples, examples)
        |> Context.put_state(:curated_count, count)
        |> Context.record_metric(:curated_count, count)
        |> then(&{:ok, &1})

      {:error, reason} ->
        Logger.error("Failed to curate examples: #{inspect(reason)}")
        {:error, {:curation_failed, reason}}
    end
  end

  defp get_deployment_id(context) do
    context.config[:deployment_id] || Context.get_state(context, :deployment_id)
  end

  defp emit_telemetry(deployment_id, examples) do
    by_source =
      examples
      |> Enum.group_by(& &1.curation_source)
      |> Enum.map(fn {source, items} -> {source, length(items)} end)
      |> Map.new()

    :telemetry.execute(
      [:crucible_kitchen, :feedback, :data_curated],
      %{count: length(examples)},
      %{
        deployment_id: deployment_id,
        by_source: by_source
      }
    )
  end
end
