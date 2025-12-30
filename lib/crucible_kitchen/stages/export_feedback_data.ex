defmodule CrucibleKitchen.Stages.ExportFeedbackData do
  @moduledoc """
  Stage that exports curated feedback data for training.

  Exports the curated examples from the CurateData stage to a format
  suitable for training (JSONL, HuggingFace, or Parquet).

  ## Context Requirements

  **Input:**
  - Config: `:deployment_id` - ID of the deployment
  - State: `:curated_examples` - Examples from CurateData stage (optional)

  **Output:**
  - State: `:feedback_data_path` - Path to exported data
  - State: `:feedback_dataset` - Dataset name for training config

  ## Options

  - `:format` - Export format: :jsonl, :huggingface, :parquet (default: :jsonl)
  - `:output_path` - Custom output path (auto-generated if not provided)
  - `:include_exported` - Re-export previously exported data (default: false)

  ## Example

      stage(:curate_data, CurateData)
      stage(:export_feedback, ExportFeedbackData)

  The exported path is stored in state for training:

      # In training config
      dataset: ctx.state.feedback_data_path
  """

  use CrucibleKitchen.Stage

  alias CrucibleKitchen.Context

  require Logger

  @impl true
  def name, do: :export_feedback_data

  @impl true
  def execute(context) do
    deployment_id = get_deployment_id(context)

    case get_adapter(context, :feedback_client) do
      nil ->
        Logger.warning("No feedback_client adapter configured, skipping export")

        context
        |> Context.put_state(:feedback_data_path, nil)
        |> then(&{:ok, &1})

      {adapter, opts} ->
        export_data(context, adapter, opts, deployment_id)
    end
  end

  @impl true
  def validate(context) do
    case get_deployment_id(context) do
      nil -> {:error, "deployment_id is required in config for feedback export"}
      _ -> :ok
    end
  end

  defp export_data(context, adapter, opts, deployment_id) do
    format = Keyword.get(opts, :format, :jsonl)
    Logger.info("Exporting feedback data for #{deployment_id} in #{format} format")

    case adapter.export(opts, deployment_id) do
      {:ok, path} ->
        Logger.info("Exported feedback data to: #{path}")

        emit_telemetry(deployment_id, path, format)

        context
        |> Context.put_state(:feedback_data_path, path)
        |> Context.put_state(:feedback_dataset, path)
        |> then(&{:ok, &1})

      {:error, reason} ->
        Logger.error("Failed to export feedback data: #{inspect(reason)}")
        {:error, {:export_failed, reason}}
    end
  end

  defp get_deployment_id(context) do
    context.config[:deployment_id] || Context.get_state(context, :deployment_id)
  end

  defp emit_telemetry(deployment_id, path, format) do
    :telemetry.execute(
      [:crucible_kitchen, :feedback, :data_exported],
      %{},
      %{
        deployment_id: deployment_id,
        path: path,
        format: format
      }
    )
  end
end
