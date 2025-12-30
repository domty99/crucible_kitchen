defmodule CrucibleKitchen.Stages.UpdateBaseline do
  @moduledoc """
  Stage that updates the drift baseline after training a new model.

  After training and deploying a new model version, this stage resets
  the drift detection baseline so future drift is measured against
  the new model's behavior.

  ## Context Requirements

  **Input:**
  - Config: `:deployment_id` - ID of the deployment
  - State: `:registered_model` - The newly registered model (optional, for logging)

  **Output:**
  - State: `:baseline_updated` - Boolean indicating success

  ## When to Use

  This stage should be called after:
  1. Training completes successfully
  2. Model is registered in the registry
  3. (Optionally) Model is deployed to production

  ## Example

      stage(:train, SupervisedWorkflow)
      stage(:register, RegisterModel)
      stage(:update_baseline, UpdateBaseline)

  This ensures the feedback loop starts fresh for the new model version.
  """

  use CrucibleKitchen.Stage

  alias CrucibleKitchen.Context

  require Logger

  @impl true
  def name, do: :update_baseline

  @impl true
  def execute(context) do
    deployment_id = get_deployment_id(context)

    case get_adapter(context, :feedback_client) do
      nil ->
        Logger.warning("No feedback_client adapter configured, skipping baseline update")

        context
        |> Context.put_state(:baseline_updated, false)
        |> then(&{:ok, &1})

      {adapter, opts} ->
        update_baseline(context, adapter, opts, deployment_id)
    end
  end

  @impl true
  def validate(context) do
    case get_deployment_id(context) do
      nil -> {:error, "deployment_id is required in config for baseline update"}
      _ -> :ok
    end
  end

  defp update_baseline(context, adapter, opts, deployment_id) do
    model_info = Context.get_state(context, :registered_model)

    if model_info do
      Logger.info(
        "Updating drift baseline for deployment #{deployment_id} " <>
          "after registering model #{model_info.name} v#{model_info.version}"
      )
    else
      Logger.info("Updating drift baseline for deployment #{deployment_id}")
    end

    case adapter.update_baseline(opts, deployment_id) do
      :ok ->
        Logger.info("Drift baseline updated successfully")

        emit_telemetry(deployment_id, model_info)

        context
        |> Context.put_state(:baseline_updated, true)
        |> then(&{:ok, &1})

      {:error, reason} ->
        Logger.error("Failed to update baseline: #{inspect(reason)}")
        # Don't fail the workflow for baseline update failures
        Logger.warning("Continuing despite baseline update failure")

        context
        |> Context.put_state(:baseline_updated, false)
        |> then(&{:ok, &1})
    end
  end

  defp get_deployment_id(context) do
    context.config[:deployment_id] || Context.get_state(context, :deployment_id)
  end

  defp emit_telemetry(deployment_id, model_info) do
    metadata =
      %{deployment_id: deployment_id}
      |> maybe_add_model_info(model_info)

    :telemetry.execute(
      [:crucible_kitchen, :feedback, :baseline_updated],
      %{},
      metadata
    )
  end

  defp maybe_add_model_info(metadata, nil), do: metadata

  defp maybe_add_model_info(metadata, model) do
    Map.merge(metadata, %{
      model_name: model.name,
      model_version: model.version,
      model_id: model.id
    })
  end
end
