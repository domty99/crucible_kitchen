defmodule CrucibleKitchen.Stages.CheckTriggers do
  @moduledoc """
  Stage that checks retraining triggers from production feedback.

  Evaluates multiple trigger conditions:
  - **Drift threshold**: Distribution shift in inputs/outputs
  - **Quality drop**: Rolling quality average below threshold
  - **Data count**: Enough curated examples for retraining
  - **Scheduled**: Time-based trigger (e.g., weekly)

  ## Context Requirements

  **Input:**
  - Config: `:deployment_id` - ID of the deployment to check

  **Output:**
  - State: `:triggers` - List of triggered conditions
  - State: `:should_retrain` - Boolean indicating if any triggers fired
  - Metric: `:trigger_count` - Number of triggers that fired

  ## Options

  - `:drift_threshold` - Drift score threshold (default: 0.2)
  - `:quality_threshold` - Quality average threshold (default: 0.7)
  - `:data_count_threshold` - Minimum curated examples (default: 1000)
  - `:trigger_types` - List of trigger types to check (default: all)

  ## Example

      stage(:check_triggers, CheckTriggers)

  The stage will set `:should_retrain` to true if any triggers fire,
  allowing downstream conditional logic:

      conditional fn ctx -> ctx.state.should_retrain end do
        stage(:curate_data, CurateData)
        stage(:train, SupervisedWorkflow)
      end
  """

  use CrucibleKitchen.Stage

  alias CrucibleKitchen.Context

  require Logger

  @impl true
  def name, do: :check_triggers

  @impl true
  def execute(context) do
    deployment_id = get_deployment_id(context)

    case get_adapter(context, :feedback_client) do
      nil ->
        Logger.warning("No feedback_client adapter configured, skipping trigger check")

        context
        |> Context.put_state(:triggers, [])
        |> Context.put_state(:should_retrain, false)
        |> Context.record_metric(:trigger_count, 0)
        |> then(&{:ok, &1})

      {adapter, opts} ->
        check_triggers(context, adapter, opts, deployment_id)
    end
  end

  @impl true
  def validate(context) do
    case get_deployment_id(context) do
      nil -> {:error, "deployment_id is required in config for trigger checking"}
      _ -> :ok
    end
  end

  defp check_triggers(context, adapter, opts, deployment_id) do
    Logger.info("Checking retraining triggers for deployment #{deployment_id}")

    triggers = adapter.check_triggers(opts, deployment_id)
    should_retrain = triggers != []

    if should_retrain do
      trigger_names = Enum.map(triggers, fn {:trigger, name} -> name end)
      Logger.info("Triggers fired: #{inspect(trigger_names)}")
    else
      Logger.info("No triggers fired, retraining not needed")
    end

    emit_telemetry(deployment_id, triggers)

    context
    |> Context.put_state(:triggers, triggers)
    |> Context.put_state(:should_retrain, should_retrain)
    |> Context.record_metric(:trigger_count, length(triggers))
    |> then(&{:ok, &1})
  end

  defp get_deployment_id(context) do
    # Try config first, then state
    context.config[:deployment_id] || Context.get_state(context, :deployment_id)
  end

  defp emit_telemetry(deployment_id, triggers) do
    trigger_types = Enum.map(triggers, fn {:trigger, type} -> type end)

    :telemetry.execute(
      [:crucible_kitchen, :feedback, :triggers_checked],
      %{count: length(triggers)},
      %{
        deployment_id: deployment_id,
        triggers: trigger_types,
        should_retrain: triggers != []
      }
    )
  end
end
