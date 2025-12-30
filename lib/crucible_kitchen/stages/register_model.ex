defmodule CrucibleKitchen.Stages.RegisterModel do
  @moduledoc """
  Stage for registering trained models in the model registry.

  This stage collects training artifacts, metrics, and lineage information
  from the context and registers the trained model in the configured
  model registry adapter.

  ## State Requirements

  - `:final_checkpoint` or `:checkpoint_path` - Path to model artifacts
  - `:epoch_metrics` or `:final_metrics` - Final training metrics (optional)
  - `:run_id` - Training run identifier (from metadata)

  ## State Updates

  - `:registered_model` - The registered model record

  ## Telemetry Events

  Emits `[:crucible_kitchen, :model, :registered]` on successful registration.

  ## Usage

  Typically used at the end of a training workflow:

      workflow do
        # ... training stages ...
        stage(:save_final, SaveFinalWeights)
        stage(:evaluate, Evaluate)
        stage(:register_model, RegisterModel)
      end
  """

  use CrucibleKitchen.Stage

  require Logger

  @impl true
  def name, do: :register_model

  @impl true
  def execute(context) do
    case get_adapter(context, :model_registry) do
      nil ->
        Logger.warning(
          "[RegisterModel] No model_registry adapter configured, skipping registration"
        )

        {:ok, context}

      {adapter, opts} ->
        do_register(context, adapter, opts)
    end
  end

  @impl true
  def validate(context) do
    # Check for required state
    checkpoint = get_state(context, :final_checkpoint) || get_state(context, :checkpoint_path)
    model = get_config(context, :model)

    cond do
      is_nil(model) ->
        {:error, {:missing_config, :model}}

      is_nil(checkpoint) ->
        # Registration can still proceed without checkpoint path
        :ok

      true ->
        :ok
    end
  end

  # Perform model registration
  defp do_register(context, adapter, adapter_opts) do
    artifact = build_artifact(context)

    case adapter.register(adapter_opts, artifact) do
      {:ok, model} ->
        emit_registered_event(model, artifact)

        context =
          context
          |> put_state(:registered_model, model)
          |> record_metric(:model_registered, 1)

        Logger.info(
          "[RegisterModel] Model registered: #{model.name} v#{model.version} (id: #{model.id})"
        )

        {:ok, context}

      {:error, {:duplicate_config, existing}} ->
        Logger.info("[RegisterModel] Model already registered with same config: #{existing.id}")
        context = put_state(context, :registered_model, existing)
        {:ok, context}

      {:error, reason} ->
        Logger.error("[RegisterModel] Registration failed: #{inspect(reason)}")
        {:error, {:registration_failed, reason}}
    end
  end

  # Build artifact map from context
  defp build_artifact(context) do
    model_name = get_config(context, :model) || "unknown"
    version = generate_version(context)

    %{
      name: model_name,
      version: version,
      artifact_uri: get_artifact_uri(context),
      metadata: build_metadata(context),
      lineage: build_lineage(context),
      metrics: build_metrics(context)
    }
  end

  # Generate version string
  defp generate_version(context) do
    run_id = CrucibleKitchen.Context.get_metadata(context, :run_id, "unknown")
    timestamp = DateTime.utc_now() |> DateTime.to_iso8601(:basic) |> String.slice(0, 15)
    run_id_suffix = String.slice(run_id, -8, 8)
    run_id_suffix = if run_id_suffix == "", do: "unknown", else: run_id_suffix
    "#{timestamp}-#{run_id_suffix}"
  end

  # Get artifact URI from context state
  defp get_artifact_uri(context) do
    get_state(context, :final_checkpoint) ||
      get_state(context, :checkpoint_path) ||
      get_state(context, :weights_path) ||
      "local://unknown"
  end

  # Build metadata from context
  defp build_metadata(context) do
    training_config =
      [:model, :dataset, :epochs, :learning_rate, :batch_size, :lora_rank, :max_seq_len]
      |> Enum.reduce(%{}, fn key, acc ->
        case get_config(context, key) do
          nil -> acc
          value -> Map.put(acc, key, value)
        end
      end)

    %{
      training_config: training_config,
      final_metrics:
        get_state(context, :epoch_metrics) || get_state(context, :final_metrics) || %{},
      total_steps: get_state(context, :global_step, 0),
      completed_epochs: get_state(context, :current_epoch, 0) + 1
    }
  end

  # Build lineage information
  defp build_lineage(context) do
    started_at = CrucibleKitchen.Context.get_metadata(context, :started_at)
    run_id = CrucibleKitchen.Context.get_metadata(context, :run_id)
    executed_stages = get_state(context, :executed_stages, [])

    %{
      dataset: get_config(context, :dataset),
      workflow: get_state(context, :workflow_name) || :supervised,
      stages: executed_stages,
      run_id: run_id,
      started_at: started_at,
      completed_at: DateTime.utc_now()
    }
  end

  # Build final metrics
  defp build_metrics(context) do
    epoch_metrics = get_state(context, :epoch_metrics, %{})
    eval_results = get_state(context, :eval_results, %{})

    epoch_metrics
    |> Map.merge(eval_results)
    |> Map.drop([:step, :evaluated, :sample_count])
  end

  # Emit telemetry event for model registration
  defp emit_registered_event(model, artifact) do
    :telemetry.execute(
      [:crucible_kitchen, :model, :registered],
      %{count: 1},
      %{
        model_id: model.id,
        name: artifact.name,
        version: artifact.version
      }
    )
  end
end
