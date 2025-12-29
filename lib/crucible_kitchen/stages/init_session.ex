defmodule CrucibleKitchen.Stages.InitSession do
  @moduledoc """
  Stage for initializing a training session.

  Creates a training session using the training_client adapter.
  Stores the session in context state as `:session`.

  ## Configuration

  - `:model` - Model identifier (required)
  - `:lora_rank` - LoRA rank for fine-tuning (default: 16)
  - `:learning_rate` - Base learning rate (default: 2.0e-5)
  """

  use CrucibleKitchen.Stage

  alias CrucibleTrain.Ports.TrainingClient

  require Logger

  @impl true
  def name, do: :init_session

  @impl true
  def validate(context) do
    model = get_config(context, :model)

    if is_nil(model) or model == "" do
      {:error, "model configuration is required"}
    else
      :ok
    end
  end

  @impl true
  def execute(context) do
    model = get_config(context, :model)
    lora_rank = get_config(context, :lora_rank, 16)
    learning_rate = get_config(context, :learning_rate, 2.0e-5)

    Logger.info("[InitSession] Starting session for model: #{model}")
    Logger.debug("[InitSession] LoRA rank: #{lora_rank}, LR: #{learning_rate}")

    ports = get_train_ports(context)

    session_config = %{
      model: model,
      lora_rank: lora_rank,
      learning_rate: learning_rate
    }

    case TrainingClient.start_session(ports, session_config) do
      {:ok, session} ->
        Logger.info("[InitSession] Session started successfully")

        context =
          context
          |> put_state(:session, session)
          |> put_state(:global_step, 0)
          |> record_metric(:session_started, 1)

        {:ok, context}

      {:error, reason} ->
        Logger.error("[InitSession] Failed to start session: #{inspect(reason)}")
        {:error, {:session_start_failed, reason}}
    end
  end

  @impl true
  def rollback(context, _error) do
    case get_state(context, :session) do
      nil ->
        context

      session ->
        ports = get_train_ports(context)
        TrainingClient.close_session(ports, session)
        put_state(context, :session, nil)
    end
  end
end
