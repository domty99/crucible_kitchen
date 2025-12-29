defmodule CrucibleKitchen.Adapters.Tinkex.TrainingClient do
  @moduledoc """
  TrainingClient adapter for the Tinker ML platform.

  Implements CrucibleTrain.Ports.TrainingClient using the Tinkex API client.
  This is the primary adapter for running training workloads on Tinker.

  ## Configuration

  Requires the following environment variables:
  - `TINKER_API_KEY` - Your Tinker API key
  - `TINKER_BASE_URL` - Optional. Tinker API base URL.

  ## Adapter Options

  - `:api_key` - Override API key (default: from TINKER_API_KEY env)
  - `:base_url` - Override base URL (default: from TINKER_BASE_URL env)

  ## Usage

      config :my_app,
        adapters: %{
          training_client: {CrucibleKitchen.Adapters.Tinkex.TrainingClient, [
            api_key: System.get_env("TINKER_API_KEY")
          ]}
        }
  """

  @behaviour CrucibleTrain.Ports.TrainingClient

  alias CrucibleTrain.Types.{Datum, EncodedTextChunk, ModelInput, TensorData}

  require Logger

  @default_base_url "https://tinker.thinkingmachines.dev/services/tinker-prod"

  @type session :: %{
          id: pid(),
          service_client: pid(),
          model: String.t(),
          config: map()
        }

  # ============================================================================
  # Required Callbacks
  # ============================================================================

  @impl true
  @spec start_session(keyword(), map()) :: {:ok, session()} | {:error, term()}
  def start_session(adapter_opts, config) do
    api_key = Keyword.get(adapter_opts, :api_key) || System.get_env("TINKER_API_KEY")

    base_url =
      Keyword.get(adapter_opts, :base_url) || System.get_env("TINKER_BASE_URL", @default_base_url)

    model = Map.fetch!(config, :model)

    if api_key do
      do_start_session(model, config, api_key, base_url)
    else
      {:error, {:missing_env, "TINKER_API_KEY"}}
    end
  end

  @impl true
  @spec forward_backward(keyword(), session(), [Datum.t()]) :: Task.t()
  def forward_backward(_adapter_opts, %{id: training_client}, datums) do
    # Convert CrucibleTrain datums to Tinkex format
    tinkex_data = Enum.map(datums, &datum_to_tinkex/1)

    # Submit forward_backward and return the task as a future
    {:ok, task} =
      Tinkex.TrainingClient.forward_backward(training_client, tinkex_data, :cross_entropy)

    task
  end

  @impl true
  @spec optim_step(keyword(), session(), float()) :: Task.t()
  def optim_step(_adapter_opts, %{id: training_client}, lr) do
    adam_params = %Tinkex.Types.AdamParams{
      learning_rate: lr,
      beta1: 0.9,
      beta2: 0.999,
      eps: 1.0e-8
    }

    {:ok, task} = Tinkex.TrainingClient.optim_step(training_client, adam_params)
    task
  end

  @impl true
  @spec await(keyword(), Task.t()) :: {:ok, map()} | {:error, term()}
  def await(_adapter_opts, task) do
    Task.await(task, :infinity)
  end

  @impl true
  @spec save_checkpoint(keyword(), session(), String.t()) :: :ok | {:error, term()}
  def save_checkpoint(_adapter_opts, %{id: training_client}, name) do
    {:ok, task} = Tinkex.TrainingClient.save_state(training_client, name)

    case Task.await(task, :infinity) do
      {:ok, _result} -> :ok
      {:error, reason} -> {:error, {:checkpoint_save_failed, reason}}
    end
  end

  @impl true
  @spec load_checkpoint(keyword(), session(), String.t()) :: :ok | {:error, term()}
  def load_checkpoint(_adapter_opts, %{id: training_client}, name) do
    {:ok, task} = Tinkex.TrainingClient.load_state(training_client, name)

    case Task.await(task, :infinity) do
      {:ok, _} -> :ok
      {:error, reason} -> {:error, {:checkpoint_load_failed, reason}}
    end
  end

  @impl true
  @spec close_session(keyword(), session()) :: :ok
  def close_session(_adapter_opts, %{service_client: service_client}) do
    # Stop the service client which will clean up training client
    if Process.alive?(service_client) do
      GenServer.stop(service_client, :normal)
    end

    :ok
  end

  # ============================================================================
  # Extended Callbacks (not part of CrucibleTrain.Ports.TrainingClient)
  # ============================================================================

  @doc """
  Get the tokenizer from the session.

  This is a Tinkex-specific extension not part of the standard port behaviour.
  """
  @spec get_tokenizer(keyword(), session()) :: {:ok, term()} | {:error, term()}
  def get_tokenizer(_adapter_opts, %{id: training_client}) do
    Tinkex.TrainingClient.get_tokenizer(training_client)
  end

  # ============================================================================
  # Tinkex-Specific Extensions (not part of port behaviour)
  # ============================================================================

  @doc """
  Save weights for use in sampling/inference.

  This is a Tinkex-specific operation not part of the standard port behaviour.
  """
  @spec save_weights_for_sampler(session(), String.t()) :: {:ok, term()} | {:error, term()}
  def save_weights_for_sampler(%{id: training_client}, name) do
    {:ok, task} = Tinkex.TrainingClient.save_weights_for_sampler(training_client, name)

    case Task.await(task, :infinity) do
      {:ok, result} -> {:ok, result}
      {:error, reason} -> {:error, {:save_weights_failed, reason}}
    end
  end

  @doc """
  Get the service client from a session.

  Useful for advanced operations that need direct access to the Tinkex client.
  """
  @spec get_service_client(session()) :: pid()
  def get_service_client(%{service_client: service_client}), do: service_client

  # ============================================================================
  # Private Functions
  # ============================================================================

  defp do_start_session(model, config, api_key, base_url) do
    tinkex_config = Tinkex.Config.new(api_key: api_key, base_url: base_url)

    with {:ok, service_client} <- create_service_client(tinkex_config),
         {:ok, training_client} <- create_training_client(service_client, model, config) do
      {:ok,
       %{
         id: training_client,
         service_client: service_client,
         model: model,
         config: config
       }}
    end
  end

  defp create_service_client(tinkex_config) do
    Logger.debug("Creating Tinkex ServiceClient...")

    case Tinkex.ServiceClient.start_link(config: tinkex_config) do
      {:ok, pid} ->
        Logger.debug("ServiceClient created")
        {:ok, pid}

      {:error, reason} ->
        {:error, {:service_client_failed, reason}}
    end
  end

  defp create_training_client(service_client, model, config) do
    lora_rank = Map.get(config, :lora_rank, 16)
    Logger.debug("Creating Tinkex TrainingClient (LoRA rank=#{lora_rank})...")

    lora_config = %Tinkex.Types.LoraConfig{rank: lora_rank}

    case Tinkex.ServiceClient.create_lora_training_client(
           service_client,
           model,
           lora_config: lora_config,
           call_timeout: :infinity
         ) do
      {:ok, pid} ->
        Logger.debug("TrainingClient created")
        {:ok, pid}

      {:error, reason} ->
        {:error, {:training_client_failed, reason}}
    end
  end

  # Convert CrucibleKitchen.Types.Datum to Tinkex format
  defp datum_to_tinkex(%Datum{} = datum) do
    tinkex_model_input = model_input_to_tinkex(datum.model_input)

    tinkex_loss_fn_inputs =
      datum.loss_fn_inputs
      |> Enum.map(fn {key, tensor_data} ->
        {key, tensor_data_to_tinkex(tensor_data)}
      end)
      |> Map.new()

    %{
      model_input: tinkex_model_input,
      loss_fn_inputs: tinkex_loss_fn_inputs
    }
  end

  defp model_input_to_tinkex(%ModelInput{chunks: chunks}) do
    tinkex_chunks =
      Enum.map(chunks, fn
        %EncodedTextChunk{tokens: tokens} ->
          %Tinkex.Types.EncodedTextChunk{tokens: tokens, type: "encoded_text"}
      end)

    %Tinkex.Types.ModelInput{chunks: tinkex_chunks}
  end

  defp tensor_data_to_tinkex(%TensorData{} = td) do
    %Tinkex.Types.TensorData{
      data: td.data,
      dtype: td.dtype,
      shape: td.shape
    }
  end
end
