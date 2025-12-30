defmodule CrucibleKitchen.Context do
  @moduledoc """
  Context flows through workflows, accumulating state.

  The context is immutable - each stage returns a new context with updated state.
  This provides a clear audit trail and enables reproducibility.

  ## Structure

  - `config` - User-provided configuration (immutable during execution)
  - `adapters` - Port implementations for this run
  - `state` - Mutable state accumulated during execution
  - `metrics` - Collected metrics for observability
  - `metadata` - Run metadata (started_at, run_id, etc.)

  ## Usage

      # In a stage:
      def execute(context) do
        # Read config
        epochs = get_config(context, :epochs, 1)

        # Read/write state
        session = get_state(context, :session)
        context = put_state(context, :current_epoch, 0)

        # Get adapter
        client = get_adapter(context, :training_client)

        # Record metrics
        context = record_metric(context, :loss, 0.5, step: 42)

        {:ok, context}
      end
  """

  defstruct [
    :config,
    :adapters,
    :state,
    :metrics,
    :metadata,
    :current_stage,
    :stage_opts
  ]

  @type adapter_map :: %{
          required(:training_client) => module() | {module(), keyword()},
          required(:dataset_store) => module() | {module(), keyword()},
          optional(:blob_store) => module() | {module(), keyword()},
          optional(:hub_client) => module() | {module(), keyword()},
          optional(:metrics_store) => module() | {module(), keyword()},
          optional(:model_registry) => module() | {module(), keyword()},
          optional(:evaluator) => module() | {module(), keyword()},
          optional(:feedback_client) => module() | {module(), keyword()}
        }

  @type metric :: %{
          name: atom(),
          value: number(),
          step: non_neg_integer(),
          timestamp: DateTime.t(),
          metadata: map()
        }

  @type t :: %__MODULE__{
          config: map(),
          adapters: adapter_map(),
          state: map(),
          metrics: [metric()],
          metadata: map(),
          current_stage: atom() | nil,
          stage_opts: keyword() | nil
        }

  @doc """
  Create a new context.
  """
  @spec new(map(), adapter_map()) :: t()
  def new(config, adapters) do
    %__MODULE__{
      config: config,
      adapters: normalize_adapters(adapters),
      state: %{},
      metrics: [],
      metadata: %{
        run_id: generate_run_id(),
        started_at: DateTime.utc_now()
      },
      current_stage: nil,
      stage_opts: nil
    }
  end

  @doc """
  Get a config value with optional default.
  """
  @spec get_config(t(), atom(), term()) :: term()
  def get_config(%__MODULE__{config: config}, key, default \\ nil) do
    Map.get(config, key, default)
  end

  @doc """
  Get a state value with optional default.
  """
  @spec get_state(t(), atom(), term()) :: term()
  def get_state(%__MODULE__{state: state}, key, default \\ nil) do
    Map.get(state, key, default)
  end

  @doc """
  Update a state value.
  """
  @spec put_state(t(), atom(), term()) :: t()
  def put_state(%__MODULE__{state: state} = context, key, value) do
    %{context | state: Map.put(state, key, value)}
  end

  @doc """
  Update multiple state values.
  """
  @spec merge_state(t(), map()) :: t()
  def merge_state(%__MODULE__{state: state} = context, updates) do
    %{context | state: Map.merge(state, updates)}
  end

  @doc """
  Get an adapter by port name.

  Returns `nil` if not found, or `{module, opts}` tuple.
  """
  @spec get_adapter(t(), atom()) :: {module(), keyword()} | nil
  def get_adapter(%__MODULE__{adapters: adapters}, port) do
    CrucibleKitchen.Ports.resolve(adapters, port)
  end

  @doc """
  Get an adapter by port name, raising if not found.

  Returns `{module, opts}` tuple.
  """
  @spec get_adapter!(t(), atom()) :: {module(), keyword()}
  def get_adapter!(%__MODULE__{adapters: adapters}, port) do
    CrucibleKitchen.Ports.resolve!(adapters, port)
  end

  @doc """
  Build a CrucibleTrain.Ports struct from the training-related adapters.
  """
  @spec get_train_ports(t()) :: CrucibleTrain.Ports.t()
  def get_train_ports(%__MODULE__{adapters: adapters}) do
    train_adapters =
      Map.take(adapters, [
        :training_client,
        :dataset_store,
        :hub_client,
        :blob_store,
        :llm_client,
        :embedding_client,
        :vector_store
      ])

    CrucibleTrain.Ports.new(ports: train_adapters)
  end

  @doc """
  Record a metric.
  """
  @spec record_metric(t(), atom(), number(), keyword()) :: t()
  def record_metric(
        %__MODULE__{metrics: metrics, state: state} = context,
        name,
        value,
        opts \\ []
      ) do
    step = Keyword.get(opts, :step, Map.get(state, :global_step, 0))
    metadata = Keyword.get(opts, :metadata, %{})

    metric = %{
      name: name,
      value: value,
      step: step,
      timestamp: DateTime.utc_now(),
      metadata: metadata
    }

    %{context | metrics: [metric | metrics]}
  end

  @doc """
  Get metadata value.
  """
  @spec get_metadata(t(), atom(), term()) :: term()
  def get_metadata(%__MODULE__{metadata: metadata}, key, default \\ nil) do
    Map.get(metadata, key, default)
  end

  @doc """
  Update metadata.
  """
  @spec put_metadata(t(), atom(), term()) :: t()
  def put_metadata(%__MODULE__{metadata: metadata} = context, key, value) do
    %{context | metadata: Map.put(metadata, key, value)}
  end

  # Private helpers

  defp normalize_adapters(adapters) when is_map(adapters), do: adapters

  defp normalize_adapters(module) when is_atom(module) do
    if function_exported?(module, :all, 0) do
      module.all()
    else
      raise ArgumentError, "adapters must be a map or a module with all/0 function"
    end
  end

  defp generate_run_id do
    timestamp = DateTime.utc_now() |> DateTime.to_unix()
    random = :crypto.strong_rand_bytes(4) |> Base.encode16(case: :lower)
    "run_#{timestamp}_#{random}"
  end
end
