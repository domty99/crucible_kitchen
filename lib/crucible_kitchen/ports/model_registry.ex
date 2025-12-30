defmodule CrucibleKitchen.Ports.ModelRegistry do
  @moduledoc """
  Port for model registration and versioning.

  Implementations handle model artifact storage, version tracking,
  and lineage management.

  ## Implementation Example

      defmodule MyApp.Adapters.ModelRegistry do
        @behaviour CrucibleKitchen.Ports.ModelRegistry

        @impl true
        def register(opts, artifact) do
          # Register the model version
          {:ok, %{id: "model-123", ...}}
        end

        @impl true
        def get(opts, model_id) do
          # Retrieve model by ID
          {:ok, %{id: model_id, ...}}
        end

        @impl true
        def list(opts) do
          # List all registered models
          {:ok, [%{id: "model-123"}, ...]}
        end
      end
  """

  @type opts :: keyword()
  @type artifact :: %{
          required(:name) => String.t(),
          required(:version) => String.t(),
          required(:artifact_uri) => String.t(),
          optional(:metadata) => map(),
          optional(:lineage) => map(),
          optional(:metrics) => map()
        }
  @type model :: %{
          id: term(),
          name: String.t(),
          version: String.t(),
          artifact_uri: String.t(),
          metadata: map(),
          lineage: map(),
          created_at: DateTime.t()
        }

  @doc """
  Register a trained model artifact.

  ## Parameters

  - `opts` - Adapter-specific options
  - `artifact` - Map containing model metadata:
    - `:name` - Model name (required)
    - `:version` - Version identifier (required)
    - `:artifact_uri` - URI to model artifacts (required)
    - `:metadata` - Training configuration and other metadata (optional)
    - `:lineage` - Training lineage information (optional)
    - `:metrics` - Final training/eval metrics (optional)

  ## Returns

  - `{:ok, model}` - Registered model with assigned ID
  - `{:error, reason}` - Registration failure
  """
  @callback register(opts(), artifact()) :: {:ok, model()} | {:error, term()}

  @doc """
  Retrieve a registered model by ID.

  ## Parameters

  - `opts` - Adapter-specific options
  - `model_id` - Model identifier

  ## Returns

  - `{:ok, model}` - The model record
  - `{:error, :not_found}` - Model not found
  """
  @callback get(opts(), model_id :: term()) :: {:ok, model()} | {:error, :not_found}

  @doc """
  List all registered models.

  ## Parameters

  - `opts` - Adapter-specific options, may include filters:
    - `:name` - Filter by model name
    - `:limit` - Maximum results
    - `:offset` - Pagination offset

  ## Returns

  - `{:ok, [model]}` - List of matching models
  """
  @callback list(opts()) :: {:ok, [model()]}

  @optional_callbacks []
end
