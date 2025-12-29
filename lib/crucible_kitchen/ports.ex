defmodule CrucibleKitchen.Ports do
  @moduledoc """
  Port system for backend-agnostic adapters.

  Ports define the contracts that adapter implementations must fulfill.
  This module provides the resolution and validation logic for adapters.

  ## Architecture

  ```
  CrucibleTrain.Ports.TrainingClient     <- Behaviour (what adapters implement)
          │
          ├── TinkexCookbook.Adapters.TrainingClient.Tinkex  <- Real impl
          ├── CrucibleKitchen.Adapters.Noop.TrainingClient   <- Test impl
          └── YourApp.Adapters.TrainingClient                <- Your impl
  ```

  ## Usage

  Adapters are passed to `CrucibleKitchen.run/3` as a map:

      adapters = %{
        training_client: {MyAdapter.TrainingClient, api_key: "..."},
        dataset_store: MyAdapter.DatasetStore,
        blob_store: CrucibleKitchen.Adapters.Noop.BlobStore
      }

      CrucibleKitchen.run(:supervised, config, adapters: adapters)

  Adapter values can be:
  - `module` - A module implementing the port behaviour
  - `{module, opts}` - A module with adapter-specific options

  ## Available Ports

  | Port | Purpose |
  |------|---------|
  | `:training_client` | Training backend (forward/backward, optim) |
  | `:dataset_store` | Dataset loading |
  | `:blob_store` | Artifact storage (checkpoints, weights) |
  | `:hub_client` | Model hub (HuggingFace, etc.) |
  | `:metrics_store` | Metrics persistence |
  """

  @type adapter_spec :: module() | {module(), keyword()}
  @type adapter_map :: %{optional(atom()) => adapter_spec()}
  @type resolved :: {module(), keyword()}

  @doc """
  Resolve an adapter from the adapter map.

  Returns `{module, opts}` tuple.

  ## Examples

      iex> resolve(%{training_client: MyAdapter}, :training_client)
      {MyAdapter, []}

      iex> resolve(%{training_client: {MyAdapter, api_key: "x"}}, :training_client)
      {MyAdapter, [api_key: "x"]}
  """
  @spec resolve(adapter_map(), atom()) :: resolved() | nil
  def resolve(adapters, port_name) when is_map(adapters) and is_atom(port_name) do
    case Map.get(adapters, port_name) do
      nil -> nil
      {module, opts} when is_atom(module) and is_list(opts) -> {module, opts}
      module when is_atom(module) -> {module, []}
    end
  end

  @doc """
  Resolve an adapter, raising if not found.
  """
  @spec resolve!(adapter_map(), atom()) :: resolved()
  def resolve!(adapters, port_name) do
    case resolve(adapters, port_name) do
      nil -> raise ArgumentError, "Missing required adapter: #{port_name}"
      resolved -> resolved
    end
  end

  @doc """
  Validate that all required adapters are present and implement their ports.
  """
  @spec validate(adapter_map(), [atom()]) :: :ok | {:error, [map()]}
  def validate(adapters, required_ports) do
    errors =
      required_ports
      |> Enum.flat_map(&validate_port(adapters, &1))

    if errors == [], do: :ok, else: {:error, errors}
  end

  defp validate_port(adapters, port_name) do
    case resolve(adapters, port_name) do
      nil ->
        [%{port: port_name, error: :missing}]

      {module, _opts} ->
        validate_port_implementation(port_name, module)
    end
  end

  defp validate_port_implementation(port_name, module) do
    port_behaviour = port_behaviour(port_name)

    if port_behaviour && not implements?(module, port_behaviour) do
      missing = missing_callbacks(module, port_behaviour)
      [%{port: port_name, module: module, error: :incomplete, missing: missing}]
    else
      []
    end
  end

  @doc """
  Check if a module implements a behaviour.
  """
  @spec implements?(module(), module()) :: boolean()
  def implements?(module, behaviour) do
    behaviours = module.module_info(:attributes)[:behaviour] || []
    behaviour in behaviours
  end

  @doc """
  List callbacks missing from a module for a behaviour.
  """
  @spec missing_callbacks(module(), module()) :: [atom()]
  def missing_callbacks(module, behaviour) do
    required = behaviour.behaviour_info(:callbacks)
    exported = module.__info__(:functions)

    required
    |> Enum.filter(fn {name, arity} -> {name, arity} not in exported end)
    |> Enum.map(fn {name, _arity} -> name end)
  end

  # Map port names to their behaviour modules
  defp port_behaviour(:training_client), do: CrucibleTrain.Ports.TrainingClient
  defp port_behaviour(:dataset_store), do: CrucibleTrain.Ports.DatasetStore
  defp port_behaviour(:blob_store), do: CrucibleTrain.Ports.BlobStore
  defp port_behaviour(:hub_client), do: CrucibleTrain.Ports.HubClient
  defp port_behaviour(:metrics_store), do: CrucibleTelemetry.Ports.MetricsStore
  defp port_behaviour(_), do: nil
end
