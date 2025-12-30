defmodule CrucibleKitchen.Adapters.Noop.ModelRegistry do
  @moduledoc """
  No-op model registry adapter for testing and dry runs.

  This adapter stores models in-memory (process dictionary) and returns
  synthetic model IDs. Useful for testing workflows without a real registry.

  ## Usage

      adapters = %{
        model_registry: CrucibleKitchen.Adapters.Noop.ModelRegistry,
        ...
      }

      CrucibleKitchen.run(:supervised, config, adapters: adapters, dry_run: true)
  """

  @behaviour CrucibleKitchen.Ports.ModelRegistry

  require Logger

  @impl true
  def register(_opts, artifact) do
    id = generate_id()

    # Keep metadata as-is from artifact; metrics are stored separately if needed
    # but not exposed in the model return to conform to the behaviour spec
    model = %{
      id: id,
      name: artifact.name,
      version: artifact.version,
      artifact_uri: Map.get(artifact, :artifact_uri) || "local://noop",
      metadata: Map.get(artifact, :metadata, %{}),
      lineage: Map.get(artifact, :lineage, %{}),
      created_at: DateTime.utc_now()
    }

    # Store in process dictionary for retrieval
    store_model(model)

    Logger.debug(
      "[Noop.ModelRegistry] Registered model: #{model.name} v#{model.version} (id: #{id})"
    )

    {:ok, model}
  end

  @impl true
  def get(_opts, model_id) do
    case get_stored_model(model_id) do
      nil -> {:error, :not_found}
      model -> {:ok, model}
    end
  end

  @impl true
  def list(opts) do
    models = get_all_models()

    filtered =
      models
      |> filter_by_name(Keyword.get(opts, :name))
      |> maybe_limit(Keyword.get(opts, :limit))

    {:ok, filtered}
  end

  # ID generation
  defp generate_id do
    timestamp = System.system_time(:second)
    random = :rand.uniform(999_999)
    "noop_#{timestamp}_#{random}"
  end

  # Process dictionary storage helpers
  defp store_model(model) do
    models = Process.get(:noop_model_registry, %{})
    Process.put(:noop_model_registry, Map.put(models, model.id, model))
  end

  defp get_stored_model(id) do
    models = Process.get(:noop_model_registry, %{})
    Map.get(models, id) || Map.get(models, to_string(id))
  end

  defp get_all_models do
    Process.get(:noop_model_registry, %{}) |> Map.values()
  end

  # Filtering helpers
  defp filter_by_name(models, nil), do: models
  defp filter_by_name(models, name), do: Enum.filter(models, &(&1.name == name))

  defp maybe_limit(models, nil), do: models
  defp maybe_limit(models, limit), do: Enum.take(models, limit)
end
