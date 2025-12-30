defmodule CrucibleKitchen.Adapters.ModelRegistry.RegistryClient do
  @moduledoc """
  Model registry adapter using CrucibleModelRegistry.

  This adapter integrates with the CrucibleModelRegistry library for
  persistent model versioning with lineage tracking.

  ## Configuration

  CrucibleModelRegistry requires a configured Repo:

      config :crucible_model_registry, repo: MyApp.Repo

  ## Usage

      adapters = %{
        model_registry: CrucibleKitchen.Adapters.ModelRegistry.RegistryClient,
        ...
      }

      CrucibleKitchen.run(:supervised, config, adapters: adapters)
  """

  @behaviour CrucibleKitchen.Ports.ModelRegistry

  require Logger

  @impl true
  def register(opts, artifact) do
    params = build_register_params(artifact, opts)

    case CrucibleModelRegistry.register(params) do
      {:ok, version} ->
        model = normalize_version(version)
        Logger.info("[ModelRegistry] Registered model: #{model.name} v#{model.version}")
        {:ok, model}

      {:error, {:duplicate_config, existing}} ->
        Logger.warning("[ModelRegistry] Duplicate config detected, returning existing version")
        {:ok, normalize_version(existing)}

      {:error, reason} ->
        Logger.error("[ModelRegistry] Registration failed: #{inspect(reason)}")
        {:error, reason}
    end
  end

  @impl true
  def get(_opts, model_id) do
    id = parse_model_id(model_id)

    case CrucibleModelRegistry.get_version(id) do
      {:ok, version} ->
        {:ok, normalize_version(version)}

      {:error, :not_found} ->
        {:error, :not_found}
    end
  rescue
    e ->
      Logger.error("[ModelRegistry] Get failed: #{inspect(e)}")
      {:error, :not_found}
  end

  @impl true
  def list(opts) do
    filters = build_query_filters(opts)
    versions = CrucibleModelRegistry.query(filters)
    {:ok, Enum.map(versions, &normalize_version/1)}
  end

  # Build registration parameters from artifact
  defp build_register_params(artifact, opts) do
    training_config = Map.get(artifact, :metadata, %{}) |> Map.get(:training_config, %{})

    metrics =
      Map.get(artifact, :metrics, %{})
      |> Map.merge(Map.get(artifact, :metadata, %{}) |> Map.get(:final_metrics, %{}))

    lineage = Map.get(artifact, :lineage, %{})

    %{
      model_name: artifact.name,
      version: artifact.version,
      recipe: Keyword.get(opts, :recipe, :supervised),
      base_model:
        Map.get(training_config, :model) || Map.get(training_config, "model") || artifact.name,
      training_config: training_config,
      metrics: metrics,
      artifacts: build_artifacts(artifact),
      stage: Keyword.get(opts, :stage, :experiment),
      lineage_type: Map.get(lineage, :workflow) |> lineage_type_from_workflow(),
      parent_version_id: Keyword.get(opts, :parent_version_id)
    }
  end

  # Build artifact list from artifact map
  defp build_artifacts(artifact) do
    artifact_uri = Map.get(artifact, :artifact_uri)

    if artifact_uri do
      [
        %{
          type: :weights,
          storage_backend: infer_storage_backend(artifact_uri),
          storage_path: artifact_uri,
          checksum: nil,
          size_bytes: nil
        }
      ]
    else
      []
    end
  end

  # Infer storage backend from URI
  defp infer_storage_backend(uri) when is_binary(uri) do
    cond do
      String.starts_with?(uri, "s3://") -> :s3
      String.starts_with?(uri, "hf://") -> :huggingface
      String.starts_with?(uri, "tinker://") -> :tinker
      true -> :local
    end
  end

  defp infer_storage_backend(_), do: :local

  # Convert workflow atom to lineage type
  defp lineage_type_from_workflow(:supervised), do: :fine_tune
  defp lineage_type_from_workflow(:distillation), do: :distillation
  defp lineage_type_from_workflow(:preference), do: :fine_tune
  defp lineage_type_from_workflow(:reinforcement), do: :fine_tune
  defp lineage_type_from_workflow(_), do: nil

  # Normalize CrucibleModelRegistry version to port model format
  defp normalize_version(%{} = version) do
    %{
      id: Map.get(version, :id),
      name: get_model_name(version),
      version: Map.get(version, :version, "unknown"),
      artifact_uri: get_artifact_uri(version),
      metadata: %{
        recipe: Map.get(version, :recipe),
        base_model: Map.get(version, :base_model),
        training_config: Map.get(version, :training_config, %{}),
        stage: Map.get(version, :stage)
      },
      lineage: %{
        parent_version_id: Map.get(version, :parent_version_id),
        lineage_type: Map.get(version, :lineage_type)
      },
      metrics: Map.get(version, :metrics, %{}),
      created_at: Map.get(version, :inserted_at) || DateTime.utc_now()
    }
  end

  defp get_model_name(%{model: %{name: name}}), do: name
  defp get_model_name(%{model_name: name}), do: name
  defp get_model_name(_), do: "unknown"

  defp get_artifact_uri(%{artifacts: [%{storage_path: path} | _]}), do: path
  defp get_artifact_uri(%{artifact_uri: uri}), do: uri
  defp get_artifact_uri(_), do: nil

  # Build query filters
  defp build_query_filters(opts) do
    opts
    |> Enum.reduce(%{}, fn
      {:name, name}, acc -> Map.put(acc, :model_name, name)
      {:limit, limit}, acc -> Map.put(acc, :limit, limit)
      {:stage, stage}, acc -> Map.put(acc, :stage, stage)
      _, acc -> acc
    end)
  end

  # Parse model ID (can be string or integer)
  defp parse_model_id(id) when is_integer(id), do: id
  defp parse_model_id(id) when is_binary(id), do: String.to_integer(id)
  defp parse_model_id(id), do: id
end
