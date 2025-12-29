defmodule CrucibleKitchen.Adapters.HfHub.HubClient do
  @moduledoc """
  HubClient adapter for HuggingFace Hub.

  Implements CrucibleTrain.Ports.HubClient using the hf_hub_ex library.
  Provides model downloading, metadata fetching, and search capabilities.

  ## Configuration

  Environment variables:
  - `HF_TOKEN` - HuggingFace API token for private repos

  ## Adapter Options

  - `:token` - Override HuggingFace token
  - `:cache_dir` - Override cache directory

  ## Usage

      config :my_app,
        adapters: %{
          hub_client: {CrucibleKitchen.Adapters.HfHub.HubClient, [
            token: System.get_env("HF_TOKEN")
          ]}
        }
  """

  @behaviour CrucibleTrain.Ports.HubClient

  require Logger

  # ============================================================================
  # CrucibleTrain.Ports.HubClient Callbacks
  # ============================================================================

  @impl true
  @spec download(keyword(), keyword()) :: {:ok, Path.t()} | {:error, term()}
  def download(opts, download_opts) do
    token = Keyword.get(opts, :token) || get_token()
    repo_id = Keyword.fetch!(download_opts, :repo_id)
    filename = Keyword.fetch!(download_opts, :filename)
    repo_type = Keyword.get(download_opts, :repo_type, :model)
    revision = Keyword.get(download_opts, :revision, "main")

    Logger.debug("Downloading file: #{repo_id}/#{filename}")

    case HfHub.Download.hf_hub_download(
           repo_id: repo_id,
           filename: filename,
           repo_type: repo_type,
           revision: revision,
           token: token
         ) do
      {:ok, path} ->
        Logger.debug("File downloaded to: #{path}")
        {:ok, path}

      {:error, reason} ->
        Logger.error("Failed to download #{repo_id}/#{filename}: #{inspect(reason)}")
        {:error, {:download_failed, reason}}
    end
  end

  @impl true
  @spec snapshot(keyword(), keyword()) :: {:ok, Path.t()} | {:error, term()}
  def snapshot(opts, snapshot_opts) do
    token = Keyword.get(opts, :token) || get_token()
    repo_id = Keyword.fetch!(snapshot_opts, :repo_id)
    repo_type = Keyword.get(snapshot_opts, :repo_type, :model)
    revision = Keyword.get(snapshot_opts, :revision, "main")
    ignore_patterns = Keyword.get(snapshot_opts, :ignore_patterns, ["*.msgpack", "*.h5"])
    allow_patterns = Keyword.get(snapshot_opts, :allow_patterns, [])

    Logger.debug("Downloading snapshot: #{repo_id} (revision: #{revision})")

    case HfHub.Download.snapshot_download(
           repo_id: repo_id,
           repo_type: repo_type,
           revision: revision,
           token: token,
           ignore_patterns: ignore_patterns,
           allow_patterns: allow_patterns
         ) do
      {:ok, path} ->
        Logger.debug("Snapshot downloaded to: #{path}")
        {:ok, path}

      {:error, reason} ->
        Logger.error("Failed to download snapshot #{repo_id}: #{inspect(reason)}")
        {:error, {:snapshot_failed, reason}}
    end
  end

  @impl true
  @spec list_files(keyword(), String.t(), keyword()) :: {:ok, [String.t()]} | {:error, term()}
  def list_files(opts, repo_id, list_opts \\ []) do
    token = Keyword.get(opts, :token) || get_token()
    repo_type = Keyword.get(list_opts, :repo_type, :model)
    revision = Keyword.get(list_opts, :revision, "main")

    case HfHub.Api.list_files(repo_id, repo_type: repo_type, revision: revision, token: token) do
      {:ok, files} ->
        # Extract just the filenames from the response
        filenames =
          Enum.map(files, fn
            %{rfilename: name} -> name
            %{"rfilename" => name} -> name
            name when is_binary(name) -> name
            _ -> nil
          end)
          |> Enum.reject(&is_nil/1)

        {:ok, filenames}

      {:error, reason} ->
        {:error, {:list_files_failed, reason}}
    end
  end

  # ============================================================================
  # Extended Functionality (not part of CrucibleTrain.Ports.HubClient)
  # ============================================================================

  @doc """
  Download a model snapshot.

  This is a convenience wrapper around `snapshot/2` for model downloads.
  """
  @spec download_model(keyword(), String.t(), keyword()) :: {:ok, String.t()} | {:error, term()}
  def download_model(opts, model_id, download_opts \\ []) do
    snapshot(opts, Keyword.merge(download_opts, repo_id: model_id, repo_type: :model))
  end

  @doc """
  Get model info from HuggingFace Hub.
  """
  @spec get_model_info(keyword(), String.t()) :: {:ok, map()} | {:error, term()}
  def get_model_info(opts, model_id) do
    token = Keyword.get(opts, :token) || get_token()

    case HfHub.Api.model_info(model_id, token: token) do
      {:ok, info} ->
        {:ok, normalize_model_info(info)}

      {:error, :not_found} ->
        {:error, :not_found}

      {:error, reason} ->
        {:error, {:api_error, reason}}
    end
  end

  @doc """
  Upload a model to HuggingFace Hub.
  """
  @spec upload_model(keyword(), String.t(), String.t(), keyword()) ::
          {:ok, String.t()} | {:error, term()}
  def upload_model(_opts, _model_id, _local_path, _upload_opts) do
    # Upload not yet implemented in hf_hub_ex
    {:error, :not_implemented}
  end

  @doc """
  List models from HuggingFace Hub.
  """
  @spec list_models(keyword(), keyword()) :: {:ok, [map()]} | {:error, term()}
  def list_models(opts, search_opts \\ []) do
    token = Keyword.get(opts, :token) || get_token()

    api_opts =
      search_opts
      |> Keyword.take([:filter, :sort, :direction, :limit, :author])
      |> Keyword.put(:token, token)

    case HfHub.Api.list_models(api_opts) do
      {:ok, models} ->
        {:ok, Enum.map(models, &normalize_model_info/1)}

      {:error, reason} ->
        {:error, {:api_error, reason}}
    end
  end

  @doc """
  Download a single file from a model repository.
  """
  @spec download_file(keyword(), String.t(), String.t()) :: {:ok, String.t()} | {:error, term()}
  def download_file(opts, model_id, filename) do
    download(opts, repo_id: model_id, filename: filename, repo_type: :model)
  end

  # ============================================================================
  # Extended Functionality (not part of port behaviour)
  # ============================================================================

  @doc """
  Download a dataset from HuggingFace Hub.

  This is specific to HfHub adapter - not part of the generic HubClient port.
  """
  @spec download_dataset(keyword(), String.t(), keyword()) :: {:ok, String.t()} | {:error, term()}
  def download_dataset(opts, dataset_id, download_opts \\ []) do
    token = Keyword.get(opts, :token) || get_token()
    revision = Keyword.get(download_opts, :revision, "main")

    case HfHub.Download.snapshot_download(
           repo_id: dataset_id,
           repo_type: :dataset,
           revision: revision,
           token: token
         ) do
      {:ok, path} ->
        {:ok, path}

      {:error, reason} ->
        {:error, {:download_failed, reason}}
    end
  end

  @doc """
  Get dataset info from HuggingFace Hub.
  """
  @spec get_dataset_info(keyword(), String.t()) :: {:ok, map()} | {:error, term()}
  def get_dataset_info(opts, dataset_id) do
    token = Keyword.get(opts, :token) || get_token()

    case HfHub.Api.dataset_info(dataset_id, token: token) do
      {:ok, info} ->
        {:ok, info}

      {:error, reason} ->
        {:error, {:api_error, reason}}
    end
  end

  @doc """
  List files in a model repository.
  """
  @spec list_model_files(keyword(), String.t(), keyword()) ::
          {:ok, [map()]} | {:error, term()}
  def list_model_files(opts, model_id, list_opts \\ []) do
    token = Keyword.get(opts, :token) || get_token()
    revision = Keyword.get(list_opts, :revision, "main")

    case HfHub.Api.list_files(model_id, repo_type: :model, revision: revision, token: token) do
      {:ok, files} ->
        {:ok, files}

      {:error, reason} ->
        {:error, {:api_error, reason}}
    end
  end

  @doc """
  Check if a file is cached locally.
  """
  @spec cached?(String.t(), String.t(), keyword()) :: boolean()
  def cached?(model_id, filename, opts \\ []) do
    revision = Keyword.get(opts, :revision, "main")
    cache_path = HfHub.FS.file_path(model_id, :model, filename, revision)
    File.exists?(cache_path)
  end

  # ============================================================================
  # Private Functions
  # ============================================================================

  defp get_token do
    case HfHub.Auth.get_token() do
      {:ok, token} -> token
      _ -> System.get_env("HF_TOKEN")
    end
  end

  defp normalize_model_info(info) when is_map(info) do
    %{
      id: Map.get(info, :id),
      author: Map.get(info, :author),
      downloads: Map.get(info, :downloads, 0),
      likes: Map.get(info, :likes, 0),
      tags: Map.get(info, :tags, []),
      pipeline_tag: Map.get(info, :pipeline_tag),
      library_name: extract_library_name(Map.get(info, :tags, []))
    }
  end

  defp extract_library_name(tags) when is_list(tags) do
    library_tags = ["transformers", "pytorch", "tensorflow", "jax", "gguf", "mlx"]

    Enum.find(tags, fn tag ->
      String.downcase(tag) in library_tags
    end)
  end

  defp extract_library_name(_), do: nil
end
