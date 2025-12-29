defmodule CrucibleKitchen.Adapters.Noop.HubClient do
  @moduledoc """
  Noop adapter for HubClient port.

  Returns synthetic model information. Useful for:
  - Testing stages in isolation
  - Development without hub access
  - Offline mode
  """

  @behaviour CrucibleTrain.Ports.HubClient

  @impl true
  def download(_opts, download_opts) do
    repo_id = Keyword.fetch!(download_opts, :repo_id)
    filename = Keyword.fetch!(download_opts, :filename)
    repo_type = Keyword.get(download_opts, :repo_type, :model)
    revision = Keyword.get(download_opts, :revision, "main")

    path = "/tmp/noop_hub/#{repo_type}/#{normalize_repo(repo_id)}/#{revision}/#{filename}"
    {:ok, path}
  end

  @impl true
  def snapshot(_opts, snapshot_opts) do
    repo_id = Keyword.fetch!(snapshot_opts, :repo_id)
    repo_type = Keyword.get(snapshot_opts, :repo_type, :model)
    revision = Keyword.get(snapshot_opts, :revision, "main")

    path = "/tmp/noop_hub/#{repo_type}/#{normalize_repo(repo_id)}/#{revision}"
    {:ok, path}
  end

  @impl true
  def list_files(_opts, repo_id, _list_opts \\ []) do
    files = [
      "README.md",
      "config.json",
      "model.safetensors",
      "#{normalize_repo(repo_id)}.bin"
    ]

    {:ok, files}
  end

  # ==========================================================================
  # Extended Functionality (not part of CrucibleTrain.Ports.HubClient)
  # ==========================================================================

  def download_model(opts, model_id, download_opts) do
    snapshot(opts, Keyword.merge(download_opts, repo_id: model_id, repo_type: :model))
  end

  def upload_model(_opts, _model_id, _local_path, _upload_opts) do
    {:error, :not_implemented}
  end

  def get_model_info(_opts, model_id) do
    info = %{
      id: model_id,
      author: extract_author(model_id),
      downloads: :rand.uniform(100_000),
      likes: :rand.uniform(10_000),
      tags: ["text-generation", "llm"],
      pipeline_tag: "text-generation",
      library_name: "transformers"
    }

    {:ok, info}
  end

  def list_models(_opts, _search_opts) do
    models = [
      %{id: "noop/model-1", author: "noop", downloads: 1000},
      %{id: "noop/model-2", author: "noop", downloads: 2000},
      %{id: "noop/model-3", author: "noop", downloads: 3000}
    ]

    {:ok, models}
  end

  def download_file(opts, model_id, filename) do
    download(opts, repo_id: model_id, filename: filename, repo_type: :model)
  end

  defp extract_author(model_id) do
    case String.split(model_id, "/", parts: 2) do
      [author, _] -> author
      _ -> "unknown"
    end
  end

  defp normalize_repo(repo_id) do
    repo_id
    |> to_string()
    |> String.replace("/", "_")
  end
end
