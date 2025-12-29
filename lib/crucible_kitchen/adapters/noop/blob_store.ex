defmodule CrucibleKitchen.Adapters.Noop.BlobStore do
  @moduledoc """
  Noop adapter for BlobStore port.

  Uses an in-memory ETS table for storage. Useful for:
  - Testing stages in isolation
  - Development without cloud storage
  - Short-lived test runs
  """

  @behaviour CrucibleTrain.Ports.BlobStore

  @table :crucible_kitchen_noop_blobs

  @impl true
  def read(_opts, path) do
    ensure_table()

    case :ets.lookup(@table, path) do
      [{^path, data}] -> {:ok, data}
      [{^path, data, _info}] -> {:ok, data}
      [] -> {:error, :not_found}
    end
  end

  @impl true
  def stream(opts, path) do
    case read(opts, path) do
      {:ok, data} -> {:ok, [data]}
      {:error, reason} -> {:error, reason}
    end
  end

  @impl true
  def write(_opts, path, data) do
    ensure_table()
    binary = IO.iodata_to_binary(data)

    info = %{
      key: path,
      size: byte_size(binary),
      created_at: DateTime.utc_now(),
      metadata: %{}
    }

    :ets.insert(@table, {path, binary, info})
    :ok
  end

  @impl true
  def exists?(_opts, path) do
    ensure_table()
    :ets.member(@table, path)
  end

  # ==========================================================================
  # Extended Functionality (not part of CrucibleTrain.Ports.BlobStore)
  # ==========================================================================

  def put(_opts, key, data, _put_opts) do
    ensure_table()
    binary = IO.iodata_to_binary(data)

    info = %{
      key: key,
      size: byte_size(binary),
      created_at: DateTime.utc_now(),
      metadata: %{}
    }

    :ets.insert(@table, {key, binary, info})
    {:ok, info}
  end

  def get(opts, key, _get_opts) do
    read(opts, key)
  end

  def delete(_opts, key) do
    ensure_table()
    :ets.delete(@table, key)
    :ok
  end

  def list(_opts, prefix) do
    ensure_table()

    blobs =
      :ets.foldl(
        fn
          {key, _data, info}, acc ->
            if String.starts_with?(key, prefix) do
              [info | acc]
            else
              acc
            end

          {key, data}, acc ->
            if String.starts_with?(key, prefix) do
              [build_info(key, data) | acc]
            else
              acc
            end
        end,
        [],
        @table
      )

    {:ok, blobs}
  end

  defp build_info(key, data) do
    %{
      key: key,
      size: byte_size(data),
      created_at: DateTime.utc_now(),
      metadata: %{}
    }
  end

  defp ensure_table do
    case :ets.whereis(@table) do
      :undefined ->
        try do
          :ets.new(@table, [:named_table, :set, :public, {:read_concurrency, true}])
        rescue
          ArgumentError -> :ok
        end

        @table

      _ ->
        @table
    end
  end
end
