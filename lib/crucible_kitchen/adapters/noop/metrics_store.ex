defmodule CrucibleKitchen.Adapters.Noop.MetricsStore do
  @moduledoc """
  Noop adapter for MetricsStore port.

  Uses an in-memory ETS table for storage. Useful for:
  - Testing stages in isolation
  - Development without external services
  - Quick experimentation
  """

  @behaviour CrucibleTelemetry.Ports.MetricsStore

  @table :crucible_kitchen_noop_metrics

  @impl true
  def record(_opts, run_id, metric_name, value, record_opts) do
    ensure_table()
    step = Keyword.get(record_opts, :step, 0)
    metadata = Keyword.get(record_opts, :metadata, %{})

    point = %{
      step: step,
      value: value,
      timestamp: DateTime.utc_now(),
      metadata: metadata
    }

    key = {run_id, metric_name}

    case :ets.lookup(@table, key) do
      [{^key, points}] ->
        :ets.insert(@table, {key, [point | points]})

      [] ->
        :ets.insert(@table, {key, [point]})
    end

    :ok
  end

  @impl true
  def flush(_opts, _run_id) do
    :ok
  end

  @impl true
  def read(_opts, run_id) do
    ensure_table()

    entries =
      :ets.foldl(
        fn
          {{^run_id, metric_name}, points}, acc ->
            new_entries =
              Enum.map(points, fn point ->
                %{
                  "run_id" => run_id,
                  "metric" => to_string(metric_name),
                  "value" => point.value,
                  "step" => point.step,
                  "timestamp" => DateTime.to_iso8601(point.timestamp),
                  "metadata" => point.metadata
                }
              end)

            new_entries ++ acc

          _, acc ->
            acc
        end,
        [],
        @table
      )

    {:ok, Enum.sort_by(entries, &(&1["step"] || 0))}
  end

  # ==========================================================================
  # Extended Functionality (not part of CrucibleTelemetry.Ports.MetricsStore)
  # ==========================================================================

  def record_batch(_opts, run_id, metrics, record_opts) do
    ensure_table()
    step = Keyword.get(record_opts, :step, 0)
    metadata = Keyword.get(record_opts, :metadata, %{})
    timestamp = DateTime.utc_now()

    Enum.each(metrics, fn {metric_name, value} ->
      point = %{
        step: step,
        value: value,
        timestamp: timestamp,
        metadata: metadata
      }

      key = {run_id, metric_name}

      case :ets.lookup(@table, key) do
        [{^key, points}] ->
          :ets.insert(@table, {key, [point | points]})

        [] ->
          :ets.insert(@table, {key, [point]})
      end
    end)

    :ok
  end

  def get_history(_opts, run_id, metric_name, query_opts) do
    ensure_table()
    key = {run_id, metric_name}

    case :ets.lookup(@table, key) do
      [{^key, points}] ->
        points =
          points
          |> maybe_filter_step_min(Keyword.get(query_opts, :step_min))
          |> maybe_filter_step_max(Keyword.get(query_opts, :step_max))
          |> maybe_limit(Keyword.get(query_opts, :limit))
          |> Enum.sort_by(& &1.step)

        {:ok, points}

      [] ->
        {:ok, []}
    end
  end

  def get_latest(_opts, run_id, metric_name) do
    ensure_table()
    key = {run_id, metric_name}

    case :ets.lookup(@table, key) do
      [{^key, [point | _]}] -> {:ok, point}
      [{^key, []}] -> {:error, :not_found}
      [] -> {:error, :not_found}
    end
  end

  def list_metrics(_opts, run_id) do
    ensure_table()

    metrics =
      :ets.foldl(
        fn
          {{^run_id, metric_name}, _points}, acc -> [metric_name | acc]
          _, acc -> acc
        end,
        [],
        @table
      )

    {:ok, metrics}
  end

  defp maybe_filter_step_min(points, nil), do: points
  defp maybe_filter_step_min(points, min), do: Enum.filter(points, &(&1.step >= min))

  defp maybe_filter_step_max(points, nil), do: points
  defp maybe_filter_step_max(points, max), do: Enum.filter(points, &(&1.step <= max))

  defp maybe_limit(points, nil), do: points
  defp maybe_limit(points, limit), do: Enum.take(points, limit)

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
