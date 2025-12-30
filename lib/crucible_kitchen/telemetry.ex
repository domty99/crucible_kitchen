defmodule CrucibleKitchen.Telemetry do
  @moduledoc """
  Telemetry event definitions and handler management.

  CrucibleKitchen emits telemetry events for all workflow and stage operations,
  enabling comprehensive observability.

  ## Events

  All events are prefixed with `[:crucible_kitchen, ...]`:

  ### Workflow Events

  - `[:crucible_kitchen, :workflow, :run, :start]` - Workflow started
  - `[:crucible_kitchen, :workflow, :run, :stop]` - Workflow completed
  - `[:crucible_kitchen, :workflow, :run, :exception]` - Workflow failed

  ### Stage Events

  - `[:crucible_kitchen, :stage, :run, :start]` - Stage started
  - `[:crucible_kitchen, :stage, :run, :stop]` - Stage completed
  - `[:crucible_kitchen, :stage, :run, :exception]` - Stage failed

  ### Training Events

  - `[:crucible_kitchen, :training, :step]` - Training step completed
  - `[:crucible_kitchen, :training, :epoch]` - Epoch completed
  - `[:crucible_kitchen, :training, :checkpoint]` - Checkpoint saved

  ## Attaching Handlers

      # Attach console handler (default)
      CrucibleKitchen.Telemetry.attach(:console)

      # Attach JSONL handler
      CrucibleKitchen.Telemetry.attach(:jsonl, path: "/var/log/kitchen.jsonl")

      # Attach custom handler
      CrucibleKitchen.Telemetry.attach_handler("my-handler", events, &my_handler/4, nil)
  """

  alias CrucibleKitchen.Telemetry.Handlers.Console
  alias CrucibleKitchen.Telemetry.Handlers.JSONL

  @events [
    # Workflow lifecycle (span events)
    [:crucible_kitchen, :workflow, :run, :start],
    [:crucible_kitchen, :workflow, :run, :stop],
    [:crucible_kitchen, :workflow, :run, :exception],

    # Stage lifecycle (span events)
    [:crucible_kitchen, :stage, :run, :start],
    [:crucible_kitchen, :stage, :run, :stop],
    [:crucible_kitchen, :stage, :run, :exception],

    # Training metrics (point events)
    [:crucible_kitchen, :training, :step],
    [:crucible_kitchen, :training, :epoch],
    [:crucible_kitchen, :training, :checkpoint],
    [:crucible_kitchen, :training, :eval],

    # Evaluation events
    [:crucible_kitchen, :eval, :complete],

    # Model registry events
    [:crucible_kitchen, :model, :registered],

    # Dataset events
    [:crucible_kitchen, :dataset, :load],
    [:crucible_kitchen, :dataset, :batch]
  ]

  @doc "Get all telemetry events."
  @spec events() :: [[atom()]]
  def events, do: @events

  @doc """
  Attach a built-in telemetry handler.

  ## Handlers

  - `:console` - Log to console with colors
  - `:jsonl` - Write JSONL to file (requires `:path` option)
  - `:prometheus` - Expose Prometheus metrics (requires `:port` option)

  ## Options

  - `:path` - File path for JSONL handler
  - `:port` - Port for Prometheus handler
  - `:filter` - Filter function `(event, measurements, metadata) -> boolean()`
  """
  @spec attach(atom(), keyword()) :: :ok
  def attach(handler, opts \\ [])

  def attach(:console, _opts) do
    handler_id = "crucible-kitchen-console"

    :telemetry.attach_many(
      handler_id,
      @events,
      &Console.handle/4,
      nil
    )

    :ok
  end

  def attach(:jsonl, opts) do
    path = Keyword.fetch!(opts, :path)
    handler_id = "crucible-kitchen-jsonl"

    :telemetry.attach_many(
      handler_id,
      @events,
      &JSONL.handle/4,
      %{path: path}
    )

    :ok
  end

  def attach(:prometheus, _opts) do
    # Prometheus handler would be implemented in a separate module
    raise "Prometheus handler not yet implemented"
  end

  @doc """
  Attach a custom telemetry handler.
  """
  @spec attach_handler(String.t(), [[atom()]], function(), term()) :: :ok | {:error, term()}
  def attach_handler(handler_id, events, handler_fun, config) do
    :telemetry.attach_many(handler_id, events, handler_fun, config)
  end

  @doc """
  Detach a handler by ID.
  """
  @spec detach(String.t()) :: :ok | {:error, :not_found}
  def detach(handler_id) do
    :telemetry.detach(handler_id)
  end

  @doc """
  Emit a training step event.
  """
  @spec emit_step(map(), map()) :: :ok
  def emit_step(measurements, metadata) do
    :telemetry.execute(
      [:crucible_kitchen, :training, :step],
      measurements,
      metadata
    )
  end

  @doc """
  Emit an epoch completion event.
  """
  @spec emit_epoch(map(), map()) :: :ok
  def emit_epoch(measurements, metadata) do
    :telemetry.execute(
      [:crucible_kitchen, :training, :epoch],
      measurements,
      metadata
    )
  end

  @doc """
  Emit a checkpoint saved event.
  """
  @spec emit_checkpoint(map(), map()) :: :ok
  def emit_checkpoint(measurements, metadata) do
    :telemetry.execute(
      [:crucible_kitchen, :training, :checkpoint],
      measurements,
      metadata
    )
  end
end

defmodule CrucibleKitchen.Telemetry.Handlers.Console do
  @moduledoc """
  Console handler for telemetry events.

  Logs events to the console with formatting.
  """

  require Logger

  @doc false
  def handle(event, measurements, metadata, _config) do
    handle_event(event, measurements, metadata)
  end

  defp handle_event([:crucible_kitchen, :workflow, :run, :start], _m, metadata) do
    Logger.info("[Kitchen] Workflow started: #{inspect(metadata[:workflow])}")
  end

  defp handle_event([:crucible_kitchen, :workflow, :run, :stop], measurements, _metadata) do
    duration = measurements[:duration] || 0
    Logger.info("[Kitchen] Workflow completed in #{format_duration(duration)}")
  end

  defp handle_event([:crucible_kitchen, :stage, :run, :start], _m, metadata) do
    Logger.debug("[Kitchen] Stage started: #{metadata[:stage]}")
  end

  defp handle_event([:crucible_kitchen, :stage, :run, :stop], measurements, metadata) do
    duration = measurements[:duration] || 0
    Logger.debug("[Kitchen] Stage #{metadata[:stage]} completed in #{format_duration(duration)}")
  end

  defp handle_event([:crucible_kitchen, :stage, :run, :exception], _m, metadata) do
    Logger.error("[Kitchen] Stage #{metadata[:stage]} failed: #{inspect(metadata[:error])}")
  end

  defp handle_event([:crucible_kitchen, :training, :step], measurements, metadata) do
    step = metadata[:step] || 0
    total = metadata[:total_steps] || "?"
    loss = measurements[:loss]
    lr = measurements[:lr]

    Logger.info(
      "[Kitchen] Step #{step}/#{total} | loss=#{format_float(loss)} | lr=#{format_float(lr, 8)}"
    )
  end

  defp handle_event([:crucible_kitchen, :training, :epoch], _m, metadata) do
    epoch = metadata[:epoch] || 0
    total = metadata[:total_epochs] || "?"
    Logger.info("[Kitchen] Epoch #{epoch + 1}/#{total} complete")
  end

  defp handle_event([:crucible_kitchen, :training, :checkpoint], _m, metadata) do
    name = metadata[:checkpoint_name] || "checkpoint"
    Logger.info("[Kitchen] Checkpoint saved: #{name}")
  end

  defp handle_event([:crucible_kitchen, :eval, :complete], measurements, metadata) do
    model = metadata[:model] || "model"
    accuracy = Map.get(measurements, :accuracy)
    f1 = Map.get(measurements, :f1)

    metrics_str =
      [
        if(accuracy, do: "acc=#{format_float(accuracy)}"),
        if(f1, do: "f1=#{format_float(f1)}")
      ]
      |> Enum.reject(&is_nil/1)
      |> Enum.join(" ")

    Logger.info("[Kitchen] Evaluation complete for #{model}: #{metrics_str}")
  end

  defp handle_event([:crucible_kitchen, :model, :registered], _m, metadata) do
    name = metadata[:name] || "model"
    version = metadata[:version] || "?"
    id = metadata[:model_id] || "?"
    Logger.info("[Kitchen] Model registered: #{name} v#{version} (id: #{id})")
  end

  defp handle_event(_event, _measurements, _metadata), do: :ok

  defp format_duration(nanoseconds) when is_integer(nanoseconds) do
    ms = nanoseconds / 1_000_000

    cond do
      ms < 1000 -> "#{Float.round(ms, 1)}ms"
      ms < 60_000 -> "#{Float.round(ms / 1000, 1)}s"
      true -> "#{Float.round(ms / 60_000, 1)}m"
    end
  end

  defp format_duration(_), do: "?"

  defp format_float(nil), do: "?"
  defp format_float(f, precision \\ 4)
  defp format_float(f, precision) when is_float(f), do: Float.round(f, precision)
  defp format_float(n, _precision), do: n
end

defmodule CrucibleKitchen.Telemetry.Handlers.JSONL do
  @moduledoc """
  JSONL file handler for telemetry events.

  Writes events as newline-delimited JSON to a file.
  """

  @doc false
  def handle(event, measurements, metadata, %{path: path}) do
    entry = %{
      timestamp: DateTime.utc_now() |> DateTime.to_iso8601(),
      event: Enum.join(event, "."),
      measurements: measurements,
      metadata: sanitize_metadata(metadata)
    }

    line = Jason.encode!(entry) <> "\n"

    File.write(path, line, [:append])
  end

  def handle(_event, _measurements, _metadata, _config), do: :ok

  defp sanitize_metadata(metadata) do
    metadata
    |> Enum.reject(fn {_k, v} -> is_function(v) or is_pid(v) or is_reference(v) end)
    |> Enum.into(%{})
  end
end
