# Public API Surface

**Purpose:** Define the public API that cookbook frontends and users interact with.

---

## Design Philosophy

1. **Small surface area** - Few public functions, well-documented
2. **Progressive disclosure** - Simple things easy, complex things possible
3. **Sensible defaults** - Works out of the box, customizable when needed
4. **Explicit over implicit** - Adapters must be provided, not inferred

---

## Primary API: CrucibleKitchen

```elixir
defmodule CrucibleKitchen do
  @moduledoc """
  Industrial ML training orchestration.

  CrucibleKitchen is the backend-agnostic core for ML training workflows.
  It provides:

  - Pre-built workflows for SL, RL, DPO, and distillation
  - A DSL for defining custom workflows
  - Port/adapter pattern for backend flexibility
  - Comprehensive telemetry for observability

  ## Quick Start

      # Define adapters for your backend
      adapters = %{
        training_client: MyBackend.TrainingClient,
        dataset_store: CrucibleKitchen.Adapters.HfDatasets,
        blob_store: CrucibleKitchen.Adapters.LocalBlobStore
      }

      # Run a built-in workflow
      {:ok, result} = CrucibleKitchen.run(:supervised, config, adapters: adapters)

  ## With Recipes

      # Recipes bundle workflow + config schema + defaults
      {:ok, result} = CrucibleKitchen.run(MyRecipe, config, adapters: adapters)

  ## Custom Workflows

      defmodule MyWorkflow do
        use CrucibleKitchen.Workflow

        workflow do
          stage :load, LoadStage
          stage :train, TrainStage
          stage :save, SaveStage
        end
      end

      {:ok, result} = CrucibleKitchen.run(MyWorkflow, config, adapters: adapters)
  """

  @type config :: map()
  @type adapters :: CrucibleKitchen.Context.adapter_map()
  @type run_opts :: [
    adapters: adapters(),
    telemetry: boolean(),
    dry_run: boolean(),
    resume_from: String.t() | nil
  ]
  @type result :: %{
    context: CrucibleKitchen.Context.t(),
    metrics: [CrucibleKitchen.Context.metric()],
    artifacts: [artifact()],
    duration_ms: non_neg_integer()
  }
  @type artifact :: %{type: atom(), path: String.t(), metadata: map()}

  # ============================================================================
  # Core Execution
  # ============================================================================

  @doc """
  Execute a recipe, workflow, or built-in workflow name.

  ## Parameters

  - `target` - Recipe module, Workflow module, or built-in name (`:supervised`, `:reinforcement`, `:preference`, `:distillation`)
  - `config` - Configuration map (merged with recipe defaults if applicable)
  - `opts` - Execution options

  ## Options

  - `:adapters` - Required. Map of port -> adapter module
  - `:telemetry` - Enable telemetry (default: true)
  - `:dry_run` - Validate without executing (default: false)
  - `:resume_from` - Checkpoint path to resume from (default: nil)

  ## Returns

  - `{:ok, result}` on success
  - `{:error, reason}` on failure

  ## Examples

      # Built-in workflow
      CrucibleKitchen.run(:supervised, %{model: "llama-3", epochs: 3}, adapters: adapters)

      # Custom recipe
      CrucibleKitchen.run(MyRecipe, %{custom_param: 42}, adapters: adapters)

      # Resume from checkpoint
      CrucibleKitchen.run(:supervised, config,
        adapters: adapters,
        resume_from: "/checkpoints/step_1000")
  """
  @spec run(module() | atom(), config(), run_opts()) :: {:ok, result()} | {:error, term()}
  def run(target, config, opts)

  @doc """
  Execute with streaming progress updates.

  Same as `run/3` but yields progress events to the caller.

  ## Examples

      CrucibleKitchen.run_stream(:supervised, config, adapters: adapters)
      |> Stream.each(fn
        {:progress, %{step: step, total: total}} ->
          IO.puts("Step \#{step}/\#{total}")
        {:metric, name, value} ->
          IO.puts("\#{name}: \#{value}")
        {:complete, result} ->
          IO.puts("Done!")
      end)
      |> Stream.run()
  """
  @spec run_stream(module() | atom(), config(), run_opts()) :: Enumerable.t()
  def run_stream(target, config, opts)

  # ============================================================================
  # Discovery & Validation
  # ============================================================================

  @doc """
  List all built-in workflows.

  ## Examples

      CrucibleKitchen.workflows()
      #=> [:supervised, :reinforcement, :preference, :distillation]
  """
  @spec workflows() :: [atom()]
  def workflows()

  @doc """
  Get information about a workflow or recipe.

  ## Examples

      CrucibleKitchen.describe(:supervised)
      #=> %{
      #=>   name: :supervised,
      #=>   description: "Standard supervised learning workflow",
      #=>   stages: [:load_dataset, :init_session, ...],
      #=>   required_adapters: [:training_client, :dataset_store],
      #=>   optional_adapters: [:metrics_store, :blob_store]
      #=> }
  """
  @spec describe(module() | atom()) :: {:ok, map()} | {:error, :not_found}
  def describe(target)

  @doc """
  Validate configuration against a recipe or workflow.

  ## Examples

      CrucibleKitchen.validate(MyRecipe, %{model: "llama-3"})
      #=> :ok

      CrucibleKitchen.validate(MyRecipe, %{model: 123})
      #=> {:error, [%{field: :model, message: "must be a string"}]}
  """
  @spec validate(module() | atom(), config()) :: :ok | {:error, [validation_error()]}
  def validate(target, config)

  @doc """
  Validate that adapters implement required ports.

  ## Examples

      CrucibleKitchen.validate_adapters(:supervised, adapters)
      #=> :ok

      CrucibleKitchen.validate_adapters(:supervised, %{training_client: BadAdapter})
      #=> {:error, [%{adapter: BadAdapter, missing: [:forward_backward, :optim_step]}]}
  """
  @spec validate_adapters(module() | atom(), adapters()) :: :ok | {:error, [adapter_error()]}
  def validate_adapters(target, adapters)

  # ============================================================================
  # Telemetry
  # ============================================================================

  @doc """
  Attach default telemetry handlers.

  Call this in your application startup to enable console logging.

  ## Examples

      CrucibleKitchen.attach_telemetry()
      CrucibleKitchen.attach_telemetry(:jsonl, path: "/var/log/kitchen.jsonl")
      CrucibleKitchen.attach_telemetry(:prometheus, port: 9090)
  """
  @spec attach_telemetry() :: :ok
  @spec attach_telemetry(atom(), keyword()) :: :ok
  def attach_telemetry(handler \\ :console, opts \\ [])

  @doc """
  List all telemetry events emitted by CrucibleKitchen.

  Useful for custom telemetry handler implementations.
  """
  @spec telemetry_events() :: [list()]
  def telemetry_events()
end
```

---

## Adapter Registration API

For apps that provide adapters:

```elixir
defmodule CrucibleKitchen.Adapters do
  @moduledoc """
  Adapter utilities and noop implementations.

  ## Providing Adapters

  Adapters are modules that implement port behaviours. Package them
  as a map for use with `CrucibleKitchen.run/3`:

      defmodule MyCookbook.Adapters do
        def all do
          %{
            training_client: MyCookbook.Adapters.TrainingClient,
            dataset_store: MyCookbook.Adapters.DatasetStore,
            blob_store: CrucibleKitchen.Adapters.Noop.BlobStore
          }
        end
      end

      CrucibleKitchen.run(:supervised, config, adapters: MyCookbook.Adapters.all())

  ## Testing with Noop Adapters

      CrucibleKitchen.run(:supervised, config, adapters: CrucibleKitchen.Adapters.noop())
  """

  @doc "Returns a map of all noop adapters for testing."
  @spec noop() :: map()
  def noop()

  @doc "Validates that an adapter module implements the required port behaviour."
  @spec implements?(module(), module()) :: boolean()
  def implements?(adapter_module, port_behaviour)

  @doc "Lists all callbacks an adapter is missing from a port behaviour."
  @spec missing_callbacks(module(), module()) :: [atom()]
  def missing_callbacks(adapter_module, port_behaviour)
end
```

---

## Stage API (For Custom Stages)

```elixir
defmodule CrucibleKitchen.Stage do
  @moduledoc """
  Behaviour for defining custom stages.

  ## Example

      defmodule MyStage do
        use CrucibleKitchen.Stage

        @impl true
        def name, do: :my_stage

        @impl true
        def execute(context) do
          # Do work...
          {:ok, Context.put_state(context, :my_result, result)}
        end
      end

  ## Context Helpers

  The `use CrucibleKitchen.Stage` macro imports helpers:

  - `get_config/2` - Get config value with default
  - `get_state/2` - Get state value with default
  - `put_state/3` - Update state
  - `get_adapter/2` - Get adapter from context
  - `emit_metric/4` - Record a metric
  - `log_info/2` - Log with context metadata

  ## Validation

  Implement `validate/1` to check preconditions:

      @impl true
      def validate(context) do
        if context.state[:session] do
          :ok
        else
          {:error, "session not initialized"}
        end
      end

  ## Rollback

  Implement `rollback/2` to clean up on failure:

      @impl true
      def rollback(context, _error) do
        # Clean up any partial state
        context
      end
  """

  @type context :: CrucibleKitchen.Context.t()
  @type result :: {:ok, context()} | {:error, term()}

  @callback name() :: atom()
  @callback execute(context()) :: result()
  @callback validate(context()) :: :ok | {:error, term()}
  @callback rollback(context(), error :: term()) :: context()

  @optional_callbacks [validate: 1, rollback: 2]
end
```

---

## Workflow DSL API

```elixir
defmodule CrucibleKitchen.Workflow do
  @moduledoc """
  DSL for defining custom workflows.

  ## Example

      defmodule MyWorkflow do
        use CrucibleKitchen.Workflow

        workflow do
          stage :load, LoadStage
          stage :process, ProcessStage, timeout: 30_000

          loop :iterations, over: fn ctx -> 1..ctx.config.num_iterations end do
            stage :iterate, IterateStage
          end

          conditional fn ctx -> ctx.config.save_result end do
            stage :save, SaveStage
          end

          parallel max_concurrency: 4 do
            stage :eval_a, EvalAStage
            stage :eval_b, EvalBStage
          end
        end
      end

  ## DSL Reference

  ### stage(name, module, opts)

  Define a stage. Options:
  - `:timeout` - Max execution time in ms
  - `:when` - Conditional execution (deprecated, use `conditional`)
  - Any other options passed to the stage module

  ### loop(name, opts, do: block)

  Iterate over a collection. Options:
  - `:over` - Function `(context) -> Enumerable.t()`

  ### conditional(predicate, do: block)

  Execute block only if predicate returns true.
  - `predicate` - Function `(context) -> boolean()`

  ### parallel(opts, do: block)

  Execute stages in parallel. Options:
  - `:max_concurrency` - Max concurrent stages (default: schedulers_online)
  - `:on_error` - `:halt` (default) or `:continue`
  """
end
```

---

## Config Schema API (ChzEx Integration)

```elixir
defmodule CrucibleKitchen.Config do
  @moduledoc """
  Configuration schema utilities.

  Integrates with ChzEx for config validation and CLI binding.

  ## Example

      defmodule MyRecipe.Config do
        use CrucibleKitchen.Config

        config_schema do
          field :model, :string, required: true
          field :epochs, :integer, default: 1, min: 1
          field :learning_rate, :float, default: 2.0e-4, min: 0.0
          field :train_on, :string,
            default: "all_assistant_messages",
            one_of: ["all_assistant_messages", "last_assistant_message", "all_tokens"]
        end
      end

      MyRecipe.Config.validate(%{model: "llama-3"})
      #=> {:ok, %{model: "llama-3", epochs: 1, learning_rate: 2.0e-4, train_on: "all_assistant_messages"}}

      MyRecipe.Config.from_argv(["--model", "llama-3", "--epochs", "3"])
      #=> {:ok, %{model: "llama-3", epochs: 3, ...}}
  """

  defmacro __using__(_opts) do
    quote do
      import CrucibleKitchen.Config.DSL
      Module.register_attribute(__MODULE__, :config_fields, accumulate: true)
      @before_compile CrucibleKitchen.Config
    end
  end
end
```

---

## CLI API

```elixir
# mix kitchen.run - Run a workflow or recipe
# mix kitchen.run supervised --model llama-3 --epochs 3 --adapters tinkex
# mix kitchen.run MyRecipe --config ./config.json

# mix kitchen.list - List available workflows
# mix kitchen.describe supervised - Show workflow details
# mix kitchen.validate MyRecipe --config ./config.json - Validate config
```

---

## API Versioning

The public API is versioned:

```elixir
defmodule CrucibleKitchen do
  @version "0.1.0"
  @api_version 1

  def version, do: @version
  def api_version, do: @api_version
end
```

Breaking changes increment `@api_version`. The `CHANGELOG.md` documents migration paths.

---

## Error Types

```elixir
defmodule CrucibleKitchen.Error do
  @moduledoc """
  Error types returned by CrucibleKitchen.
  """

  defmodule ValidationError do
    defexception [:field, :message, :value]
  end

  defmodule AdapterError do
    defexception [:adapter, :port, :missing_callbacks]
  end

  defmodule StageError do
    defexception [:stage, :reason, :context]
  end

  defmodule WorkflowError do
    defexception [:workflow, :stage, :reason]
  end

  defmodule ConfigError do
    defexception [:field, :message]
  end
end
```

This API surface provides:
1. **Simple entry point** - `CrucibleKitchen.run/3` does it all
2. **Discovery** - Introspect workflows and recipes
3. **Validation** - Check configs and adapters before running
4. **Extensibility** - Clean DSLs for custom stages and workflows
5. **Observability** - Built-in telemetry with easy attachment
