# Component Design

**Purpose:** Define the core components of crucible_kitchen and their responsibilities.

---

## Component Overview

```
crucible_kitchen/
├── lib/
│   ├── crucible_kitchen.ex              # Public API facade
│   ├── crucible_kitchen/
│   │   ├── application.ex               # OTP Application
│   │   ├── supervisor.ex                # Top-level supervisor
│   │   │
│   │   ├── recipe/                      # Recipe Layer
│   │   │   ├── recipe.ex                # Recipe behaviour
│   │   │   ├── config.ex                # Config normalization
│   │   │   └── registry.ex              # Recipe lookup
│   │   │
│   │   ├── workflow/                    # Workflow Layer
│   │   │   ├── workflow.ex              # Workflow DSL macro
│   │   │   ├── builder.ex               # Workflow compilation
│   │   │   ├── runner.ex                # Workflow execution
│   │   │   └── builtins/                # Built-in workflows
│   │   │       ├── supervised.ex        # SL workflow
│   │   │       ├── reinforcement.ex     # RL workflow
│   │   │       ├── preference.ex        # DPO workflow
│   │   │       └── distillation.ex      # Distillation workflow
│   │   │
│   │   ├── stage/                       # Stage Layer
│   │   │   ├── stage.ex                 # Stage behaviour
│   │   │   ├── context.ex               # Flowing context
│   │   │   └── builtins/                # Built-in stages
│   │   │       ├── load_dataset.ex
│   │   │       ├── init_session.ex
│   │   │       ├── render_batch.ex
│   │   │       ├── forward_backward.ex
│   │   │       ├── optim_step.ex
│   │   │       ├── checkpoint.ex
│   │   │       ├── evaluate.ex
│   │   │       └── save_weights.ex
│   │   │
│   │   ├── ports/                       # Port Layer
│   │   │   ├── training_client.ex       # Training backend
│   │   │   ├── dataset_store.ex         # Dataset loading
│   │   │   ├── blob_store.ex            # Artifact storage
│   │   │   ├── hub_client.ex            # Model hub
│   │   │   ├── metrics_store.ex         # Metrics persistence
│   │   │   ├── vector_store.ex          # Embeddings
│   │   │   └── noop/                    # Noop implementations
│   │   │       └── *.ex
│   │   │
│   │   ├── telemetry/                   # Telemetry Layer
│   │   │   ├── events.ex                # Event definitions
│   │   │   ├── handlers.ex              # Default handlers
│   │   │   └── formatters/              # Output formatters
│   │   │       ├── jsonl.ex
│   │   │       ├── console.ex
│   │   │       └── prometheus.ex
│   │   │
│   │   └── utils/                       # Utilities
│   │       ├── lr_schedule.ex           # Learning rate computation
│   │       ├── seed.ex                  # Deterministic seeding
│   │       ├── colorize.ex              # Token visualization
│   │       └── timing.ex                # Performance measurement
│   │
│   └── mix/tasks/                       # Mix Tasks
│       └── kitchen.run.ex               # CLI entrypoint
```

---

## Core Components

### 1. CrucibleKitchen (Facade)

The single public entrypoint for all operations.

```elixir
defmodule CrucibleKitchen do
  @moduledoc """
  Industrial ML training orchestration.

  ## Usage

      # Run a recipe with adapters
      CrucibleKitchen.run(:supervised_training, config,
        adapters: %{
          training_client: {CrucibleKitchen.Adapters.Tinkex.TrainingClient, []},
          dataset_store: {CrucibleKitchen.Adapters.HfDatasets.DatasetStore, []}
        })

      # Run with custom workflow
      CrucibleKitchen.run(MyWorkflow, config,
        adapters: MyAdapters)
  """

  @doc "Execute a recipe or workflow with given config and adapters."
  @spec run(atom() | module(), map(), keyword()) :: {:ok, result()} | {:error, term()}
  def run(recipe_or_workflow, config, opts \\ [])

  @doc "List available built-in recipes."
  @spec recipes() :: [atom()]
  def recipes()

  @doc "Get recipe metadata."
  @spec describe(atom()) :: {:ok, Recipe.t()} | {:error, :not_found}
  def describe(recipe)

  @doc "Validate config against recipe schema."
  @spec validate(atom(), map()) :: :ok | {:error, [validation_error()]}
  def validate(recipe, config)
end
```

### 2. Recipe

Configuration-driven recipe definitions.

```elixir
defmodule CrucibleKitchen.Recipe do
  @moduledoc """
  Behaviour for recipe definitions.

  Recipes are declarative - they describe WHAT should happen,
  not HOW. The kitchen handles the HOW.
  """

  @type t :: %__MODULE__{
    name: atom(),
    description: String.t(),
    workflow: module(),
    config_schema: module() | nil,
    defaults: map()
  }

  @callback name() :: atom()
  @callback description() :: String.t()
  @callback workflow() :: module()
  @callback config_schema() :: module() | nil
  @callback defaults() :: map()

  defmacro __using__(_opts) do
    quote do
      @behaviour CrucibleKitchen.Recipe

      def config_schema, do: nil
      def defaults, do: %{}

      defoverridable [config_schema: 0, defaults: 0]
    end
  end
end

# Example recipe (in cookbook, not kitchen):
defmodule TinkexCookbook.Recipes.SlBasic do
  use CrucibleKitchen.Recipe

  def name, do: :sl_basic
  def description, do: "Basic supervised fine-tuning"
  def workflow, do: CrucibleKitchen.Workflows.Supervised
  def config_schema, do: TinkexCookbook.Recipes.SlBasicConfig
  def defaults do
    %{
      model: "meta-llama/Llama-3.1-8B",
      epochs: 1,
      batch_size: 128,
      learning_rate: 2.0e-4
    }
  end
end
```

### 3. Workflow

Composable workflow definitions using a DSL.

```elixir
defmodule CrucibleKitchen.Workflow do
  @moduledoc """
  DSL for defining training workflows.

  Workflows are compositions of stages with control flow.
  """

  defmacro __using__(_opts) do
    quote do
      import CrucibleKitchen.Workflow.DSL

      Module.register_attribute(__MODULE__, :workflow_stages, accumulate: true)

      @before_compile CrucibleKitchen.Workflow
    end
  end

  defmacro __before_compile__(_env) do
    quote do
      def __workflow__ do
        @workflow_stages |> Enum.reverse()
      end
    end
  end
end

defmodule CrucibleKitchen.Workflow.DSL do
  @moduledoc "DSL macros for workflow definition."

  defmacro stage(name, module, opts \\ []) do
    quote do
      @workflow_stages {:stage, unquote(name), unquote(module), unquote(opts)}
    end
  end

  defmacro loop(name, opts, do: block) do
    quote do
      @workflow_stages {:loop_start, unquote(name), unquote(opts)}
      unquote(block)
      @workflow_stages {:loop_end, unquote(name)}
    end
  end

  defmacro parallel(opts \\ [], do: block) do
    quote do
      @workflow_stages {:parallel_start, unquote(opts)}
      unquote(block)
      @workflow_stages {:parallel_end}
    end
  end

  defmacro conditional(condition, do: block) do
    quote do
      @workflow_stages {:conditional_start, unquote(condition)}
      unquote(block)
      @workflow_stages {:conditional_end}
    end
  end
end
```

### 4. Stage

Individual operations with lifecycle hooks.

```elixir
defmodule CrucibleKitchen.Stage do
  @moduledoc """
  Behaviour for stage implementations.

  Stages are the atomic units of work in a workflow.
  They receive context, perform work, and return updated context.
  """

  @type context :: CrucibleKitchen.Context.t()
  @type result :: {:ok, context()} | {:error, term()}

  @callback name() :: atom()
  @callback execute(context()) :: result()

  # Optional callbacks
  @callback validate(context()) :: :ok | {:error, term()}
  @callback rollback(context(), error :: term()) :: context()

  @optional_callbacks [validate: 1, rollback: 2]

  defmacro __using__(_opts) do
    quote do
      @behaviour CrucibleKitchen.Stage

      import CrucibleKitchen.Stage.Helpers

      def validate(_context), do: :ok
      def rollback(context, _error), do: context

      defoverridable [validate: 1, rollback: 2]
    end
  end
end

# Example built-in stage:
defmodule CrucibleKitchen.Stages.ForwardBackward do
  use CrucibleKitchen.Stage

  alias CrucibleTrain.Ports.TrainingClient

  def name, do: :forward_backward

  def execute(%{state: state} = context) do
    ports = CrucibleKitchen.Context.get_train_ports(context)
    batch = state.current_batch

    future = TrainingClient.forward_backward(ports, state.session, batch)

    case TrainingClient.await(ports, future) do
      {:ok, result} ->
        {:ok, update_in(context, [:state, :last_fb_result], fn _ -> result end)}

      {:error, reason} ->
        {:error, reason}
    end
  end
end
```

### 5. Context

The flowing state container.

```elixir
defmodule CrucibleKitchen.Context do
  @moduledoc """
  Context flows through the workflow, accumulating state.

  Immutable - each stage returns a new context.
  """

  defstruct [
    :recipe,
    :workflow,
    :config,
    :adapters,
    :state,
    :metrics,
    :metadata,
    :started_at,
    :current_stage
  ]

  @type t :: %__MODULE__{
    recipe: atom(),
    workflow: module(),
    config: map(),
    adapters: adapter_map(),
    state: map(),
    metrics: [metric()],
    metadata: map(),
    started_at: DateTime.t(),
    current_stage: atom() | nil
  }

  @type adapter_map :: %{
    training_client: module(),
    dataset_store: module(),
    blob_store: module(),
    optional(:hub_client) => module(),
    optional(:vector_store) => module(),
    optional(:metrics_store) => module()
  }

  @type metric :: %{
    name: atom(),
    value: number(),
    step: non_neg_integer(),
    timestamp: DateTime.t()
  }

  def new(recipe, config, adapters) do
    %__MODULE__{
      recipe: recipe,
      workflow: recipe.workflow(),
      config: Map.merge(recipe.defaults(), config),
      adapters: adapters,
      state: %{},
      metrics: [],
      metadata: %{},
      started_at: DateTime.utc_now(),
      current_stage: nil
    }
  end

  def put_state(context, key, value) do
    update_in(context, [:state, key], fn _ -> value end)
  end

  def get_state(context, key, default \\ nil) do
    get_in(context, [:state, key]) || default
  end

  def record_metric(context, name, value, step) do
    metric = %{
      name: name,
      value: value,
      step: step,
      timestamp: DateTime.utc_now()
    }
    update_in(context, [:metrics], &[metric | &1])
  end
end
```

### 6. Ports

Behaviour contracts for external integrations.

```elixir
defmodule CrucibleTrain.Ports.TrainingClient do
  @moduledoc """
  Port for ML training backends.

  Implementations: Tinkex, Fireworks, Modal, LocalNx, Noop
  """

  @type adapter_opts :: keyword()
  @type session :: term()
  @type future :: term()
  @type datum :: CrucibleTrain.Types.Datum.t()

  @callback start_session(adapter_opts(), config :: map()) :: {:ok, session()} | {:error, term()}
  @callback forward_backward(adapter_opts(), session(), [datum()]) :: future()
  @callback optim_step(adapter_opts(), session(), learning_rate :: float()) :: future()
  @callback await(adapter_opts(), future()) :: {:ok, map()} | {:error, term()}
  @callback save_checkpoint(adapter_opts(), session(), path :: String.t()) :: :ok | {:error, term()}
  @callback load_checkpoint(adapter_opts(), session(), path :: String.t()) :: :ok | {:error, term()}
  @callback close_session(adapter_opts(), session()) :: :ok
end

defmodule CrucibleTrain.Ports.DatasetStore do
  @moduledoc """
  Port for dataset loading and common dataset operations.

  Implementations: HfDatasets, LocalFiles, Noop
  """

  @type adapter_opts :: keyword()
  @type dataset :: term()

  @callback load_dataset(adapter_opts(), String.t(), keyword()) ::
              {:ok, dataset()} | {:error, term()}
  @callback get_split(adapter_opts(), dataset(), String.t() | atom()) ::
              {:ok, dataset()} | {:error, term()}
  @callback shuffle(adapter_opts(), dataset(), keyword()) ::
              {:ok, dataset()} | {:error, term()}
  @callback take(adapter_opts(), dataset(), non_neg_integer()) ::
              {:ok, dataset()} | {:error, term()}
  @callback skip(adapter_opts(), dataset(), non_neg_integer()) ::
              {:ok, dataset()} | {:error, term()}
  @callback select(adapter_opts(), dataset(), Range.t() | [non_neg_integer()]) ::
              {:ok, dataset()} | {:error, term()}
  @callback to_list(adapter_opts(), dataset()) :: {:ok, [map()]} | {:error, term()}
end

defmodule CrucibleTrain.Ports.BlobStore do
  @moduledoc """
  Port for file/blob access (local or remote).

  Implementations: Local, S3, GCS, Noop
  """

  @type adapter_opts :: keyword()
  @type path :: String.t()

  @callback read(adapter_opts(), path()) :: {:ok, binary()} | {:error, term()}
  @callback stream(adapter_opts(), path()) :: {:ok, Enumerable.t()} | {:error, term()}
  @callback write(adapter_opts(), path(), iodata()) :: :ok | {:error, term()}
  @callback exists?(adapter_opts(), path()) :: boolean()
end

defmodule CrucibleTrain.Ports.HubClient do
  @moduledoc """
  Port for model hub operations.

  Implementations: HfHub, Noop
  """

  @type adapter_opts :: keyword()

  @callback download(adapter_opts(), keyword()) :: {:ok, Path.t()} | {:error, term()}
  @callback snapshot(adapter_opts(), keyword()) :: {:ok, Path.t()} | {:error, term()}
  @callback list_files(adapter_opts(), String.t(), keyword()) ::
              {:ok, [String.t()]} | {:error, term()}
end

defmodule CrucibleTelemetry.Ports.MetricsStore do
  @moduledoc """
  Port for metrics persistence and querying.

  Implementations: JSONL, WandB, Neptune, Noop
  """

  @type adapter_opts :: keyword()
  @type run_id :: String.t()
  @type metric_name :: String.t() | atom()
  @type value :: number()

  @callback record(adapter_opts(), run_id(), metric_name(), value(), keyword()) ::
              :ok | {:error, term()}
  @callback flush(adapter_opts(), run_id()) :: :ok | {:error, term()}
  @callback read(adapter_opts(), run_id()) :: {:ok, [map()]} | {:error, term()}
end
```

### 7. Telemetry

Comprehensive observability.

```elixir
defmodule CrucibleKitchen.Telemetry do
  @moduledoc """
  Telemetry event definitions for crucible_kitchen.

  All events are prefixed with [:crucible_kitchen, ...].
  """

  @events [
    # Workflow lifecycle
    [:crucible_kitchen, :workflow, :start],
    [:crucible_kitchen, :workflow, :stop],
    [:crucible_kitchen, :workflow, :exception],

    # Stage lifecycle
    [:crucible_kitchen, :stage, :start],
    [:crucible_kitchen, :stage, :stop],
    [:crucible_kitchen, :stage, :exception],

    # Training metrics
    [:crucible_kitchen, :training, :step],
    [:crucible_kitchen, :training, :epoch],
    [:crucible_kitchen, :training, :checkpoint],

    # Evaluation
    [:crucible_kitchen, :eval, :start],
    [:crucible_kitchen, :eval, :stop],
    [:crucible_kitchen, :eval, :result]
  ]

  def events, do: @events

  def attach_default_handlers do
    :telemetry.attach_many(
      "crucible-kitchen-default",
      @events,
      &CrucibleKitchen.Telemetry.Handlers.Console.handle/4,
      nil
    )
  end
end

defmodule CrucibleKitchen.Telemetry.Handlers.Console do
  @moduledoc "Console logging handler for telemetry events."

  require Logger

  def handle([:crucible_kitchen, :training, :step], measurements, metadata, _config) do
    Logger.info(
      "Step #{metadata.step}/#{metadata.total_steps} | " <>
      "loss=#{Float.round(measurements.loss, 4)} | " <>
      "lr=#{Float.round(measurements.lr, 8)}"
    )
  end

  def handle([:crucible_kitchen, :stage, :stop], measurements, metadata, _config) do
    Logger.debug("Stage #{metadata.stage} completed in #{measurements.duration}ms")
  end

  # ... other handlers
end
```

---

## Component Interactions

```
User calls:
  CrucibleKitchen.run(:sl_basic, config, adapters: TinkexAdapters)
    │
    ▼
Recipe.Registry.get(:sl_basic)
    │ Returns SlBasicRecipe
    ▼
Context.new(recipe, config, adapters)
    │ Creates initial context
    ▼
Workflow.Runner.run(recipe.workflow(), context)
    │
    ▼
For each stage in workflow:
    │
    ├── :telemetry.execute([:stage, :start], ...)
    │
    ├── Stage.validate(context)
    │     │
    │     ▼ (if validation passes)
    │
    ├── Stage.execute(context)
    │     │ Uses adapters from context
    │     ▼
    │   Returns {:ok, new_context} or {:error, reason}
    │
    ├── :telemetry.execute([:stage, :stop], ...)
    │
    └── Continue to next stage with new_context
          │
          ▼
Final context with all accumulated state/metrics
```

---

## Dependency Flow

```
crucible_kitchen depends on:
├── crucible_train (Types, Renderers - NOT training loops)
├── crucible_ir (Experiment specs for interop)
├── crucible_framework (Optional - for pipeline interop)
├── telemetry (Observability)
└── jason (Serialization)

crucible_kitchen does NOT depend on:
├── tinkex (Tinker SDK - that's an adapter)
├── hf_datasets_ex (HuggingFace - that's an adapter)
├── any specific ML backend
└── any specific storage backend
```

This ensures the kitchen remains truly backend-agnostic.
