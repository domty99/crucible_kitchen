defmodule CrucibleKitchen do
  @moduledoc """
  Industrial ML training orchestration - backend-agnostic workflow engine.

  CrucibleKitchen is the missing orchestration layer that makes ML cookbooks
  trivially thin. It provides:

  - **Pre-built workflows** for supervised, reinforcement, preference (DPO), and distillation training
  - **Declarative workflow DSL** for composing custom training pipelines
  - **Port/adapter pattern** for backend flexibility (Tinker, Fireworks, Modal, local Nx)
  - **Comprehensive telemetry** for observability and debugging
  - **First-class reproducibility** with deterministic seeding and artifact versioning

  ## Quick Start

      # Define adapters for your backend
      adapters = %{
        training_client: MyCookbook.Adapters.TrainingClient,
        dataset_store: CrucibleKitchen.Adapters.Noop.DatasetStore,
        blob_store: CrucibleKitchen.Adapters.Noop.BlobStore
      }

      # Run a built-in workflow
      {:ok, result} = CrucibleKitchen.run(:supervised, config, adapters: adapters)

  ## Architecture

  ```
  ┌─────────────────────────────────────────────────────────────────┐
  │                    COOKBOOK FRONTENDS                            │
  │  tinkex_cookbook    fireworks_cookbook    modal_cookbook        │
  │  (config + adapters only, <2K LOC each)                         │
  └─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │                     CRUCIBLE KITCHEN                             │
  │  Recipes → Workflows → Stages → Ports                           │
  │  (backend-agnostic orchestration)                               │
  └─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │                    BACKEND ADAPTERS                              │
  │  Tinker, Fireworks, Modal, LocalNx, Noop (testing)             │
  └─────────────────────────────────────────────────────────────────┘
  ```

  ## Key Concepts

  - **Recipe**: Configuration-driven training definition (what to train)
  - **Workflow**: Composition of stages with control flow (how to orchestrate)
  - **Stage**: Individual operation with lifecycle hooks (atomic unit of work)
  - **Port**: Behaviour contract for external integrations (abstraction)
  - **Adapter**: Port implementation for a specific backend (provided by cookbooks)

  ## Built-in Workflows

  - `:supervised` - Standard supervised fine-tuning (SFT)
  - `:reinforcement` - RL with rollouts and PPO (GRPO, etc.)
  - `:preference` - Direct Preference Optimization (DPO)
  - `:distillation` - Knowledge distillation (on-policy, multi-teacher)

  ## Documentation

  - [Getting Started](docs/guides/getting_started.md)
  - [Custom Workflows](docs/guides/custom_workflows.md)
  - [Writing Adapters](docs/guides/adapters.md)
  - [Telemetry](docs/guides/telemetry.md)
  """

  alias CrucibleKitchen.Context
  alias CrucibleKitchen.Workflow.Runner

  @version Mix.Project.config()[:version]

  @type config :: map()
  @type adapters :: Context.adapter_map()
  @type run_opts :: [
          adapters: adapters(),
          telemetry: boolean(),
          dry_run: boolean(),
          resume_from: String.t() | nil
        ]
  @type result :: %{
          context: Context.t(),
          metrics: [Context.metric()],
          artifacts: [artifact()],
          duration_ms: non_neg_integer()
        }
  @type artifact :: %{type: atom(), path: String.t(), metadata: map()}

  @builtin_workflows %{
    supervised: CrucibleKitchen.Workflows.Supervised,
    reinforcement: CrucibleKitchen.Workflows.Reinforcement,
    preference: CrucibleKitchen.Workflows.Preference,
    distillation: CrucibleKitchen.Workflows.Distillation
  }

  # ============================================================================
  # Core API
  # ============================================================================

  @doc """
  Execute a recipe, workflow, or built-in workflow name.

  ## Parameters

  - `target` - Recipe module, Workflow module, or built-in name
  - `config` - Configuration map
  - `opts` - Execution options

  ## Options

  - `:adapters` - Required. Map of port -> adapter module
  - `:telemetry` - Enable telemetry (default: true)
  - `:dry_run` - Validate without executing (default: false)
  - `:resume_from` - Checkpoint path to resume from

  ## Examples

      CrucibleKitchen.run(:supervised, %{model: "llama-3", epochs: 3}, adapters: adapters)

      CrucibleKitchen.run(MyRecipe, config, adapters: adapters)
  """
  @spec run(module() | atom(), config(), run_opts()) :: {:ok, result()} | {:error, term()}
  def run(target, config, opts \\ []) do
    adapters = Keyword.fetch!(opts, :adapters)
    telemetry_enabled = Keyword.get(opts, :telemetry, true)
    dry_run = Keyword.get(opts, :dry_run, false)

    with {:ok, workflow} <- resolve_workflow(target),
         {:ok, merged_config} <- merge_config(target, config),
         :ok <- validate_adapters_for_workflow(workflow, adapters) do
      if dry_run do
        {:ok, %{validated: true, workflow: workflow, config: merged_config}}
      else
        execute_workflow(workflow, merged_config, adapters, telemetry_enabled)
      end
    end
  end

  @doc """
  List all built-in workflows.
  """
  @spec workflows() :: [atom()]
  def workflows, do: Map.keys(@builtin_workflows)

  @doc """
  Get information about a workflow or recipe.
  """
  @spec describe(module() | atom()) :: {:ok, map()} | {:error, :not_found}
  def describe(target) do
    case resolve_workflow(target) do
      {:ok, workflow} ->
        {:ok,
         %{
           name: workflow_name(target),
           workflow: workflow,
           stages: workflow.__workflow__() |> extract_stage_names(),
           required_adapters: [:training_client, :dataset_store],
           optional_adapters: [:blob_store, :hub_client, :metrics_store]
         }}

      {:error, reason} ->
        {:error, reason}
    end
  end

  @doc """
  Validate configuration against a recipe.
  """
  @spec validate(module() | atom(), config()) :: :ok | {:error, [map()]}
  def validate(target, config) do
    case resolve_recipe(target) do
      {:ok, recipe} when is_atom(recipe) ->
        validate_config_with_schema(recipe, config)

      _ ->
        :ok
    end
  end

  defp validate_config_with_schema(recipe, config) do
    if function_exported?(recipe, :config_schema, 0) do
      schema = recipe.config_schema()
      if schema, do: schema.validate(config), else: :ok
    else
      :ok
    end
  end

  @doc """
  Attach default telemetry handlers.
  """
  @spec attach_telemetry(atom(), keyword()) :: :ok
  def attach_telemetry(handler \\ :console, opts \\ []) do
    CrucibleKitchen.Telemetry.attach(handler, opts)
  end

  @doc """
  Get the library version.
  """
  @spec version() :: String.t()
  def version, do: @version

  # ============================================================================
  # Private
  # ============================================================================

  defp resolve_workflow(name) when is_atom(name) do
    case Map.get(@builtin_workflows, name) do
      nil -> resolve_recipe_workflow(name)
      workflow -> {:ok, workflow}
    end
  end

  defp resolve_workflow(module) when is_atom(module) do
    cond do
      function_exported?(module, :__workflow__, 0) -> {:ok, module}
      function_exported?(module, :workflow, 0) -> {:ok, module.workflow()}
      true -> {:error, {:invalid_target, module}}
    end
  end

  defp resolve_recipe_workflow(name) do
    case resolve_recipe(name) do
      {:ok, recipe} -> {:ok, recipe.workflow()}
      error -> error
    end
  end

  defp resolve_recipe(module) when is_atom(module) do
    if function_exported?(module, :workflow, 0) do
      {:ok, module}
    else
      {:error, {:not_a_recipe, module}}
    end
  end

  defp merge_config(target, config) do
    case resolve_recipe(target) do
      {:ok, recipe} ->
        if function_exported?(recipe, :defaults, 0) do
          {:ok, Map.merge(recipe.defaults(), config)}
        else
          {:ok, config}
        end

      _ ->
        {:ok, config}
    end
  end

  defp validate_adapters_for_workflow(_workflow, adapters) do
    required = [:training_client, :dataset_store]
    missing = Enum.filter(required, fn port -> not Map.has_key?(adapters, port) end)

    if missing == [] do
      :ok
    else
      {:error, {:missing_adapters, missing}}
    end
  end

  defp execute_workflow(workflow, config, adapters, telemetry_enabled) do
    started_at = System.monotonic_time(:millisecond)

    context = Context.new(config, adapters)

    result =
      if telemetry_enabled do
        :telemetry.span(
          [:crucible_kitchen, :workflow, :run],
          %{workflow: workflow, config: config},
          fn ->
            result = Runner.run(workflow, context)
            {result, %{}}
          end
        )
      else
        Runner.run(workflow, context)
      end

    duration_ms = System.monotonic_time(:millisecond) - started_at

    case result do
      {:ok, final_context} ->
        {:ok,
         %{
           context: final_context,
           metrics: final_context.metrics,
           artifacts: Map.get(final_context.state, :artifacts, []),
           duration_ms: duration_ms
         }}

      {:error, reason} ->
        {:error, reason}
    end
  end

  defp workflow_name(name) when is_atom(name), do: name

  defp extract_stage_names(workflow_ir) do
    workflow_ir
    |> Enum.flat_map(fn
      {:stage, name, _, _} -> [name]
      {:loop, _, _, body} -> extract_stage_names(body)
      {:conditional, _, body} -> extract_stage_names(body)
      {:parallel, _, body} -> extract_stage_names(body)
      _ -> []
    end)
  end
end
