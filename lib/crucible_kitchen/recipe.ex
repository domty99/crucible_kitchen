defmodule CrucibleKitchen.Recipe do
  @moduledoc """
  Behaviour for defining training recipes.

  A Recipe is a high-level abstraction that combines:
  - A Workflow (the sequence of stages)
  - Default configuration
  - Default adapter requirements
  - Validation logic

  Recipes allow users to get started quickly with proven training patterns
  while maintaining full customizability.

  ## Implementing a Recipe

      defmodule MyApp.Recipes.SupervisedFineTuning do
        use CrucibleKitchen.Recipe

        @impl true
        def name, do: :supervised_finetuning

        @impl true
        def description do
          "Supervised fine-tuning for instruction-following models"
        end

        @impl true
        def default_config do
          %{
            model: nil,  # Required - must be provided
            epochs: 1,
            batch_size: 4,
            learning_rate: 2.0e-5,
            warmup_steps: 100,
            lora_rank: 16
          }
        end

        @impl true
        def required_adapters do
          [:training_client, :dataset_store]
        end

        @impl true
        def optional_adapters do
          [:metrics_store, :blob_store, :hub_client]
        end

        @impl true
        def workflow do
          CrucibleKitchen.Workflow.new(__MODULE__)
          |> CrucibleKitchen.Workflow.stage(:load_dataset, LoadDataset)
          |> CrucibleKitchen.Workflow.stage(:init_session, InitSession)
          |> CrucibleKitchen.Workflow.loop(:training, over: :epochs_range) do
            _workflow
            |> CrucibleKitchen.Workflow.stage(:train_epoch, TrainEpoch)
            |> CrucibleKitchen.Workflow.stage(:eval_epoch, EvalEpoch)
            |> CrucibleKitchen.Workflow.stage(:checkpoint, SaveCheckpoint)
          end
          |> CrucibleKitchen.Workflow.stage(:finalize, Finalize)
        end

        @impl true
        def validate_config(config) do
          cond do
            is_nil(config[:model]) ->
              {:error, "model is required"}

            config[:epochs] < 1 ->
              {:error, "epochs must be >= 1"}

            true ->
              :ok
          end
        end
      end

  ## Using a Recipe

      # Use with defaults
      {:ok, result} = CrucibleKitchen.run(MyApp.Recipes.SupervisedFineTuning,
        %{model: "meta-llama/Llama-2-7b", dataset: "my_dataset"},
        adapters: my_adapters
      )

      # Override defaults
      {:ok, result} = CrucibleKitchen.run(MyApp.Recipes.SupervisedFineTuning,
        %{model: "meta-llama/Llama-2-7b", epochs: 3, batch_size: 8},
        adapters: my_adapters
      )

  ## Built-in Recipes

  CrucibleKitchen provides these recipes out of the box:

  - `CrucibleKitchen.Recipes.SupervisedFineTuning` - Instruction tuning
  - `CrucibleKitchen.Recipes.RewardModeling` - RLHF reward model training
  - `CrucibleKitchen.Recipes.DPO` - Direct Preference Optimization
  - `CrucibleKitchen.Recipes.Evaluation` - Model evaluation runs
  """

  alias CrucibleKitchen.{Context, Workflow}

  @type config :: map()
  @type adapter_name :: atom()

  @doc """
  Returns the recipe name as an atom.
  """
  @callback name() :: atom()

  @doc """
  Returns a human-readable description of what this recipe does.
  """
  @callback description() :: String.t()

  @doc """
  Returns the default configuration.

  These values are merged with user-provided config, with user values taking precedence.
  """
  @callback default_config() :: config()

  @doc """
  Returns the list of required adapter ports.

  The recipe will fail to run if any of these adapters are missing.
  """
  @callback required_adapters() :: [adapter_name()]

  @doc """
  Returns the list of optional adapter ports.

  These adapters enhance functionality but aren't required.
  """
  @callback optional_adapters() :: [adapter_name()]

  @doc """
  Returns the workflow definition for this recipe.
  """
  @callback workflow() :: Workflow.t()

  @doc """
  Validates the merged configuration.

  Called after merging defaults with user config, before running.

  Returns `:ok` if valid, `{:error, reason}` otherwise.
  """
  @callback validate_config(config()) :: :ok | {:error, term()}

  @optional_callbacks [validate_config: 1]

  @doc """
  Use macro for recipes.

  Provides:
  - Default implementations for optional callbacks
  - Helper functions for building workflows
  - Module attribute for recipe metadata
  """
  defmacro __using__(_opts) do
    quote do
      @behaviour CrucibleKitchen.Recipe

      import CrucibleKitchen.Workflow

      # Default implementation - no validation
      @impl CrucibleKitchen.Recipe
      def validate_config(_config), do: :ok

      # Can be overridden
      defoverridable validate_config: 1

      # Default optional adapters - none
      @impl CrucibleKitchen.Recipe
      def optional_adapters, do: []

      defoverridable optional_adapters: 0
    end
  end

  @doc """
  Merge user config with recipe defaults.

  User values take precedence over defaults.
  """
  @spec merge_config(module(), config()) :: config()
  def merge_config(recipe, user_config) when is_atom(recipe) and is_map(user_config) do
    defaults = recipe.default_config()
    Map.merge(defaults, user_config)
  end

  @doc """
  Validate that required adapters are present.
  """
  @spec validate_adapters(module(), Context.adapter_map()) ::
          :ok | {:error, [atom()]}
  def validate_adapters(recipe, adapters) when is_atom(recipe) and is_map(adapters) do
    required = recipe.required_adapters()
    missing = Enum.filter(required, fn port -> not Map.has_key?(adapters, port) end)

    if missing == [], do: :ok, else: {:error, missing}
  end

  @doc """
  Get recipe info as a map.
  """
  @spec info(module()) :: map()
  def info(recipe) when is_atom(recipe) do
    %{
      name: recipe.name(),
      description: recipe.description(),
      default_config: recipe.default_config(),
      required_adapters: recipe.required_adapters(),
      optional_adapters: recipe.optional_adapters()
    }
  end
end
