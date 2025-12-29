defmodule CrucibleKitchen.Stage do
  @moduledoc """
  Behaviour for defining workflow stages.

  Stages are the atomic units of work in a workflow. They receive a context,
  perform some operation, and return an updated context.

  ## Example

      defmodule MyStage do
        use CrucibleKitchen.Stage

        @impl true
        def name, do: :my_stage

        @impl true
        def execute(context) do
          # Get config and state
          model = get_config(context, :model)
          session = get_state(context, :session)

          # Do work using adapters
          client = get_adapter(context, :training_client)
          {:ok, result} = client.do_something(session)

          # Update state and return
          context = put_state(context, :my_result, result)
          context = record_metric(context, :my_metric, 42)

          {:ok, context}
        end
      end

  ## Lifecycle

  1. `validate/1` (optional) - Check preconditions before execution
  2. `execute/1` - Perform the stage's work
  3. `rollback/2` (optional) - Clean up on failure

  ## Helpers

  The `use CrucibleKitchen.Stage` macro imports these helpers:

  - `get_config/2,3` - Get configuration value
  - `get_state/2,3` - Get state value
  - `put_state/3` - Update state
  - `merge_state/2` - Merge multiple state updates
  - `get_adapter/2` - Get an adapter by port name
  - `record_metric/3,4` - Record a metric
  """

  alias CrucibleKitchen.Context

  @type context :: Context.t()
  @type result :: {:ok, context()} | {:error, term()}

  @doc "Return the stage name (used for telemetry and logging)."
  @callback name() :: atom()

  @doc "Execute the stage's work."
  @callback execute(context()) :: result()

  @doc "Validate preconditions before execution."
  @callback validate(context()) :: :ok | {:error, term()}

  @doc "Clean up on failure."
  @callback rollback(context(), error :: term()) :: context()

  @optional_callbacks [validate: 1, rollback: 2]

  defmacro __using__(_opts) do
    quote do
      @behaviour CrucibleKitchen.Stage

      import CrucibleKitchen.Stage.Helpers

      # Default implementations
      def validate(_context), do: :ok
      def rollback(context, _error), do: context

      defoverridable validate: 1, rollback: 2
    end
  end
end

defmodule CrucibleKitchen.Stage.Helpers do
  @moduledoc """
  Helper functions imported into stages via `use CrucibleKitchen.Stage`.
  """

  alias CrucibleKitchen.Context

  defdelegate get_config(context, key), to: Context
  defdelegate get_config(context, key, default), to: Context
  defdelegate get_state(context, key), to: Context
  defdelegate get_state(context, key, default), to: Context
  defdelegate put_state(context, key, value), to: Context
  defdelegate merge_state(context, updates), to: Context
  defdelegate get_adapter(context, port), to: Context
  defdelegate get_train_ports(context), to: Context
  defdelegate record_metric(context, name, value), to: Context
  defdelegate record_metric(context, name, value, opts), to: Context
end
