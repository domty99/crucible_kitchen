defmodule CrucibleKitchen.Workflow do
  @moduledoc """
  DSL for defining training workflows.

  Workflows are compositions of stages with control flow constructs.
  They describe HOW training is orchestrated, not WHAT the stages do.

  ## Example

      defmodule MyWorkflow do
        use CrucibleKitchen.Workflow

        workflow do
          stage :load, LoadStage
          stage :init, InitStage

          loop :epochs, over: :epochs_range do
            stage :train_epoch, TrainEpochStage

            conditional :should_eval? do
              stage :eval, EvalStage
            end
          end

          stage :save, SaveStage
        end

        def epochs_range(ctx), do: 0..(ctx.config.epochs - 1)
        def should_eval?(ctx), do: rem(ctx.state.step, 100) == 0
      end

  ## DSL Reference

  ### stage(name, module, opts \\\\ [])

  Define a stage to execute.

  - `name` - Stage identifier (atom)
  - `module` - Module implementing `CrucibleKitchen.Stage`
  - `opts` - Options passed to the stage

  ### loop(name, opts, do: block)

  Iterate over a collection.

  - `name` - Loop identifier
  - `opts[:over]` - Atom name of function in this module that takes context and returns Enumerable

  ### conditional(predicate_name, do: block)

  Execute block only if predicate returns true.

  - `predicate_name` - Atom name of function in this module that takes context and returns boolean

  ### parallel(opts \\\\ [], do: block)

  Execute stages in parallel.

  - `opts[:max_concurrency]` - Max concurrent stages (default: schedulers_online)
  """

  @type stage_def :: {:stage, atom(), module(), keyword()}
  @type loop_def :: {:loop, atom(), keyword(), list()}
  @type conditional_def :: {:conditional, atom(), list()}
  @type parallel_def :: {:parallel, keyword(), list()}
  @type t :: [stage_def() | loop_def() | conditional_def() | parallel_def()]

  @doc false
  defmacro __using__(_opts) do
    quote do
      import CrucibleKitchen.Workflow.DSL

      Module.register_attribute(__MODULE__, :workflow_stages, accumulate: true)

      @before_compile CrucibleKitchen.Workflow
    end
  end

  @doc false
  defmacro __before_compile__(_env) do
    quote do
      def __workflow__ do
        @workflow_stages |> Enum.reverse()
      end
    end
  end
end

defmodule CrucibleKitchen.Workflow.DSL do
  @moduledoc false

  @doc "Define a stage."
  defmacro stage(name, module, opts \\ []) do
    quote do
      @workflow_stages {:stage, unquote(name), unquote(module), unquote(opts)}
    end
  end

  @doc "Define a loop. The :over option should be an atom naming a function in this module."
  defmacro loop(name, opts, do: block) do
    # Extract the :over option and ensure it's an atom (function name)
    quote do
      @workflow_stages {:loop_start, unquote(name), unquote(opts)}
      unquote(block)
      @workflow_stages {:loop_end, unquote(name)}
    end
  end

  @doc "Define a conditional block. Predicate should be an atom naming a function in this module."
  defmacro conditional(predicate, do: block) do
    quote do
      @workflow_stages {:conditional_start, unquote(predicate)}
      unquote(block)
      @workflow_stages {:conditional_end}
    end
  end

  @doc "Define a parallel block."
  defmacro parallel(opts \\ [], do: block) do
    quote do
      @workflow_stages {:parallel_start, unquote(opts)}
      unquote(block)
      @workflow_stages {:parallel_end}
    end
  end

  @doc """
  Define an async producer-consumer loop for off-policy training.

  This is used for async RL training where:
  - Producer collects rollouts asynchronously
  - Consumer trains on buffered rollouts

  ## Options

  - `:producer` - Atom name of function that produces items (called repeatedly)
  - `:buffer_size` - Max items to buffer before blocking producer (default: 4)
  - `:stop_when` - Atom name of predicate function to stop the loop

  ## Example

      async_loop :off_policy, producer: :collect_rollouts, stop_when: :done_training? do
        stage :train_batch, TrainBatchStage
      end
  """
  defmacro async_loop(name, opts, do: block) do
    quote do
      @workflow_stages {:async_loop_start, unquote(name), unquote(opts)}
      unquote(block)
      @workflow_stages {:async_loop_end, unquote(name)}
    end
  end

  @doc """
  Define a streaming loop for memory-efficient batch processing.

  Unlike regular loop which materializes the full collection, stream_loop
  lazily evaluates items, supporting:
  - Infinite streams
  - Memory-efficient large dataset iteration
  - Prefetching with configurable buffer

  ## Options

  - `:over` - Atom name of function returning a Stream
  - `:prefetch` - Number of items to prefetch (default: 2)

  ## Example

      stream_loop :batches, over: :stream_batches, prefetch: 4 do
        stage :train_batch, TrainBatchStage
      end
  """
  defmacro stream_loop(name, opts, do: block) do
    quote do
      @workflow_stages {:stream_loop_start, unquote(name), unquote(opts)}
      unquote(block)
      @workflow_stages {:stream_loop_end, unquote(name)}
    end
  end

  @doc "Workflow block (required wrapper)."
  defmacro workflow(do: block) do
    quote do
      unquote(block)
    end
  end
end
