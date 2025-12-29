# Workflow Engine Design

**Purpose:** Define how workflows are composed, compiled, and executed.

---

## Workflow Philosophy

Workflows are **declarative compositions** of stages. They describe:
1. What stages to run
2. In what order (sequential, parallel, conditional)
3. With what control flow (loops, branches)

Workflows do NOT describe:
1. How stages are implemented
2. What adapters to use
3. Configuration values

---

## Built-in Workflows

### Supervised Training Workflow

```elixir
defmodule CrucibleKitchen.Workflows.Supervised do
  @moduledoc """
  Standard supervised learning workflow.

  Stages:
  1. Load dataset
  2. Initialize training session
  3. For each epoch:
     a. For each batch:
        - Render messages to tokens
        - Forward-backward pass
        - Optimizer step
        - Log metrics
     b. Maybe checkpoint
     c. Maybe evaluate
  4. Save final weights
  """

  use CrucibleKitchen.Workflow

  alias CrucibleKitchen.Stages

  workflow do
    stage :load_dataset, Stages.LoadDataset
    stage :init_session, Stages.InitSession
    stage :init_tokenizer, Stages.InitTokenizer
    stage :build_dataset, Stages.BuildSupervisedDataset

    loop :epochs, over: fn ctx -> 0..(ctx.config.num_epochs - 1) end do
      stage :set_epoch, Stages.SetEpoch

      loop :batches, over: fn ctx -> ctx.state.dataset end do
        stage :get_batch, Stages.GetBatch
        stage :forward_backward, Stages.ForwardBackward
        stage :await_fb, Stages.AwaitFuture, key: :fb_future
        stage :optim_step, Stages.OptimStep
        stage :await_optim, Stages.AwaitFuture, key: :optim_future
        stage :log_step_metrics, Stages.LogStepMetrics
      end

      stage :log_epoch_metrics, Stages.LogEpochMetrics

      conditional fn ctx -> should_checkpoint?(ctx) end do
        stage :checkpoint, Stages.SaveCheckpoint
      end

      conditional fn ctx -> should_evaluate?(ctx) end do
        stage :evaluate, Stages.Evaluate
      end
    end

    stage :save_final, Stages.SaveFinalWeights
    stage :cleanup, Stages.Cleanup
  end

  defp should_checkpoint?(%{config: config, state: state}) do
    config[:save_every] && rem(state.global_step, config.save_every) == 0
  end

  defp should_evaluate?(%{config: config, state: state}) do
    config[:eval_every] && rem(state.global_step, config.eval_every) == 0
  end
end
```

### Reinforcement Learning Workflow

```elixir
defmodule CrucibleKitchen.Workflows.Reinforcement do
  @moduledoc """
  RL training workflow with rollouts and advantage estimation.
  """

  use CrucibleKitchen.Workflow

  alias CrucibleKitchen.Stages

  workflow do
    stage :load_envs, Stages.LoadEnvironments
    stage :init_session, Stages.InitSession
    stage :init_policy, Stages.InitPolicy

    loop :iterations, over: fn ctx -> 1..ctx.config.num_iterations end do
      # Collect rollouts (parallel across env groups)
      parallel max_concurrency: fn ctx -> ctx.config.rollout_parallelism end do
        stage :rollout, Stages.CollectRollouts
      end

      # Compute advantages
      stage :compute_advantages, Stages.ComputeAdvantages

      # Policy optimization (multiple epochs over rollout data)
      loop :ppo_epochs, over: fn ctx -> 1..ctx.config.ppo_epochs end do
        loop :minibatches, over: fn ctx -> ctx.state.rollout_batches end do
          stage :forward_backward, Stages.ForwardBackwardPPO
          stage :optim_step, Stages.OptimStep
        end
      end

      # Logging and checkpointing
      stage :log_iteration, Stages.LogRLMetrics

      conditional fn ctx -> should_checkpoint?(ctx) end do
        stage :checkpoint, Stages.SaveCheckpoint
      end
    end

    stage :save_final, Stages.SaveFinalWeights
  end
end
```

### DPO Workflow

```elixir
defmodule CrucibleKitchen.Workflows.Preference do
  @moduledoc """
  Direct Preference Optimization workflow.
  """

  use CrucibleKitchen.Workflow

  alias CrucibleKitchen.Stages

  workflow do
    stage :load_preference_data, Stages.LoadPreferenceDataset
    stage :init_session, Stages.InitSession
    stage :init_reference, Stages.InitReferenceModel  # For DPO β penalty

    loop :epochs, over: fn ctx -> 0..(ctx.config.num_epochs - 1) end do
      loop :batches, over: fn ctx -> ctx.state.dataset end do
        stage :get_preference_batch, Stages.GetPreferenceBatch
        stage :forward_backward_dpo, Stages.ForwardBackwardDPO
        stage :optim_step, Stages.OptimStep
        stage :log_dpo_metrics, Stages.LogDPOMetrics
      end

      conditional fn ctx -> should_evaluate?(ctx) end do
        stage :evaluate_preferences, Stages.EvaluatePreferences
      end
    end

    stage :save_final, Stages.SaveFinalWeights
  end
end
```

---

## Workflow DSL Compilation

The workflow DSL compiles to an intermediate representation that the runner executes.

```elixir
defmodule CrucibleKitchen.Workflow.Builder do
  @moduledoc """
  Compiles workflow DSL into executable IR.
  """

  @type stage_node :: {:stage, atom(), module(), keyword()}
  @type loop_node :: {:loop, atom(), iterator :: fun(), [node()]}
  @type parallel_node :: {:parallel, keyword(), [node()]}
  @type conditional_node :: {:conditional, predicate :: fun(), [node()]}
  @type node :: stage_node() | loop_node() | parallel_node() | conditional_node()

  def build(workflow_module) do
    workflow_module.__workflow__()
    |> parse_dsl([])
    |> validate_workflow()
  end

  defp parse_dsl([], acc), do: Enum.reverse(acc)

  defp parse_dsl([{:stage, name, module, opts} | rest], acc) do
    node = {:stage, name, module, opts}
    parse_dsl(rest, [node | acc])
  end

  defp parse_dsl([{:loop_start, name, opts} | rest], acc) do
    {body, rest} = collect_until_loop_end(name, rest, [])
    iterator = Keyword.fetch!(opts, :over)
    node = {:loop, name, iterator, body}
    parse_dsl(rest, [node | acc])
  end

  defp parse_dsl([{:parallel_start, opts} | rest], acc) do
    {body, rest} = collect_until_parallel_end(rest, [])
    node = {:parallel, opts, body}
    parse_dsl(rest, [node | acc])
  end

  defp parse_dsl([{:conditional_start, predicate} | rest], acc) do
    {body, rest} = collect_until_conditional_end(rest, [])
    node = {:conditional, predicate, body}
    parse_dsl(rest, [node | acc])
  end

  # ... collect helpers
end
```

---

## Workflow Runner

The runner executes compiled workflows with full telemetry.

```elixir
defmodule CrucibleKitchen.Workflow.Runner do
  @moduledoc """
  Executes compiled workflows.
  """

  require Logger

  def run(workflow_module, initial_context) do
    workflow_ir = CrucibleKitchen.Workflow.Builder.build(workflow_module)

    :telemetry.span(
      [:crucible_kitchen, :workflow, :run],
      %{workflow: workflow_module, config: initial_context.config},
      fn ->
        result = execute_nodes(workflow_ir, initial_context)
        {result, %{}}
      end
    )
  end

  defp execute_nodes([], context), do: {:ok, context}

  defp execute_nodes([node | rest], context) do
    case execute_node(node, context) do
      {:ok, new_context} -> execute_nodes(rest, new_context)
      {:error, reason} -> {:error, reason}
    end
  end

  defp execute_node({:stage, name, module, opts}, context) do
    context = %{context | current_stage: name}

    :telemetry.span(
      [:crucible_kitchen, :stage, :run],
      %{stage: name, module: module},
      fn ->
        with :ok <- validate_stage(module, context),
             {:ok, new_context} <- execute_stage(module, context) do
          {{:ok, new_context}, %{}}
        else
          {:error, reason} -> {{:error, {name, reason}}, %{}}
        end
      end
    )
  end

  defp execute_node({:loop, name, iterator, body}, context) do
    items = iterator.(context)

    Enum.reduce_while(items, {:ok, context}, fn item, {:ok, ctx} ->
      ctx = Context.put_state(ctx, :"#{name}_current", item)

      case execute_nodes(body, ctx) do
        {:ok, new_ctx} -> {:cont, {:ok, new_ctx}}
        {:error, reason} -> {:halt, {:error, reason}}
      end
    end)
  end

  defp execute_node({:parallel, opts, body}, context) do
    max_concurrency = Keyword.get(opts, :max_concurrency, fn _ -> System.schedulers_online() end)
    concurrency = if is_function(max_concurrency), do: max_concurrency.(context), else: max_concurrency

    # Each parallel branch gets a copy of context
    # Results are merged after all complete
    tasks = Enum.map(body, fn node ->
      Task.async(fn -> execute_node(node, context) end)
    end)

    results = Task.await_many(tasks, :infinity)
    merge_parallel_results(results, context)
  end

  defp execute_node({:conditional, predicate, body}, context) do
    if predicate.(context) do
      execute_nodes(body, context)
    else
      {:ok, context}
    end
  end

  defp validate_stage(module, context) do
    if function_exported?(module, :validate, 1) do
      module.validate(context)
    else
      :ok
    end
  end

  defp execute_stage(module, context) do
    module.execute(context)
  rescue
    e ->
      if function_exported?(module, :rollback, 2) do
        context = module.rollback(context, e)
        {:error, {e, context}}
      else
        reraise e, __STACKTRACE__
      end
  end

  defp merge_parallel_results(results, original_context) do
    Enum.reduce(results, {:ok, original_context}, fn
      {:ok, ctx}, {:ok, acc} -> {:ok, merge_contexts(acc, ctx)}
      {:error, reason}, _ -> {:error, reason}
      _, {:error, reason} -> {:error, reason}
    end)
  end

  defp merge_contexts(base, overlay) do
    # Merge state maps, metrics lists
    %{base |
      state: Map.merge(base.state, overlay.state),
      metrics: overlay.metrics ++ base.metrics
    }
  end
end
```

---

## Pipelining Support

For backends that support async pipelining (like Tinker), stages can overlap:

```elixir
defmodule CrucibleKitchen.Stages.ForwardBackward do
  use CrucibleKitchen.Stage

  alias CrucibleKitchen.Context
  alias CrucibleTrain.Ports.TrainingClient

  @impl true
  def execute(context) do
    ports = Context.get_train_ports(context)
    session = Context.get_state(context, :session)
    batch = Context.get_state(context, :current_batch)

    future = TrainingClient.forward_backward(ports, session, batch)
    {:ok, Context.put_state(context, :fb_future, future)}
  end
end

defmodule CrucibleKitchen.Stages.AwaitFuture do
  use CrucibleKitchen.Stage

  alias CrucibleKitchen.Context
  alias CrucibleTrain.Ports.TrainingClient

  @impl true
  def execute(context) do
    ports = Context.get_train_ports(context)
    key = context.stage_opts[:key]
    future = Context.get_state(context, key)

    if future do
      case TrainingClient.await(ports, future) do
        {:ok, result} ->
          result_key = String.to_atom("#{key}_result")
          {:ok, Context.put_state(context, result_key, result)}

        {:error, reason} ->
          {:error, reason}
      end
    else
      {:ok, context}
    end
  end
end
```

---

## Workflow Composition

Workflows can be composed from other workflows:

```elixir
defmodule CrucibleKitchen.Workflows.RLHF do
  @moduledoc """
  Full RLHF pipeline: SFT → Reward Model → RL
  """

  use CrucibleKitchen.Workflow

  workflow do
    # Phase 1: Supervised Fine-Tuning
    stage :sft_phase, Stages.RunSubworkflow,
      workflow: CrucibleKitchen.Workflows.Supervised,
      config_key: :sft_config

    # Phase 2: Train Reward Model
    stage :reward_phase, Stages.RunSubworkflow,
      workflow: CrucibleKitchen.Workflows.RewardModel,
      config_key: :reward_config

    # Phase 3: RL Fine-Tuning
    stage :rl_phase, Stages.RunSubworkflow,
      workflow: CrucibleKitchen.Workflows.Reinforcement,
      config_key: :rl_config
  end
end
```

---

## Error Handling and Recovery

```elixir
defmodule CrucibleKitchen.Workflow.ErrorHandler do
  @moduledoc """
  Centralized error handling for workflow execution.
  """

  def handle_stage_error(context, stage_name, error) do
    :telemetry.execute(
      [:crucible_kitchen, :stage, :exception],
      %{},
      %{stage: stage_name, error: error, context: context}
    )

    case context.config[:on_error] do
      :halt -> {:error, {stage_name, error}}
      :skip -> {:ok, mark_skipped(context, stage_name)}
      :retry -> retry_stage(context, stage_name, error)
      {:retry, max_attempts} -> retry_stage(context, stage_name, error, max_attempts)
    end
  end

  defp retry_stage(context, stage_name, error, max_attempts \\ 3) do
    attempt = Context.get_state(context, :"#{stage_name}_attempt", 0) + 1

    if attempt <= max_attempts do
      Logger.warning("Retrying stage #{stage_name}, attempt #{attempt}/#{max_attempts}")
      context = Context.put_state(context, :"#{stage_name}_attempt", attempt)
      {:retry, context}
    else
      {:error, {stage_name, {:max_retries_exceeded, error}}}
    end
  end
end
```

This workflow engine design provides:
1. **Declarative composition** via DSL
2. **Flexible control flow** (loops, conditionals, parallel)
3. **Full observability** via telemetry spans
4. **Pipelining support** for async backends
5. **Error recovery** with retry/skip strategies
6. **Workflow composition** for complex pipelines like RLHF
