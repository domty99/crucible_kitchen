defmodule CrucibleKitchen.Workflow.Runner do
  @moduledoc """
  Executes compiled workflows.

  The runner takes a workflow module and initial context, then executes
  each stage in order, handling control flow constructs like loops,
  conditionals, and parallel blocks.
  """

  require Logger

  alias CrucibleKitchen.Context

  @doc """
  Run a workflow with the given initial context.
  """
  @spec run(module(), Context.t()) :: {:ok, Context.t()} | {:error, term()}
  def run(workflow_module, initial_context) do
    workflow_ir = build_workflow(workflow_module)

    Logger.debug("Starting workflow: #{inspect(workflow_module)}")

    execute_nodes(workflow_ir, initial_context, workflow_module)
  end

  # Build workflow IR from module
  defp build_workflow(workflow_module) do
    raw_stages = workflow_module.__workflow__()
    parse_dsl(raw_stages, [])
  end

  # Parse DSL into IR
  defp parse_dsl([], acc), do: Enum.reverse(acc)

  defp parse_dsl([{:stage, name, module, opts} | rest], acc) do
    node = {:stage, name, module, opts}
    parse_dsl(rest, [node | acc])
  end

  defp parse_dsl([{:loop_start, name, opts} | rest], acc) do
    {body, rest} = collect_until(:loop_end, name, rest, [])
    iterator = Keyword.fetch!(opts, :over)
    node = {:loop, name, iterator, parse_dsl(body, [])}
    parse_dsl(rest, [node | acc])
  end

  defp parse_dsl([{:conditional_start, predicate} | rest], acc) do
    {body, rest} = collect_until(:conditional_end, nil, rest, [])
    node = {:conditional, predicate, parse_dsl(body, [])}
    parse_dsl(rest, [node | acc])
  end

  defp parse_dsl([{:parallel_start, opts} | rest], acc) do
    {body, rest} = collect_until(:parallel_end, nil, rest, [])
    node = {:parallel, opts, parse_dsl(body, [])}
    parse_dsl(rest, [node | acc])
  end

  defp parse_dsl([{:async_loop_start, name, opts} | rest], acc) do
    {body, rest} = collect_until(:async_loop_end, name, rest, [])
    node = {:async_loop, name, opts, parse_dsl(body, [])}
    parse_dsl(rest, [node | acc])
  end

  defp parse_dsl([{:stream_loop_start, name, opts} | rest], acc) do
    {body, rest} = collect_until(:stream_loop_end, name, rest, [])
    iterator = Keyword.fetch!(opts, :over)
    prefetch = Keyword.get(opts, :prefetch, 2)
    node = {:stream_loop, name, iterator, prefetch, parse_dsl(body, [])}
    parse_dsl(rest, [node | acc])
  end

  defp parse_dsl([_unknown | rest], acc) do
    # Skip unknown nodes
    parse_dsl(rest, acc)
  end

  # Collect nodes until end marker
  defp collect_until(end_type, name, [{end_type, name} | rest], acc) do
    {Enum.reverse(acc), rest}
  end

  defp collect_until(end_type, nil, [{end_type} | rest], acc) do
    {Enum.reverse(acc), rest}
  end

  defp collect_until(end_type, name, [node | rest], acc) do
    collect_until(end_type, name, rest, [node | acc])
  end

  defp collect_until(_end_type, _name, [], acc) do
    # Unclosed block - return what we have
    {Enum.reverse(acc), []}
  end

  # Execute IR nodes
  defp execute_nodes([], context, _workflow_module), do: {:ok, context}

  defp execute_nodes([node | rest], context, workflow_module) do
    case execute_node(node, context, workflow_module) do
      {:ok, new_context} -> execute_nodes(rest, new_context, workflow_module)
      {:error, reason} -> {:error, reason}
    end
  end

  defp execute_node({:stage, name, module, opts}, context, _workflow_module) do
    context = %{context | current_stage: name, stage_opts: opts}

    Logger.debug("Executing stage: #{name}")

    :telemetry.span(
      [:crucible_kitchen, :stage, :run],
      %{stage: name, module: module},
      fn ->
        with :ok <- validate_stage(module, context),
             {:ok, new_context} <- execute_stage(module, context) do
          {{:ok, new_context}, %{stage: name, success: true}}
        else
          {:error, reason} ->
            Logger.error("Stage #{name} failed: #{inspect(reason)}")
            {{:error, {name, reason}}, %{stage: name, success: false, error: reason}}
        end
      end
    )
  end

  defp execute_node({:loop, name, iterator, body}, context, workflow_module) do
    items = resolve_callable(iterator, context, workflow_module)

    Logger.debug("Starting loop: #{name} with #{Enum.count(items)} items")

    Enum.reduce_while(items, {:ok, context}, fn item, {:ok, ctx} ->
      ctx = Context.put_state(ctx, :"#{name}_current", item)

      case execute_nodes(body, ctx, workflow_module) do
        {:ok, new_ctx} -> {:cont, {:ok, new_ctx}}
        {:error, reason} -> {:halt, {:error, reason}}
      end
    end)
  end

  defp execute_node({:conditional, predicate, body}, context, workflow_module) do
    should_execute = resolve_callable(predicate, context, workflow_module)

    if should_execute do
      Logger.debug("Conditional: executing")
      execute_nodes(body, context, workflow_module)
    else
      Logger.debug("Conditional: skipping")
      {:ok, context}
    end
  end

  defp execute_node({:parallel, opts, body}, context, workflow_module) do
    max_concurrency =
      case Keyword.get(opts, :max_concurrency) do
        nil -> System.schedulers_online()
        fun when is_function(fun, 1) -> fun.(context)
        n when is_integer(n) -> n
      end

    Logger.debug(
      "Parallel: executing #{length(body)} stages with max_concurrency=#{max_concurrency}"
    )

    # For now, execute sequentially (parallel execution is a future enhancement)
    # NOTE: Parallel execution with Task.async_stream is a future enhancement
    execute_nodes(body, context, workflow_module)
  end

  defp execute_node({:async_loop, name, opts, body}, context, workflow_module) do
    producer_fn = Keyword.fetch!(opts, :producer)
    stop_predicate = Keyword.fetch!(opts, :stop_when)
    buffer_size = Keyword.get(opts, :buffer_size, 4)

    Logger.debug("Starting async loop: #{name} with buffer_size=#{buffer_size}")

    # Create a blocking queue for producer-consumer pattern
    {:ok, buffer} = Agent.start_link(fn -> :queue.new() end)
    {:ok, done_ref} = Agent.start_link(fn -> false end)

    # Start producer task - producer just produces items until signaled to stop
    producer_task =
      Task.async(fn ->
        produce_items(buffer, done_ref, buffer_size, producer_fn, context, workflow_module)
      end)

    # Consumer loop - process items from buffer and check stop condition
    result = consume_items(buffer, done_ref, body, context, workflow_module, name, stop_predicate)

    # Wait for producer to finish
    Task.await(producer_task, :infinity)

    # Cleanup
    Agent.stop(buffer)
    Agent.stop(done_ref)

    result
  end

  defp execute_node({:stream_loop, name, iterator, prefetch, body}, context, workflow_module) do
    stream = resolve_callable(iterator, context, workflow_module)

    Logger.debug("Starting stream loop: #{name} with prefetch=#{prefetch}")

    # Use Stream with async prefetch for efficient processing
    stream
    |> Stream.with_index()
    |> Task.async_stream(
      fn {item, _idx} -> item end,
      max_concurrency: prefetch,
      ordered: true
    )
    |> Enum.reduce_while({:ok, context}, fn {:ok, item}, {:ok, ctx} ->
      ctx = Context.put_state(ctx, :"#{name}_current", item)

      case execute_nodes(body, ctx, workflow_module) do
        {:ok, new_ctx} -> {:cont, {:ok, new_ctx}}
        {:error, reason} -> {:halt, {:error, reason}}
      end
    end)
  end

  # Async loop helpers

  defp produce_items(buffer, done_ref, buffer_size, producer_fn, context, workflow_module) do
    with false <- Agent.get(done_ref, & &1),
         :ok <- wait_for_buffer_space(buffer, done_ref, buffer_size) do
      item = resolve_callable(producer_fn, context, workflow_module)
      Agent.update(buffer, fn queue -> :queue.in(item, queue) end)
      produce_items(buffer, done_ref, buffer_size, producer_fn, context, workflow_module)
    else
      _ -> :done
    end
  end

  defp wait_for_buffer_space(buffer, done_ref, max_size) do
    cond do
      Agent.get(done_ref, & &1) ->
        :done

      Agent.get(buffer, fn q -> :queue.len(q) end) < max_size ->
        :ok

      true ->
        Process.sleep(10)
        wait_for_buffer_space(buffer, done_ref, max_size)
    end
  end

  defp consume_items(buffer, done_ref, body, context, workflow_module, name, stop_predicate) do
    if resolve_callable(stop_predicate, context, workflow_module) do
      Agent.update(done_ref, fn _ -> true end)
      {:ok, context}
    else
      buffer
      |> dequeue_item()
      |> process_consumer_item(
        buffer,
        done_ref,
        body,
        context,
        workflow_module,
        name,
        stop_predicate
      )
    end
  end

  defp dequeue_item(buffer) do
    Agent.get_and_update(buffer, fn queue ->
      case :queue.out(queue) do
        {{:value, item}, new_queue} -> {{:item, item}, new_queue}
        {:empty, queue} -> {:empty, queue}
      end
    end)
  end

  defp process_consumer_item(
         :empty,
         buffer,
         done_ref,
         body,
         ctx,
         workflow_module,
         name,
         stop_pred
       ) do
    Process.sleep(10)
    consume_items(buffer, done_ref, body, ctx, workflow_module, name, stop_pred)
  end

  defp process_consumer_item(
         {:item, item},
         buffer,
         done_ref,
         body,
         ctx,
         workflow_module,
         name,
         stop_pred
       ) do
    new_ctx = Context.put_state(ctx, :"#{name}_current", item)

    case execute_nodes(body, new_ctx, workflow_module) do
      {:ok, result_ctx} ->
        consume_items(buffer, done_ref, body, result_ctx, workflow_module, name, stop_pred)

      {:error, reason} ->
        Agent.update(done_ref, fn _ -> true end)
        {:error, reason}
    end
  end

  # Resolve a callable - can be a function, an atom (function name on workflow), or a literal
  defp resolve_callable(callable, context, workflow_module) do
    case callable do
      fun when is_function(fun, 1) ->
        fun.(context)

      atom when is_atom(atom) ->
        # Call the function on the workflow module
        apply(workflow_module, atom, [context])

      literal ->
        # Return literal values as-is (true, false, lists, etc.)
        literal
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
end
