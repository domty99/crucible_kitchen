defmodule CrucibleKitchenTest do
  use ExUnit.Case
  doctest CrucibleKitchen

  alias CrucibleKitchen.Context
  alias CrucibleKitchen.Workflow.Runner

  describe "Context" do
    test "new/2 creates context with config and adapters" do
      config = %{model: "test-model", epochs: 3}
      adapters = %{training_client: NoopAdapter, dataset_store: NoopAdapter}

      context = Context.new(config, adapters)

      assert context.config == config
      assert context.adapters == adapters
      assert context.state == %{}
      assert context.metrics == []
      assert context.metadata.run_id =~ ~r/^run_\d+_[a-f0-9]+$/
    end

    test "get/put state" do
      context = Context.new(%{}, %{})

      context = Context.put_state(context, :foo, :bar)
      assert Context.get_state(context, :foo) == :bar
      assert Context.get_state(context, :missing, :default) == :default
    end

    test "get config" do
      context = Context.new(%{model: "llama"}, %{})

      assert Context.get_config(context, :model) == "llama"
      assert Context.get_config(context, :missing, "default") == "default"
    end

    test "record_metric accumulates metrics" do
      context = Context.new(%{}, %{})

      context = Context.record_metric(context, :loss, 0.5, step: 1)
      context = Context.record_metric(context, :loss, 0.4, step: 2)

      assert length(context.metrics) == 2
      [metric2, metric1] = context.metrics
      assert metric1.name == :loss
      assert metric1.value == 0.5
      assert metric2.value == 0.4
    end
  end

  describe "Workflow DSL" do
    defmodule TestStage do
      use CrucibleKitchen.Stage

      def name, do: :test_stage

      def execute(context) do
        executed = Context.get_state(context, :executed, [])
        {:ok, Context.put_state(context, :executed, [:test_stage | executed])}
      end
    end

    defmodule SimpleWorkflow do
      use CrucibleKitchen.Workflow

      workflow do
        stage(:step1, TestStage)
        stage(:step2, TestStage)
      end
    end

    test "workflow defines stages" do
      stages = SimpleWorkflow.__workflow__()
      assert length(stages) == 2
      assert {:stage, :step1, TestStage, []} in stages
      assert {:stage, :step2, TestStage, []} in stages
    end

    defmodule LoopWorkflow do
      use CrucibleKitchen.Workflow

      workflow do
        loop :items, over: :items_range do
          stage(:process, TestStage)
        end
      end

      def items_range(ctx), do: Context.get_config(ctx, :items, [])
    end

    test "workflow with loop" do
      stages = LoopWorkflow.__workflow__()
      assert {:loop_start, :items, _} = Enum.at(stages, 0)
      assert {:stage, :process, TestStage, []} = Enum.at(stages, 1)
      assert {:loop_end, :items} = Enum.at(stages, 2)
    end
  end

  describe "Workflow Runner" do
    defmodule CountingStage do
      use CrucibleKitchen.Stage

      def name, do: :counting

      def execute(context) do
        count = Context.get_state(context, :count, 0)
        {:ok, Context.put_state(context, :count, count + 1)}
      end
    end

    defmodule CountingWorkflow do
      use CrucibleKitchen.Workflow

      workflow do
        stage(:count1, CountingStage)
        stage(:count2, CountingStage)
        stage(:count3, CountingStage)
      end
    end

    test "runs stages in sequence" do
      context = Context.new(%{}, %{training_client: NoopAdapter, dataset_store: NoopAdapter})

      {:ok, result} = Runner.run(CountingWorkflow, context)

      assert Context.get_state(result, :count) == 3
    end

    defmodule LoopCountingWorkflow do
      use CrucibleKitchen.Workflow

      workflow do
        loop :iterations, over: :iterations_range do
          stage(:count, CountingStage)
        end
      end

      def iterations_range(_ctx), do: 1..5
    end

    test "runs loop iterations" do
      context = Context.new(%{}, %{training_client: NoopAdapter, dataset_store: NoopAdapter})

      {:ok, result} = Runner.run(LoopCountingWorkflow, context)

      assert Context.get_state(result, :count) == 5
    end

    defmodule ConditionalWorkflow do
      use CrucibleKitchen.Workflow

      workflow do
        conditional :should_do_it? do
          stage(:count, CountingStage)
        end
      end

      def should_do_it?(ctx), do: Context.get_config(ctx, :do_it, false)
    end

    test "conditional executes when true" do
      context =
        Context.new(%{do_it: true}, %{training_client: NoopAdapter, dataset_store: NoopAdapter})

      {:ok, result} = Runner.run(ConditionalWorkflow, context)
      assert Context.get_state(result, :count) == 1
    end

    test "conditional skips when false" do
      context =
        Context.new(%{do_it: false}, %{training_client: NoopAdapter, dataset_store: NoopAdapter})

      {:ok, result} = Runner.run(ConditionalWorkflow, context)
      assert Context.get_state(result, :count, 0) == 0
    end

    defmodule StreamLoopWorkflow do
      use CrucibleKitchen.Workflow

      workflow do
        stream_loop :items, over: :item_stream, prefetch: 2 do
          stage(:count, CountingStage)
        end
      end

      def item_stream(_ctx) do
        Stream.take(1..10, 5)
      end
    end

    test "stream_loop processes items lazily" do
      context = Context.new(%{}, %{training_client: NoopAdapter, dataset_store: NoopAdapter})

      {:ok, result} = Runner.run(StreamLoopWorkflow, context)
      assert Context.get_state(result, :count) == 5
    end

    defmodule AsyncLoopWorkflow do
      use CrucibleKitchen.Workflow

      workflow do
        async_loop :producer_consumer,
          producer: :produce_item,
          stop_when: :should_stop?,
          buffer_size: 2 do
          stage(:count, CountingStage)
        end
      end

      def produce_item(ctx) do
        # Produce a simple item
        count = Context.get_state(ctx, :produced, 0)
        count + 1
      end

      def should_stop?(ctx) do
        # Stop after consuming 5 items
        Context.get_state(ctx, :count, 0) >= 5
      end
    end

    test "async_loop runs producer-consumer pattern" do
      context = Context.new(%{}, %{training_client: NoopAdapter, dataset_store: NoopAdapter})

      {:ok, result} = Runner.run(AsyncLoopWorkflow, context)
      # Should have consumed at least 5 items
      assert Context.get_state(result, :count) >= 5
    end
  end
end

defmodule NoopAdapter do
  # Minimal adapter for testing
  def start_session(_config), do: {:ok, :mock_session}
  def close_session(_session), do: :ok
  def forward_backward(_session, _datums), do: {:ok, :mock_future}
  def optim_step(_session, _lr), do: {:ok, :mock_future}
  def await(_future, _timeout \\ 5000), do: {:ok, %{}}
  def save_checkpoint(_session, _name), do: {:ok, "/tmp/checkpoint"}
  def load_checkpoint(_session, _path), do: :ok
  def get_tokenizer(_session), do: {:ok, :mock_tokenizer}
  def load(_name, _opts), do: {:ok, []}
  def stream(_name, _opts), do: {:ok, []}
  def info(_name), do: {:ok, %{}}
end
