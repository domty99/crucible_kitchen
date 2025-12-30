defmodule CrucibleKitchen.Stages.CheckTriggersTest do
  use ExUnit.Case, async: true

  alias CrucibleKitchen.Context
  alias CrucibleKitchen.Stages.CheckTriggers

  describe "name/0" do
    test "returns :check_triggers" do
      assert CheckTriggers.name() == :check_triggers
    end
  end

  describe "execute/1" do
    test "returns empty triggers when none fire" do
      context = build_context(%{deployment_id: "test-deployment"})

      assert {:ok, result_context} = CheckTriggers.execute(context)
      assert result_context.state.triggers == []
      assert result_context.state.should_retrain == false
    end

    test "returns triggers when they fire" do
      context =
        build_context(%{deployment_id: "test-deployment"})
        |> with_feedback_adapter(triggers: [{:trigger, :drift_threshold}])

      assert {:ok, result_context} = CheckTriggers.execute(context)
      assert result_context.state.triggers == [{:trigger, :drift_threshold}]
      assert result_context.state.should_retrain == true
    end

    test "records trigger_count metric" do
      context =
        build_context(%{deployment_id: "test-deployment"})
        |> with_feedback_adapter(triggers: [{:trigger, :a}, {:trigger, :b}])

      {:ok, result_context} = CheckTriggers.execute(context)

      metric = Enum.find(result_context.metrics, &(&1.name == :trigger_count))
      assert metric.value == 2
    end

    test "gets deployment_id from config" do
      context = build_context(%{deployment_id: "from-config"})

      assert {:ok, result_context} = CheckTriggers.execute(context)
      assert result_context.state.should_retrain == false
    end

    test "falls back to deployment_id from state" do
      context =
        build_context(%{})
        |> Context.put_state(:deployment_id, "from-state")

      assert {:ok, result_context} = CheckTriggers.execute(context)
      assert result_context.state.should_retrain == false
    end

    test "skips when no feedback_client adapter" do
      context = build_context_without_feedback(%{deployment_id: "test"})

      assert {:ok, result_context} = CheckTriggers.execute(context)
      assert result_context.state.triggers == []
      assert result_context.state.should_retrain == false
    end

    test "emits telemetry event" do
      :telemetry.attach(
        "check-triggers-test-handler",
        [:crucible_kitchen, :feedback, :triggers_checked],
        fn event, measurements, metadata, _ ->
          send(self(), {:telemetry, event, measurements, metadata})
        end,
        nil
      )

      context = build_context(%{deployment_id: "test"})
      {:ok, _} = CheckTriggers.execute(context)

      assert_receive {:telemetry, [:crucible_kitchen, :feedback, :triggers_checked], measurements,
                      metadata}

      assert Map.has_key?(measurements, :count)
      assert metadata.deployment_id == "test"

      :telemetry.detach("check-triggers-test-handler")
    end
  end

  describe "validate/1" do
    test "returns :ok when deployment_id in config" do
      context = build_context(%{deployment_id: "test"})
      assert :ok = CheckTriggers.validate(context)
    end

    test "returns :ok when deployment_id in state" do
      context =
        build_context(%{})
        |> Context.put_state(:deployment_id, "test")

      assert :ok = CheckTriggers.validate(context)
    end

    test "returns error when no deployment_id" do
      context = build_context(%{})
      assert {:error, message} = CheckTriggers.validate(context)
      assert String.contains?(message, "deployment_id")
    end
  end

  defp build_context(extra_config) do
    config = Map.merge(%{}, extra_config)

    Context.new(
      config,
      %{
        training_client: CrucibleKitchen.Adapters.Noop.TrainingClient,
        dataset_store: CrucibleKitchen.Adapters.Noop.DatasetStore,
        feedback_client: CrucibleKitchen.Adapters.Noop.FeedbackClient
      }
    )
  end

  defp build_context_without_feedback(extra_config) do
    config = Map.merge(%{}, extra_config)

    Context.new(
      config,
      %{
        training_client: CrucibleKitchen.Adapters.Noop.TrainingClient,
        dataset_store: CrucibleKitchen.Adapters.Noop.DatasetStore
      }
    )
  end

  defp with_feedback_adapter(context, opts) do
    adapters =
      Map.put(
        context.adapters,
        :feedback_client,
        {CrucibleKitchen.Adapters.Noop.FeedbackClient, opts}
      )

    %{context | adapters: adapters}
  end
end
