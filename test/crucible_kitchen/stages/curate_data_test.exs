defmodule CrucibleKitchen.Stages.CurateDataTest do
  use ExUnit.Case, async: true

  alias CrucibleKitchen.Context
  alias CrucibleKitchen.Stages.CurateData

  describe "name/0" do
    test "returns :curate_data" do
      assert CurateData.name() == :curate_data
    end
  end

  describe "execute/1" do
    test "curates examples and stores in state" do
      context =
        build_context(%{deployment_id: "test-deployment"})
        |> with_feedback_adapter(curated_count: 50)

      assert {:ok, result_context} = CurateData.execute(context)
      assert length(result_context.state.curated_examples) == 50
      assert result_context.state.curated_count == 50
    end

    test "records curated_count metric" do
      context =
        build_context(%{deployment_id: "test-deployment"})
        |> with_feedback_adapter(curated_count: 25)

      {:ok, result_context} = CurateData.execute(context)

      metric = Enum.find(result_context.metrics, &(&1.name == :curated_count))
      assert metric.value == 25
    end

    test "skips when no feedback_client adapter" do
      context = build_context_without_feedback(%{deployment_id: "test"})

      assert {:ok, result_context} = CurateData.execute(context)
      assert result_context.state.curated_examples == []
      assert result_context.state.curated_count == 0
    end

    test "emits telemetry event" do
      :telemetry.attach(
        "curate-data-test-handler",
        [:crucible_kitchen, :feedback, :data_curated],
        fn event, measurements, metadata, _ ->
          send(self(), {:telemetry, event, measurements, metadata})
        end,
        nil
      )

      context =
        build_context(%{deployment_id: "test"})
        |> with_feedback_adapter(curated_count: 10)

      {:ok, _} = CurateData.execute(context)

      assert_receive {:telemetry, [:crucible_kitchen, :feedback, :data_curated], measurements,
                      metadata}

      assert measurements.count == 10
      assert metadata.deployment_id == "test"
      assert Map.has_key?(metadata, :by_source)

      :telemetry.detach("curate-data-test-handler")
    end
  end

  describe "validate/1" do
    test "returns :ok when deployment_id in config" do
      context = build_context(%{deployment_id: "test"})
      assert :ok = CurateData.validate(context)
    end

    test "returns error when no deployment_id" do
      context = build_context(%{})
      assert {:error, message} = CurateData.validate(context)
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
