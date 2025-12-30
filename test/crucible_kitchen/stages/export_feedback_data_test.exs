defmodule CrucibleKitchen.Stages.ExportFeedbackDataTest do
  use ExUnit.Case, async: true

  alias CrucibleKitchen.Context
  alias CrucibleKitchen.Stages.ExportFeedbackData

  describe "name/0" do
    test "returns :export_feedback_data" do
      assert ExportFeedbackData.name() == :export_feedback_data
    end
  end

  describe "execute/1" do
    test "exports data and stores path in state" do
      context = build_context(%{deployment_id: "test-deployment"})

      assert {:ok, result_context} = ExportFeedbackData.execute(context)
      assert result_context.state.feedback_data_path != nil
      assert String.contains?(result_context.state.feedback_data_path, "test-deployment")
    end

    test "sets feedback_dataset for training config" do
      context = build_context(%{deployment_id: "test-deployment"})

      {:ok, result_context} = ExportFeedbackData.execute(context)

      assert result_context.state.feedback_dataset == result_context.state.feedback_data_path
    end

    test "uses configured format" do
      context =
        build_context(%{deployment_id: "test"})
        |> with_feedback_adapter(format: :parquet)

      {:ok, result_context} = ExportFeedbackData.execute(context)

      assert String.ends_with?(result_context.state.feedback_data_path, ".parquet")
    end

    test "skips when no feedback_client adapter" do
      context = build_context_without_feedback(%{deployment_id: "test"})

      assert {:ok, result_context} = ExportFeedbackData.execute(context)
      assert result_context.state.feedback_data_path == nil
    end

    test "emits telemetry event" do
      :telemetry.attach(
        "export-feedback-test-handler",
        [:crucible_kitchen, :feedback, :data_exported],
        fn event, measurements, metadata, _ ->
          send(self(), {:telemetry, event, measurements, metadata})
        end,
        nil
      )

      context = build_context(%{deployment_id: "test"})
      {:ok, _} = ExportFeedbackData.execute(context)

      assert_receive {:telemetry, [:crucible_kitchen, :feedback, :data_exported], _, metadata}

      assert metadata.deployment_id == "test"
      assert Map.has_key?(metadata, :path)
      assert Map.has_key?(metadata, :format)

      :telemetry.detach("export-feedback-test-handler")
    end
  end

  describe "validate/1" do
    test "returns :ok when deployment_id in config" do
      context = build_context(%{deployment_id: "test"})
      assert :ok = ExportFeedbackData.validate(context)
    end

    test "returns error when no deployment_id" do
      context = build_context(%{})
      assert {:error, message} = ExportFeedbackData.validate(context)
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
