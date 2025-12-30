defmodule CrucibleKitchen.Stages.UpdateBaselineTest do
  use ExUnit.Case, async: true

  alias CrucibleKitchen.Context
  alias CrucibleKitchen.Stages.UpdateBaseline

  describe "name/0" do
    test "returns :update_baseline" do
      assert UpdateBaseline.name() == :update_baseline
    end
  end

  describe "execute/1" do
    test "updates baseline and sets state" do
      context = build_context(%{deployment_id: "test-deployment"})

      assert {:ok, result_context} = UpdateBaseline.execute(context)
      assert result_context.state.baseline_updated == true
    end

    test "includes model info in logging when present" do
      context =
        build_context(%{deployment_id: "test-deployment"})
        |> Context.put_state(:registered_model, %{
          id: "model-123",
          name: "test-model",
          version: "1.0.0"
        })

      assert {:ok, result_context} = UpdateBaseline.execute(context)
      assert result_context.state.baseline_updated == true
    end

    test "skips when no feedback_client adapter" do
      context = build_context_without_feedback(%{deployment_id: "test"})

      assert {:ok, result_context} = UpdateBaseline.execute(context)
      assert result_context.state.baseline_updated == false
    end

    test "emits telemetry event on success" do
      :telemetry.attach(
        "update-baseline-test-handler",
        [:crucible_kitchen, :feedback, :baseline_updated],
        fn event, measurements, metadata, _ ->
          send(self(), {:telemetry, event, measurements, metadata})
        end,
        nil
      )

      context = build_context(%{deployment_id: "test"})
      {:ok, _} = UpdateBaseline.execute(context)

      assert_receive {:telemetry, [:crucible_kitchen, :feedback, :baseline_updated], _, metadata}

      assert metadata.deployment_id == "test"

      :telemetry.detach("update-baseline-test-handler")
    end

    test "includes model info in telemetry when present" do
      :telemetry.attach(
        "update-baseline-model-handler",
        [:crucible_kitchen, :feedback, :baseline_updated],
        fn event, measurements, metadata, _ ->
          send(self(), {:telemetry, event, measurements, metadata})
        end,
        nil
      )

      context =
        build_context(%{deployment_id: "test"})
        |> Context.put_state(:registered_model, %{
          id: "model-123",
          name: "test-model",
          version: "1.0.0"
        })

      {:ok, _} = UpdateBaseline.execute(context)

      assert_receive {:telemetry, [:crucible_kitchen, :feedback, :baseline_updated], _, metadata}

      assert metadata.model_name == "test-model"
      assert metadata.model_version == "1.0.0"
      assert metadata.model_id == "model-123"

      :telemetry.detach("update-baseline-model-handler")
    end
  end

  describe "validate/1" do
    test "returns :ok when deployment_id in config" do
      context = build_context(%{deployment_id: "test"})
      assert :ok = UpdateBaseline.validate(context)
    end

    test "returns :ok when deployment_id in state" do
      context =
        build_context(%{})
        |> Context.put_state(:deployment_id, "test")

      assert :ok = UpdateBaseline.validate(context)
    end

    test "returns error when no deployment_id" do
      context = build_context(%{})
      assert {:error, message} = UpdateBaseline.validate(context)
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
end
