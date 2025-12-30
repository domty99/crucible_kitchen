defmodule CrucibleKitchen.Integration.FeedbackLoopTest do
  @moduledoc """
  Integration tests for the feedback loop workflow.

  These tests verify the workflow structure, stage composition, and
  feedback-driven retraining flow.
  """

  use ExUnit.Case

  @moduletag :integration

  alias CrucibleKitchen.Context
  alias CrucibleKitchen.Stages.{CheckTriggers, CurateData, ExportFeedbackData, UpdateBaseline}

  describe "feedback loop workflow structure" do
    test "workflow is registered as builtin" do
      assert :feedback_loop in CrucibleKitchen.workflows()
    end

    test "all expected stages are present in workflow" do
      {:ok, description} = CrucibleKitchen.describe(:feedback_loop)

      expected_stages = [
        :check_triggers,
        :curate_data,
        :export_feedback,
        :init_session,
        :init_tokenizer,
        :build_dataset,
        :update_baseline
      ]

      for stage <- expected_stages do
        assert stage in description.stages,
               "Expected stage #{inspect(stage)} not found in workflow"
      end
    end
  end

  describe "feedback loop stage integration" do
    test "check_triggers feeds into curate_data when triggers fire" do
      context =
        build_context(%{deployment_id: "test-deployment"})
        |> with_triggers([{:trigger, :drift_threshold}])

      # Execute check_triggers
      assert {:ok, after_check} = CheckTriggers.execute(context)
      assert after_check.state.should_retrain == true

      # Execute curate_data
      assert {:ok, after_curate} = CurateData.execute(after_check)
      assert after_curate.state.curated_examples != []
    end

    test "curate_data feeds into export_feedback_data" do
      context =
        build_context(%{deployment_id: "test-deployment"})
        |> with_curated_examples(10)

      # Execute curate_data first
      {:ok, after_curate} = CurateData.execute(context)

      # Execute export
      {:ok, after_export} = ExportFeedbackData.execute(after_curate)

      assert after_export.state.feedback_data_path != nil
      assert String.contains?(after_export.state.feedback_data_path, "test-deployment")
    end

    test "update_baseline completes the loop after training" do
      context =
        build_context(%{deployment_id: "test-deployment"})
        |> Context.put_state(:registered_model, %{
          id: "model-123",
          name: "retrained-model",
          version: "2.0.0"
        })

      assert {:ok, after_update} = UpdateBaseline.execute(context)
      assert after_update.state.baseline_updated == true
    end

    test "complete feedback loop flow without training when no triggers" do
      context = build_context(%{deployment_id: "test-deployment"})

      # No triggers should fire with default noop adapter settings
      assert {:ok, result} = CheckTriggers.execute(context)
      assert result.state.should_retrain == false
      assert result.state.triggers == []

      # Workflow would skip training stages via conditional
    end
  end

  describe "telemetry integration" do
    test "all feedback stages emit telemetry events" do
      ref = make_ref()

      :telemetry.attach_many(
        "feedback-loop-test-#{inspect(ref)}",
        [
          [:crucible_kitchen, :feedback, :triggers_checked],
          [:crucible_kitchen, :feedback, :data_curated],
          [:crucible_kitchen, :feedback, :data_exported],
          [:crucible_kitchen, :feedback, :baseline_updated]
        ],
        fn event, _measurements, _metadata, %{pid: pid} ->
          send(pid, {:telemetry_event, event})
        end,
        %{pid: self()}
      )

      context =
        build_context(%{deployment_id: "telemetry-test"})
        |> with_triggers([{:trigger, :drift_threshold}])

      # Run through all stages
      {:ok, ctx} = CheckTriggers.execute(context)
      assert_receive {:telemetry_event, [:crucible_kitchen, :feedback, :triggers_checked]}

      {:ok, ctx} = CurateData.execute(ctx)
      assert_receive {:telemetry_event, [:crucible_kitchen, :feedback, :data_curated]}

      {:ok, ctx} = ExportFeedbackData.execute(ctx)
      assert_receive {:telemetry_event, [:crucible_kitchen, :feedback, :data_exported]}

      {:ok, _ctx} = UpdateBaseline.execute(ctx)
      assert_receive {:telemetry_event, [:crucible_kitchen, :feedback, :baseline_updated]}

      :telemetry.detach("feedback-loop-test-#{inspect(ref)}")
    end
  end

  describe "adapter integration" do
    test "noop feedback_client adapter works correctly" do
      assert function_exported?(CrucibleKitchen.Adapters.Noop.FeedbackClient, :check_triggers, 2)
      assert function_exported?(CrucibleKitchen.Adapters.Noop.FeedbackClient, :curate, 2)
      assert function_exported?(CrucibleKitchen.Adapters.Noop.FeedbackClient, :export, 2)
      assert function_exported?(CrucibleKitchen.Adapters.Noop.FeedbackClient, :update_baseline, 2)
    end

    test "feedback stages gracefully handle missing feedback_client" do
      context = build_context_without_feedback(%{deployment_id: "test"})

      # All stages should succeed with warnings, not errors
      {:ok, _} = CheckTriggers.execute(context)
      {:ok, _} = CurateData.execute(context)
      {:ok, _} = ExportFeedbackData.execute(context)
      {:ok, _} = UpdateBaseline.execute(context)
    end
  end

  describe "trigger scenarios" do
    test "drift threshold trigger" do
      context =
        build_context(%{deployment_id: "drift-test"})
        |> with_feedback_adapter(drift_score: 0.5, drift_threshold: 0.2)

      {:ok, result} = CheckTriggers.execute(context)

      assert {:trigger, :drift_threshold} in result.state.triggers
      assert result.state.should_retrain == true
    end

    test "quality drop trigger" do
      context =
        build_context(%{deployment_id: "quality-test"})
        |> with_feedback_adapter(quality_average: 0.5, quality_threshold: 0.7)

      {:ok, result} = CheckTriggers.execute(context)

      assert {:trigger, :quality_drop} in result.state.triggers
      assert result.state.should_retrain == true
    end

    test "data count trigger" do
      context =
        build_context(%{deployment_id: "count-test"})
        |> with_feedback_adapter(event_count: 2000, data_count_threshold: 1000)

      {:ok, result} = CheckTriggers.execute(context)

      assert {:trigger, :data_count} in result.state.triggers
      assert result.state.should_retrain == true
    end

    test "multiple triggers can fire together" do
      context =
        build_context(%{deployment_id: "multi-test"})
        |> with_feedback_adapter(
          drift_score: 0.5,
          drift_threshold: 0.2,
          quality_average: 0.5,
          quality_threshold: 0.7,
          event_count: 2000,
          data_count_threshold: 1000
        )

      {:ok, result} = CheckTriggers.execute(context)

      assert {:trigger, :drift_threshold} in result.state.triggers
      assert {:trigger, :quality_drop} in result.state.triggers
      assert {:trigger, :data_count} in result.state.triggers
      assert length(result.state.triggers) == 3
    end
  end

  defp build_context(extra_config) do
    config = Map.merge(%{}, extra_config)

    Context.new(
      config,
      %{
        training_client: CrucibleKitchen.Adapters.Noop.TrainingClient,
        dataset_store: CrucibleKitchen.Adapters.Noop.DatasetStore,
        hub_client: CrucibleKitchen.Adapters.Noop.HubClient,
        blob_store: CrucibleKitchen.Adapters.Noop.BlobStore,
        evaluator: CrucibleKitchen.Adapters.Noop.Evaluator,
        model_registry: CrucibleKitchen.Adapters.Noop.ModelRegistry,
        feedback_client: CrucibleKitchen.Adapters.Noop.FeedbackClient
      }
    )
    |> Context.put_state(:global_step, 0)
    |> Context.put_state(:session, :mock_session)
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

  defp with_triggers(context, triggers) do
    adapters =
      Map.put(
        context.adapters,
        :feedback_client,
        {CrucibleKitchen.Adapters.Noop.FeedbackClient, triggers: triggers}
      )

    %{context | adapters: adapters}
  end

  defp with_curated_examples(context, count) do
    adapters =
      Map.put(
        context.adapters,
        :feedback_client,
        {CrucibleKitchen.Adapters.Noop.FeedbackClient, curated_count: count}
      )

    %{context | adapters: adapters}
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
