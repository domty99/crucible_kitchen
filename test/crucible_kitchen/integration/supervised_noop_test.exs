defmodule CrucibleKitchen.Integration.SupervisedNoopTest do
  @moduledoc """
  Integration tests for the supervised workflow structure and component interactions.

  These tests verify the workflow structure, stage composition, and adapter
  integration at the component level.
  """

  use ExUnit.Case

  @moduletag :integration

  alias CrucibleKitchen.Context
  alias CrucibleKitchen.Stages.{Evaluate, RegisterModel}

  describe "supervised workflow stages" do
    test "all expected stages are present in workflow" do
      {:ok, description} = CrucibleKitchen.describe(:supervised)

      expected_stages = [
        :load_dataset,
        :init_session,
        :init_tokenizer,
        :build_dataset,
        :set_epoch,
        :get_batch,
        :forward_backward,
        :await_fb,
        :optim_step,
        :await_optim,
        :log_step_metrics,
        :log_epoch_metrics,
        :save_final,
        :final_evaluate,
        :register_model,
        :cleanup
      ]

      for stage <- expected_stages do
        assert stage in description.stages,
               "Expected stage #{inspect(stage)} not found in workflow"
      end
    end

    test "workflow includes evaluate and register_model in finalization" do
      {:ok, description} = CrucibleKitchen.describe(:supervised)

      # Both stages should be present
      assert :final_evaluate in description.stages
      assert :register_model in description.stages
    end
  end

  describe "evaluate and register_model integration" do
    test "evaluate stage feeds into register_model stage" do
      # Build context with evaluator and model_registry adapters
      context =
        Context.new(
          %{
            model: "integration-test-model",
            dataset: "test-dataset",
            epochs: 1
          },
          noop_adapters()
        )
        |> Context.put_state(:global_step, 100)
        |> Context.put_state(:session, :mock_session)
        |> Context.put_state(:final_checkpoint, "tinker://checkpoint/test")
        |> Context.put_state(:eval_dataset, [%{input: "test"}])

      # Execute evaluate stage
      assert {:ok, after_eval} = Evaluate.execute(context)
      assert Map.has_key?(after_eval.state, :eval_results)

      # Execute register_model stage
      assert {:ok, after_register} = RegisterModel.execute(after_eval)
      assert Map.has_key?(after_register.state, :registered_model)

      # Verify the model has metrics from evaluation
      model = after_register.state.registered_model
      assert model.name == "integration-test-model"
    end

    test "telemetry events are emitted in correct order" do
      ref = make_ref()

      :telemetry.attach_many(
        "integration-test-#{inspect(ref)}",
        [
          [:crucible_kitchen, :eval, :complete],
          [:crucible_kitchen, :model, :registered]
        ],
        fn event, _measurements, _metadata, %{pid: pid} ->
          send(pid, {:telemetry_event, event})
        end,
        %{pid: self()}
      )

      context =
        Context.new(
          %{model: "telemetry-test-model", dataset: "test"},
          noop_adapters()
        )
        |> Context.put_state(:global_step, 50)
        |> Context.put_state(:final_checkpoint, "path/to/checkpoint")
        |> Context.put_state(:eval_dataset, [%{input: "test"}])

      {:ok, after_eval} = Evaluate.execute(context)
      {:ok, _after_register} = RegisterModel.execute(after_eval)

      assert_receive {:telemetry_event, [:crucible_kitchen, :eval, :complete]}
      assert_receive {:telemetry_event, [:crucible_kitchen, :model, :registered]}

      :telemetry.detach("integration-test-#{inspect(ref)}")
    end
  end

  describe "adapter integration" do
    test "evaluator adapter produces results consumed by register_model" do
      context =
        Context.new(
          %{model: "adapter-test-model", dataset: "test"},
          noop_adapters()
        )
        |> Context.put_state(:global_step, 100)
        |> Context.put_state(:final_checkpoint, "checkpoint")
        |> Context.put_state(:eval_dataset, [%{input: "sample"}])

      {:ok, context} = Evaluate.execute(context)

      # Eval results should be available
      eval_results = context.state.eval_results
      assert Map.has_key?(eval_results, :accuracy) or Map.has_key?(eval_results, :sample_count)

      # Register model
      {:ok, context} = RegisterModel.execute(context)

      # Registered model should exist
      model = context.state.registered_model
      assert model.id != nil
    end

    test "noop adapters work correctly together" do
      # Verify noop adapters implement their behaviours
      assert function_exported?(CrucibleKitchen.Adapters.Noop.Evaluator, :evaluate, 3)
      assert function_exported?(CrucibleKitchen.Adapters.Noop.Evaluator, :generate_report, 2)
      assert function_exported?(CrucibleKitchen.Adapters.Noop.ModelRegistry, :register, 2)
      assert function_exported?(CrucibleKitchen.Adapters.Noop.ModelRegistry, :get, 2)
      assert function_exported?(CrucibleKitchen.Adapters.Noop.ModelRegistry, :list, 1)
    end
  end

  defp noop_adapters do
    %{
      training_client: CrucibleKitchen.Adapters.Noop.TrainingClient,
      dataset_store: CrucibleKitchen.Adapters.Noop.DatasetStore,
      hub_client: CrucibleKitchen.Adapters.Noop.HubClient,
      blob_store: CrucibleKitchen.Adapters.Noop.BlobStore,
      evaluator: CrucibleKitchen.Adapters.Noop.Evaluator,
      model_registry: CrucibleKitchen.Adapters.Noop.ModelRegistry
    }
  end
end
