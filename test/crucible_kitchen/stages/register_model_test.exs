defmodule CrucibleKitchen.Stages.RegisterModelTest do
  use ExUnit.Case, async: true

  alias CrucibleKitchen.Context
  alias CrucibleKitchen.Stages.RegisterModel

  describe "name/0" do
    test "returns :register_model" do
      assert RegisterModel.name() == :register_model
    end
  end

  describe "execute/1" do
    test "registers model with correct artifact structure" do
      context = build_context_with_training_complete()

      assert {:ok, result_context} = RegisterModel.execute(context)
      assert result_context.state.registered_model != nil
      assert result_context.state.registered_model.id != nil
    end

    test "includes lineage in artifact" do
      context = build_context_with_training_complete()

      {:ok, result_context} = RegisterModel.execute(context)

      model = result_context.state.registered_model
      assert model.lineage.dataset == "test-dataset"
    end

    test "skips registration when no model_registry adapter configured" do
      context =
        Context.new(
          %{model: "test-model", dataset: "test-dataset"},
          %{
            training_client: CrucibleKitchen.Adapters.Noop.TrainingClient,
            dataset_store: CrucibleKitchen.Adapters.Noop.DatasetStore
          }
        )
        |> Context.put_state(:final_checkpoint, "path/to/checkpoint")

      assert {:ok, result_context} = RegisterModel.execute(context)
      refute Map.has_key?(result_context.state, :registered_model)
    end

    test "records metric on successful registration" do
      context = build_context_with_training_complete()

      {:ok, result_context} = RegisterModel.execute(context)

      assert Enum.any?(result_context.metrics, fn m -> m.name == :model_registered end)
    end

    test "emits telemetry event on success" do
      :telemetry.attach(
        "register-model-test-handler",
        [:crucible_kitchen, :model, :registered],
        fn event, measurements, metadata, _ ->
          send(self(), {:telemetry, event, measurements, metadata})
        end,
        nil
      )

      context = build_context_with_training_complete()
      {:ok, _} = RegisterModel.execute(context)

      assert_receive {:telemetry, [:crucible_kitchen, :model, :registered], _, metadata}
      assert metadata.model_id != nil
      assert metadata.name != nil

      :telemetry.detach("register-model-test-handler")
    end
  end

  describe "validate/1" do
    test "returns :ok with model config" do
      context =
        Context.new(
          %{model: "test-model"},
          %{
            training_client: CrucibleKitchen.Adapters.Noop.TrainingClient,
            dataset_store: CrucibleKitchen.Adapters.Noop.DatasetStore
          }
        )

      assert :ok = RegisterModel.validate(context)
    end

    test "returns error when model config is missing" do
      context =
        Context.new(
          %{dataset: "test-dataset"},
          %{
            training_client: CrucibleKitchen.Adapters.Noop.TrainingClient,
            dataset_store: CrucibleKitchen.Adapters.Noop.DatasetStore
          }
        )

      assert {:error, {:missing_config, :model}} = RegisterModel.validate(context)
    end
  end

  defp build_context_with_training_complete do
    Context.new(
      %{
        model: "test-model",
        dataset: "test-dataset",
        epochs: 1,
        learning_rate: 2.0e-4,
        batch_size: 32
      },
      %{
        training_client: CrucibleKitchen.Adapters.Noop.TrainingClient,
        dataset_store: CrucibleKitchen.Adapters.Noop.DatasetStore,
        model_registry: CrucibleKitchen.Adapters.Noop.ModelRegistry
      }
    )
    |> Context.put_state(:final_checkpoint, "tinker://checkpoints/abc123")
    |> Context.put_state(:epoch_metrics, %{loss: 1.23, accuracy: 0.82})
    |> Context.put_state(:global_step, 1000)
    |> Context.put_state(:current_epoch, 0)
  end
end
