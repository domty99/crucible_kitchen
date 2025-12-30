defmodule CrucibleKitchen.Stages.EvaluateTest do
  use ExUnit.Case, async: true

  alias CrucibleKitchen.Context
  alias CrucibleKitchen.Stages.Evaluate

  describe "name/0" do
    test "returns :evaluate" do
      assert Evaluate.name() == :evaluate
    end
  end

  describe "execute/1" do
    test "skips evaluation when skip_eval config is true" do
      context = build_context(%{skip_eval: true})

      assert {:ok, result_context} = Evaluate.execute(context)
      assert result_context.state.eval_results.skipped == true
    end

    test "runs evaluation with evaluator adapter" do
      context = build_context_with_evaluator()

      assert {:ok, result_context} = Evaluate.execute(context)
      assert Map.has_key?(result_context.state, :eval_results)
      assert Map.has_key?(result_context.state.eval_results, :accuracy)
    end

    test "generates evaluation report" do
      context = build_context_with_evaluator()

      {:ok, result_context} = Evaluate.execute(context)

      assert result_context.state.eval_report != nil
      assert is_binary(result_context.state.eval_report)
    end

    test "records metrics" do
      context = build_context_with_evaluator()

      {:ok, result_context} = Evaluate.execute(context)

      assert Enum.any?(result_context.metrics, fn m -> m.name == :eval_run end)
    end

    test "emits telemetry event on completion" do
      :telemetry.attach(
        "evaluate-test-handler",
        [:crucible_kitchen, :eval, :complete],
        fn event, measurements, metadata, _ ->
          send(self(), {:telemetry, event, measurements, metadata})
        end,
        nil
      )

      context = build_context_with_evaluator()
      {:ok, _} = Evaluate.execute(context)

      assert_receive {:telemetry, [:crucible_kitchen, :eval, :complete], measurements, _}
      assert Map.has_key?(measurements, :accuracy) or Map.has_key?(measurements, :f1)

      :telemetry.detach("evaluate-test-handler")
    end

    test "falls back to basic evaluation without evaluator adapter" do
      context = build_context(%{})

      assert {:ok, result_context} = Evaluate.execute(context)
      assert result_context.state.eval_results.evaluated == true
    end
  end

  describe "validate/1" do
    test "always returns :ok" do
      context = build_context(%{})
      assert :ok = Evaluate.validate(context)
    end
  end

  defp build_context(extra_config) do
    config =
      Map.merge(
        %{
          model: "test-model",
          dataset: "test-dataset"
        },
        extra_config
      )

    Context.new(
      config,
      %{
        training_client: CrucibleKitchen.Adapters.Noop.TrainingClient,
        dataset_store: CrucibleKitchen.Adapters.Noop.DatasetStore
      }
    )
    |> Context.put_state(:global_step, 100)
    |> Context.put_state(:session, :mock_session)
  end

  defp build_context_with_evaluator do
    Context.new(
      %{
        model: "test-model",
        dataset: "test-dataset",
        generate_eval_report: true
      },
      %{
        training_client: CrucibleKitchen.Adapters.Noop.TrainingClient,
        dataset_store: CrucibleKitchen.Adapters.Noop.DatasetStore,
        evaluator: CrucibleKitchen.Adapters.Noop.Evaluator
      }
    )
    |> Context.put_state(:global_step, 100)
    |> Context.put_state(:session, :mock_session)
    |> Context.put_state(:eval_dataset, [%{input: "test1"}, %{input: "test2"}])
  end
end
