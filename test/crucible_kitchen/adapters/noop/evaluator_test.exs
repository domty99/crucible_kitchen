defmodule CrucibleKitchen.Adapters.Noop.EvaluatorTest do
  use ExUnit.Case, async: true

  alias CrucibleKitchen.Adapters.Noop.Evaluator

  describe "evaluate/3" do
    test "returns mock metrics for all requested metrics" do
      model = :test_model
      dataset = [%{input: "test"}]
      opts = [metrics: [:accuracy, :f1, :precision, :recall]]

      assert {:ok, results} = Evaluator.evaluate(opts, model, dataset)
      assert Map.has_key?(results, :accuracy)
      assert Map.has_key?(results, :f1)
      assert Map.has_key?(results, :precision)
      assert Map.has_key?(results, :recall)
      assert Map.has_key?(results, :sample_count)
    end

    test "returns realistic metric values in expected range" do
      assert {:ok, results} = Evaluator.evaluate([metrics: [:accuracy]], :model, [])

      assert is_float(results.accuracy)
      assert results.accuracy >= 0.0 and results.accuracy <= 1.0
    end

    test "allows overriding mock metrics" do
      opts = [metrics: [:accuracy], mock_accuracy: 0.99]

      assert {:ok, results} = Evaluator.evaluate(opts, :model, [])
      assert results.accuracy == 0.99
    end

    test "counts samples from list dataset" do
      dataset = [%{a: 1}, %{a: 2}, %{a: 3}]

      assert {:ok, results} = Evaluator.evaluate([], :model, dataset)
      assert results.sample_count == 3
    end
  end

  describe "generate_report/2" do
    test "generates markdown report by default" do
      results = %{accuracy: 0.95, f1: 0.92, sample_count: 100}

      assert {:ok, report} = Evaluator.generate_report([], results)
      assert is_binary(report)
      assert String.contains?(report, "Evaluation Report")
      assert String.contains?(report, "Accuracy")
    end

    test "generates JSON report when requested" do
      results = %{accuracy: 0.95, sample_count: 100}

      assert {:ok, report} = Evaluator.generate_report([format: :json], results)
      assert {:ok, parsed} = Jason.decode(report)
      assert parsed["mock"] == true
    end
  end
end
