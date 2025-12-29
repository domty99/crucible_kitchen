defmodule CrucibleKitchen.EvaluationTest do
  use ExUnit.Case, async: true

  alias CrucibleKitchen.Evaluation

  describe "exact_match evaluator" do
    test "passes when strings are identical" do
      result = Evaluation.evaluate(:exact_match, "hello", "hello")

      assert result.passed == true
      assert result.score == 1.0
    end

    test "fails when strings differ" do
      result = Evaluation.evaluate(:exact_match, "hello", "world")

      assert result.passed == false
      assert result.score == 0.0
    end

    test "is case sensitive by default" do
      result = Evaluation.evaluate(:exact_match, "Hello", "hello")

      assert result.passed == false
    end

    test "case insensitive option" do
      result = Evaluation.evaluate(:exact_match, "Hello", "hello", case_sensitive: false)

      assert result.passed == true
    end

    test "normalize_whitespace option" do
      result =
        Evaluation.evaluate(:exact_match, "hello  world", "hello world",
          normalize_whitespace: true
        )

      assert result.passed == true
    end
  end

  describe "contains evaluator" do
    test "passes when output contains expected" do
      result = Evaluation.evaluate(:contains, "hello world", "world")

      assert result.passed == true
      assert result.score == 1.0
    end

    test "fails when output does not contain expected" do
      result = Evaluation.evaluate(:contains, "hello", "world")

      assert result.passed == false
    end

    test "case insensitive option" do
      result = Evaluation.evaluate(:contains, "Hello World", "WORLD", case_sensitive: false)

      assert result.passed == true
    end
  end

  describe "regex evaluator" do
    test "passes when pattern matches" do
      result = Evaluation.evaluate(:regex, "answer: 42", "\\d+")

      assert result.passed == true
    end

    test "fails when pattern does not match" do
      result = Evaluation.evaluate(:regex, "no numbers here", "\\d+")

      assert result.passed == false
    end

    test "accepts compiled regex" do
      result = Evaluation.evaluate(:regex, "test123", "unused", pattern: ~r/\d+/)

      assert result.passed == true
    end
  end

  describe "numeric evaluator" do
    test "passes when numbers match exactly" do
      result = Evaluation.evaluate(:numeric, "42", "42")

      assert result.passed == true
      assert result.score == 1.0
    end

    test "passes within tolerance" do
      result = Evaluation.evaluate(:numeric, "42.001", "42", tolerance: 0.01)

      assert result.passed == true
    end

    test "fails outside tolerance" do
      result = Evaluation.evaluate(:numeric, "43", "42", tolerance: 0.5)

      assert result.passed == false
    end

    test "handles floats" do
      result = Evaluation.evaluate(:numeric, "3.14159", "3.14159")

      assert result.passed == true
    end

    test "handles parsing errors" do
      result = Evaluation.evaluate(:numeric, "not a number", "42")

      assert result.passed == false
      assert result.details[:error] != nil
    end
  end

  describe "f1 evaluator" do
    test "perfect match gives f1 of 1.0" do
      result = Evaluation.evaluate(:f1, "the quick brown fox", "the quick brown fox")

      assert result.score == 1.0
      assert result.passed == true
    end

    test "partial overlap gives partial score" do
      result = Evaluation.evaluate(:f1, "the quick brown", "the quick brown fox")

      assert result.score > 0.5
      assert result.score < 1.0
    end

    test "no overlap gives f1 of 0.0" do
      result = Evaluation.evaluate(:f1, "hello", "world")

      assert result.score == 0.0
      assert result.passed == false
    end

    test "both empty gives f1 of 1.0" do
      result = Evaluation.evaluate(:f1, "", "")

      assert result.score == 1.0
      assert result.passed == true
    end
  end

  describe "custom evaluator" do
    test "calls custom function with output and expected" do
      custom_fn = fn output, expected ->
        %{
          score: if(String.length(output) == String.length(expected), do: 1.0, else: 0.0),
          passed: String.length(output) == String.length(expected),
          details: nil
        }
      end

      result = Evaluation.evaluate(:custom, "abc", "xyz", custom_fn: custom_fn)

      assert result.passed == true
      assert result.score == 1.0
    end

    test "raises without custom_fn option" do
      assert_raise ArgumentError, ~r/custom_fn/, fn ->
        Evaluation.evaluate(:custom, "abc", "xyz", [])
      end
    end
  end

  describe "evaluate_batch/3" do
    test "aggregates results from multiple samples" do
      samples = [
        %{output: "yes", expected: "yes"},
        %{output: "no", expected: "yes"},
        %{output: "maybe", expected: "maybe"}
      ]

      result = Evaluation.evaluate_batch(:exact_match, samples)

      assert result.total == 3
      assert result.passed == 2
      assert result.failed == 1
      assert_in_delta result.accuracy, 0.666, 0.01
    end

    test "accepts tuple format" do
      samples = [
        {"yes", "yes"},
        {"no", "yes"}
      ]

      result = Evaluation.evaluate_batch(:exact_match, samples)

      assert result.total == 2
      assert result.passed == 1
    end

    test "calculates mean_score" do
      samples = [
        %{output: "yes", expected: "yes"},
        %{output: "no", expected: "yes"}
      ]

      result = Evaluation.evaluate_batch(:exact_match, samples)

      assert result.mean_score == 0.5
    end

    test "includes individual results" do
      samples = [
        %{output: "a", expected: "a"},
        %{output: "b", expected: "c"}
      ]

      result = Evaluation.evaluate_batch(:exact_match, samples)

      assert length(result.results) == 2
      assert Enum.at(result.results, 0).passed == true
      assert Enum.at(result.results, 1).passed == false
    end
  end

  describe "evaluate_multi/3" do
    test "runs multiple evaluators on same samples" do
      samples = [
        %{output: "hello world", expected: "world"},
        %{output: "foo", expected: "bar"}
      ]

      results = Evaluation.evaluate_multi([:exact_match, :contains], samples)

      assert Map.has_key?(results, :exact_match)
      assert Map.has_key?(results, :contains)

      # Exact match should fail both
      assert results.exact_match.passed == 0

      # Contains should pass one
      assert results.contains.passed == 1
    end

    test "accepts named evaluators with options" do
      samples = [
        %{output: "Hello", expected: "hello"}
      ]

      results =
        Evaluation.evaluate_multi(
          [
            {:case_sensitive, :exact_match, []},
            {:case_insensitive, :exact_match, case_sensitive: false}
          ],
          samples
        )

      assert results.case_sensitive.passed == 0
      assert results.case_insensitive.passed == 1
    end
  end

  describe "evaluator helpers" do
    test "exact_match_evaluator creates evaluator function" do
      evaluator = Evaluation.exact_match_evaluator(case_sensitive: false)

      result = evaluator.("Hello", "hello")

      assert result.passed == true
    end

    test "contains_evaluator creates evaluator function" do
      evaluator = Evaluation.contains_evaluator()

      result = evaluator.("hello world", "world")

      assert result.passed == true
    end

    test "numeric_evaluator creates evaluator with tolerance" do
      evaluator = Evaluation.numeric_evaluator(0.1)

      result = evaluator.("42.05", "42")

      assert result.passed == true
    end
  end

  describe "compose_and/1" do
    test "passes only if all evaluators pass" do
      composed = Evaluation.compose_and([:exact_match, :contains])

      result = composed.("hello", "hello")
      assert result.passed == true

      result = composed.("hello world", "hello")
      assert result.passed == false
    end

    test "accepts function evaluators" do
      always_pass = fn _, _ -> %{score: 1.0, passed: true, details: nil} end
      always_fail = fn _, _ -> %{score: 0.0, passed: false, details: nil} end

      composed = Evaluation.compose_and([always_pass, always_fail])

      result = composed.("any", "thing")
      assert result.passed == false
    end
  end

  describe "compose_or/1" do
    test "passes if any evaluator passes" do
      composed = Evaluation.compose_or([:exact_match, :contains])

      result = composed.("hello world", "hello")
      assert result.passed == true
    end

    test "fails only if all evaluators fail" do
      composed = Evaluation.compose_or([:exact_match, :contains])

      result = composed.("foo", "bar")
      assert result.passed == false
    end
  end

  describe "edge cases" do
    test "empty strings" do
      result = Evaluation.evaluate(:exact_match, "", "")
      assert result.passed == true
    end

    test "unknown evaluator raises" do
      assert_raise ArgumentError, ~r/Unknown evaluator/, fn ->
        Evaluation.evaluate(:nonexistent, "a", "b")
      end
    end

    test "batch with empty samples" do
      result = Evaluation.evaluate_batch(:exact_match, [])

      assert result.total == 0
      assert result.accuracy == 0.0
    end
  end
end
