defmodule CrucibleKitchen.Evaluation do
  @moduledoc """
  Evaluation framework for assessing model outputs.

  This module provides a composable system for evaluating model completions
  against reference data using various metrics.

  ## Evaluator Types

  - **Exact Match** - Binary comparison (case-sensitive or insensitive)
  - **Contains** - Check if output contains expected substring
  - **Regex** - Pattern matching evaluation
  - **Numeric** - Compare numeric values with tolerance
  - **Code** - Execute code and compare results
  - **LLM Judge** - Use an LLM to evaluate quality
  - **Custom** - User-defined evaluation function

  ## Examples

      # Evaluate with exact match
      result = Evaluation.evaluate(:exact_match, "hello", "hello")
      # => %{score: 1.0, passed: true}

      # Evaluate with contains
      result = Evaluation.evaluate(:contains, "hello world", "world")
      # => %{score: 1.0, passed: true}

      # Batch evaluation
      samples = [
        %{output: "4", expected: "4"},
        %{output: "5", expected: "4"}
      ]
      results = Evaluation.evaluate_batch(:exact_match, samples)
      # => %{accuracy: 0.5, passed: 1, failed: 1, total: 2}

  ## Integration with Training

  Evaluators can be used in training loops for validation:

      defp validation_step(model, batch) do
        outputs = generate(model, batch.prompts)
        Evaluation.evaluate_batch(:exact_match, Enum.zip(outputs, batch.targets))
      end
  """

  @type evaluator_type ::
          :exact_match
          | :contains
          | :regex
          | :numeric
          | :code_exec
          | :llm_judge
          | :f1
          | :bleu
          | :rouge
          | :custom

  @type eval_result :: %{
          score: float(),
          passed: boolean(),
          details: map() | nil
        }

  @type batch_result :: %{
          accuracy: float(),
          passed: non_neg_integer(),
          failed: non_neg_integer(),
          total: non_neg_integer(),
          mean_score: float(),
          results: [eval_result()]
        }

  @type eval_opts :: [
          case_sensitive: boolean(),
          normalize_whitespace: boolean(),
          tolerance: float(),
          pattern: Regex.t() | String.t(),
          judge_prompt: String.t(),
          custom_fn: (String.t(), String.t() -> eval_result())
        ]

  # ==========================================================================
  # Single Evaluation
  # ==========================================================================

  @doc """
  Evaluate a single output against expected value.

  ## Parameters

  - `evaluator` - The evaluator type to use
  - `output` - The model's output
  - `expected` - The expected/reference value
  - `opts` - Evaluator-specific options

  ## Returns

  An eval_result map with score, passed, and optional details.
  """
  @spec evaluate(evaluator_type(), String.t(), String.t(), eval_opts()) :: eval_result()
  def evaluate(evaluator, output, expected, opts \\ [])

  def evaluate(:exact_match, output, expected, opts) do
    case_sensitive = Keyword.get(opts, :case_sensitive, true)
    normalize = Keyword.get(opts, :normalize_whitespace, false)

    output_normalized = normalize_string(output, case_sensitive, normalize)
    expected_normalized = normalize_string(expected, case_sensitive, normalize)

    passed = output_normalized == expected_normalized

    %{
      score: if(passed, do: 1.0, else: 0.0),
      passed: passed,
      details: nil
    }
  end

  def evaluate(:contains, output, expected, opts) do
    case_sensitive = Keyword.get(opts, :case_sensitive, true)

    {output_check, expected_check} =
      if case_sensitive do
        {output, expected}
      else
        {String.downcase(output), String.downcase(expected)}
      end

    passed = String.contains?(output_check, expected_check)

    %{
      score: if(passed, do: 1.0, else: 0.0),
      passed: passed,
      details: nil
    }
  end

  def evaluate(:regex, output, expected, opts) do
    pattern =
      case Keyword.get(opts, :pattern, expected) do
        %Regex{} = r -> r
        s when is_binary(s) -> Regex.compile!(s)
      end

    passed = Regex.match?(pattern, output)

    %{
      score: if(passed, do: 1.0, else: 0.0),
      passed: passed,
      details: %{pattern: Regex.source(pattern)}
    }
  end

  def evaluate(:numeric, output, expected, opts) do
    tolerance = Keyword.get(opts, :tolerance, 0.0001)

    with {:ok, output_num} <- parse_number(output),
         {:ok, expected_num} <- parse_number(expected) do
      diff = abs(output_num - expected_num)
      passed = diff <= tolerance

      %{
        score: if(passed, do: 1.0, else: max(0.0, 1.0 - diff / max(abs(expected_num), 1.0))),
        passed: passed,
        details: %{output_value: output_num, expected_value: expected_num, diff: diff}
      }
    else
      {:error, reason} ->
        %{
          score: 0.0,
          passed: false,
          details: %{error: reason}
        }
    end
  end

  def evaluate(:f1, output, expected, opts) do
    # Token-level F1 score
    tokenizer = Keyword.get(opts, :tokenizer, &default_tokenize/1)

    output_tokens = MapSet.new(tokenizer.(output))
    expected_tokens = MapSet.new(tokenizer.(expected))

    if MapSet.size(expected_tokens) == 0 and MapSet.size(output_tokens) == 0 do
      %{score: 1.0, passed: true, details: %{precision: 1.0, recall: 1.0}}
    else
      intersection = MapSet.intersection(output_tokens, expected_tokens)
      precision = safe_div(MapSet.size(intersection), MapSet.size(output_tokens))
      recall = safe_div(MapSet.size(intersection), MapSet.size(expected_tokens))

      f1 =
        if precision + recall > 0 do
          2 * precision * recall / (precision + recall)
        else
          0.0
        end

      threshold = Keyword.get(opts, :threshold, 0.5)

      %{
        score: f1,
        passed: f1 >= threshold,
        details: %{precision: precision, recall: recall, f1: f1}
      }
    end
  end

  def evaluate(:custom, output, expected, opts) do
    case Keyword.get(opts, :custom_fn) do
      nil ->
        raise ArgumentError, "custom evaluator requires :custom_fn option"

      fun when is_function(fun, 2) ->
        fun.(output, expected)

      fun when is_function(fun, 3) ->
        fun.(output, expected, opts)
    end
  end

  # Fallback for unknown evaluators
  def evaluate(evaluator, _output, _expected, _opts) do
    raise ArgumentError, "Unknown evaluator: #{inspect(evaluator)}"
  end

  # ==========================================================================
  # Batch Evaluation
  # ==========================================================================

  @doc """
  Evaluate multiple samples and aggregate results.

  ## Parameters

  - `evaluator` - The evaluator type
  - `samples` - List of %{output: String.t(), expected: String.t()} or tuples
  - `opts` - Evaluator options

  ## Returns

  Aggregated batch results with accuracy, counts, and individual results.
  """
  @spec evaluate_batch(evaluator_type(), [map() | {String.t(), String.t()}], eval_opts()) ::
          batch_result()
  def evaluate_batch(evaluator, samples, opts \\ []) do
    results =
      samples
      |> Enum.map(fn sample ->
        {output, expected} = extract_sample(sample)
        evaluate(evaluator, output, expected, opts)
      end)

    passed_count = Enum.count(results, & &1.passed)
    total = length(results)
    mean_score = if total > 0, do: Enum.sum(Enum.map(results, & &1.score)) / total, else: 0.0

    %{
      accuracy: if(total > 0, do: passed_count / total, else: 0.0),
      passed: passed_count,
      failed: total - passed_count,
      total: total,
      mean_score: mean_score,
      results: results
    }
  end

  @doc """
  Run multiple evaluators on the same samples.

  Returns a map of evaluator name to batch results.
  """
  @spec evaluate_multi(
          [evaluator_type() | {atom(), evaluator_type(), eval_opts()}],
          [map()],
          eval_opts()
        ) :: %{atom() => batch_result()}
  def evaluate_multi(evaluators, samples, default_opts \\ []) do
    evaluators
    |> Enum.map(fn
      {name, evaluator, opts} ->
        {name, evaluate_batch(evaluator, samples, Keyword.merge(default_opts, opts))}

      evaluator when is_atom(evaluator) ->
        {evaluator, evaluate_batch(evaluator, samples, default_opts)}
    end)
    |> Map.new()
  end

  # ==========================================================================
  # Evaluator Helpers
  # ==========================================================================

  @doc """
  Create an exact match evaluator with options baked in.
  """
  @spec exact_match_evaluator(eval_opts()) :: (String.t(), String.t() -> eval_result())
  def exact_match_evaluator(opts \\ []) do
    fn output, expected -> evaluate(:exact_match, output, expected, opts) end
  end

  @doc """
  Create a contains evaluator with options baked in.
  """
  @spec contains_evaluator(eval_opts()) :: (String.t(), String.t() -> eval_result())
  def contains_evaluator(opts \\ []) do
    fn output, expected -> evaluate(:contains, output, expected, opts) end
  end

  @doc """
  Create a numeric evaluator with tolerance.
  """
  @spec numeric_evaluator(float()) :: (String.t(), String.t() -> eval_result())
  def numeric_evaluator(tolerance \\ 0.0001) do
    fn output, expected -> evaluate(:numeric, output, expected, tolerance: tolerance) end
  end

  @doc """
  Compose multiple evaluators with AND logic (all must pass).
  """
  @spec compose_and([evaluator_type() | (String.t(), String.t() -> eval_result())]) ::
          (String.t(), String.t() -> eval_result())
  def compose_and(evaluators) do
    fn output, expected ->
      results =
        Enum.map(evaluators, fn
          fun when is_function(fun) -> fun.(output, expected)
          evaluator -> evaluate(evaluator, output, expected, [])
        end)

      all_passed = Enum.all?(results, & &1.passed)
      mean_score = Enum.sum(Enum.map(results, & &1.score)) / length(results)

      %{
        score: mean_score,
        passed: all_passed,
        details: %{individual_results: results}
      }
    end
  end

  @doc """
  Compose multiple evaluators with OR logic (any must pass).
  """
  @spec compose_or([evaluator_type() | (String.t(), String.t() -> eval_result())]) ::
          (String.t(), String.t() -> eval_result())
  def compose_or(evaluators) do
    fn output, expected ->
      results =
        Enum.map(evaluators, fn
          fun when is_function(fun) -> fun.(output, expected)
          evaluator -> evaluate(evaluator, output, expected, [])
        end)

      any_passed = Enum.any?(results, & &1.passed)
      max_score = Enum.max(Enum.map(results, & &1.score))

      %{
        score: max_score,
        passed: any_passed,
        details: %{individual_results: results}
      }
    end
  end

  # ==========================================================================
  # Private Helpers
  # ==========================================================================

  defp normalize_string(s, case_sensitive, normalize_whitespace) do
    s
    |> then(fn str -> if case_sensitive, do: str, else: String.downcase(str) end)
    |> then(fn str ->
      if normalize_whitespace, do: String.replace(str, ~r/\s+/, " ") |> String.trim(), else: str
    end)
  end

  defp parse_number(s) when is_binary(s) do
    trimmed = String.trim(s)

    case Float.parse(trimmed) do
      {num, ""} -> {:ok, num}
      {num, _rest} -> {:ok, num}
      :error -> {:error, "Cannot parse as number: #{inspect(trimmed)}"}
    end
  end

  defp parse_number(n) when is_number(n), do: {:ok, n * 1.0}

  defp default_tokenize(text) do
    text
    |> String.downcase()
    |> String.replace(~r/[^\w\s]/, "")
    |> String.split(~r/\s+/, trim: true)
  end

  defp safe_div(_num, denom) when denom == 0 or denom == 0.0, do: 0.0
  defp safe_div(num, denom), do: num / denom

  defp extract_sample(%{output: output, expected: expected}), do: {output, expected}
  defp extract_sample({output, expected}), do: {output, expected}
end
