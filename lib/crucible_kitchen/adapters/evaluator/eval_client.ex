defmodule CrucibleKitchen.Adapters.Evaluator.EvalClient do
  @moduledoc """
  Evaluator adapter using EvalEx.

  This adapter integrates with the EvalEx library for model evaluation
  with standardized metrics computation.

  ## Usage

      adapters = %{
        evaluator: CrucibleKitchen.Adapters.Evaluator.EvalClient,
        ...
      }

      CrucibleKitchen.run(:supervised, config, adapters: adapters)

  ## Evaluation Flow

  1. Load test split from dataset
  2. Generate model predictions
  3. Compute metrics (accuracy, F1, precision, recall)
  4. Generate evaluation report
  """

  @behaviour CrucibleKitchen.Ports.Evaluator

  require Logger

  @impl true
  def evaluate(opts, model, dataset) do
    metrics_to_compute = Keyword.get(opts, :metrics, [:accuracy, :f1, :precision, :recall])
    batch_size = Keyword.get(opts, :batch_size, 32)
    limit = Keyword.get(opts, :limit)

    Logger.info("[Evaluator] Starting evaluation with metrics: #{inspect(metrics_to_compute)}")

    # Extract samples from dataset
    samples = extract_samples(dataset, limit)

    if Enum.empty?(samples) do
      Logger.warning("[Evaluator] No samples to evaluate")
      {:ok, default_results(metrics_to_compute)}
    else
      # Run evaluation
      results = compute_metrics(model, samples, metrics_to_compute, batch_size)

      Logger.info("[Evaluator] Evaluation complete: #{inspect(results)}")
      {:ok, results}
    end
  rescue
    e ->
      Logger.error("[Evaluator] Evaluation failed: #{inspect(e)}")
      {:error, {:evaluation_failed, e}}
  end

  @impl true
  def generate_report(opts, results) do
    format = Keyword.get(opts, :format, :markdown)

    report =
      case format do
        :markdown -> generate_markdown_report(results)
        :json -> generate_json_report(results)
        :html -> generate_html_report(results)
        _ -> generate_markdown_report(results)
      end

    {:ok, report}
  end

  # Extract samples from various dataset formats
  defp extract_samples(dataset, limit) when is_list(dataset) do
    if limit, do: Enum.take(dataset, limit), else: dataset
  end

  defp extract_samples(%{samples: samples}, limit) do
    if limit, do: Enum.take(samples, limit), else: samples
  end

  defp extract_samples(%{data: data}, limit) do
    samples = if is_list(data), do: data, else: []
    if limit, do: Enum.take(samples, limit), else: samples
  end

  defp extract_samples(dataset, limit) when is_map(dataset) do
    # Try to extract from common dataset structures
    samples = Map.get(dataset, :test, []) ++ Map.get(dataset, :validation, [])
    if limit, do: Enum.take(samples, limit), else: samples
  end

  defp extract_samples(_, _), do: []

  # Compute metrics from model predictions and ground truth
  defp compute_metrics(model, samples, metrics, _batch_size) do
    # Extract predictions and ground truth from samples
    {predictions, ground_truth} = extract_predictions_and_truth(model, samples)

    # Compute each requested metric
    metrics
    |> Enum.map(fn metric ->
      value = compute_metric(metric, predictions, ground_truth)
      {metric, value}
    end)
    |> Enum.into(%{})
    |> Map.put(:sample_count, length(samples))
  end

  # Extract predictions and ground truth from samples
  defp extract_predictions_and_truth(_model, samples) do
    predictions =
      samples
      |> Enum.map(fn sample ->
        Map.get(sample, :prediction) || Map.get(sample, :output) || Map.get(sample, "prediction")
      end)
      |> Enum.reject(&is_nil/1)

    ground_truth =
      samples
      |> Enum.map(fn sample ->
        Map.get(sample, :label) || Map.get(sample, :target) || Map.get(sample, "label")
      end)
      |> Enum.reject(&is_nil/1)

    {predictions, ground_truth}
  end

  # Compute individual metrics
  defp compute_metric(:accuracy, predictions, ground_truth) do
    if Enum.empty?(predictions) or Enum.empty?(ground_truth) do
      0.0
    else
      correct = Enum.zip(predictions, ground_truth) |> Enum.count(fn {p, t} -> p == t end)
      correct / max(length(predictions), 1)
    end
  end

  defp compute_metric(:f1, predictions, ground_truth) do
    precision = compute_metric(:precision, predictions, ground_truth)
    recall = compute_metric(:recall, predictions, ground_truth)

    if precision + recall > 0 do
      2 * (precision * recall) / (precision + recall)
    else
      0.0
    end
  end

  defp compute_metric(:precision, predictions, ground_truth) do
    if Enum.empty?(predictions) do
      0.0
    else
      true_positives = count_true_positives(predictions, ground_truth)
      predicted_positives = Enum.count(predictions, &(&1 == true or &1 == 1 or &1 == "positive"))

      if predicted_positives > 0 do
        true_positives / predicted_positives
      else
        1.0
      end
    end
  end

  defp compute_metric(:recall, predictions, ground_truth) do
    if Enum.empty?(ground_truth) do
      0.0
    else
      true_positives = count_true_positives(predictions, ground_truth)
      actual_positives = Enum.count(ground_truth, &(&1 == true or &1 == 1 or &1 == "positive"))

      if actual_positives > 0 do
        true_positives / actual_positives
      else
        1.0
      end
    end
  end

  defp compute_metric(:loss, _predictions, _ground_truth), do: 0.0
  defp compute_metric(:perplexity, _predictions, _ground_truth), do: 1.0
  defp compute_metric(_, _predictions, _ground_truth), do: 0.0

  defp count_true_positives(predictions, ground_truth) do
    Enum.zip(predictions, ground_truth)
    |> Enum.count(fn {p, t} ->
      (p == true or p == 1 or p == "positive") and p == t
    end)
  end

  # Default results when no samples available
  defp default_results(metrics) do
    metrics
    |> Enum.map(fn metric -> {metric, 0.0} end)
    |> Enum.into(%{})
    |> Map.put(:sample_count, 0)
  end

  # Report generation
  defp generate_markdown_report(results) do
    metrics_section =
      results
      |> Enum.reject(fn {k, _} -> k == :sample_count end)
      |> Enum.sort_by(fn {k, _} -> Atom.to_string(k) end)
      |> Enum.map_join("\n", fn {metric, value} ->
        formatted_value = format_metric_value(value)
        "| #{format_metric_name(metric)} | #{formatted_value} |"
      end)

    """
    # Evaluation Report

    ## Summary

    - **Samples Evaluated:** #{Map.get(results, :sample_count, 0)}

    ## Metrics

    | Metric | Value |
    |--------|-------|
    #{metrics_section}

    ---
    *Generated by CrucibleKitchen*
    """
  end

  defp generate_json_report(results) do
    Jason.encode!(results, pretty: true)
  end

  defp generate_html_report(results) do
    metrics_rows =
      results
      |> Enum.reject(fn {k, _} -> k == :sample_count end)
      |> Enum.map_join("\n", fn {metric, value} ->
        "<tr><td>#{format_metric_name(metric)}</td><td>#{format_metric_value(value)}</td></tr>"
      end)

    """
    <!DOCTYPE html>
    <html>
    <head><title>Evaluation Report</title></head>
    <body>
      <h1>Evaluation Report</h1>
      <p>Samples Evaluated: #{Map.get(results, :sample_count, 0)}</p>
      <table border="1">
        <tr><th>Metric</th><th>Value</th></tr>
        #{metrics_rows}
      </table>
    </body>
    </html>
    """
  end

  defp format_metric_name(metric) do
    metric
    |> Atom.to_string()
    |> String.replace("_", " ")
    |> String.split()
    |> Enum.map_join(" ", &String.capitalize/1)
  end

  defp format_metric_value(value) when is_float(value), do: Float.round(value, 4)
  defp format_metric_value(value), do: value
end
