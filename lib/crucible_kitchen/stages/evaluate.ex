defmodule CrucibleKitchen.Stages.Evaluate do
  @moduledoc """
  Stage for running evaluation during or after training.

  This stage loads the evaluation dataset, runs model predictions,
  computes metrics, and optionally generates a report.

  ## State Requirements

  - `:session` - Training session (for model access)
  - `:eval_dataset_handle` - Optional eval dataset (will load test split if not provided)

  ## State Updates

  - `:eval_results` - Map of computed metrics (accuracy, f1, precision, recall, etc.)
  - `:eval_report` - Generated evaluation report (if report generation enabled)

  ## Configuration Options

  - `:eval_samples` - Maximum samples to evaluate (default: 500)
  - `:eval_metrics` - Metrics to compute (default: [:accuracy, :f1, :precision, :recall])
  - `:eval_report_format` - Report format: :markdown, :json, :html (default: :markdown)
  - `:skip_eval` - Skip evaluation entirely (default: false)

  ## Telemetry Events

  Emits `[:crucible_kitchen, :eval, :complete]` with computed metrics.

  ## Usage

      workflow do
        # ... training stages ...
        stage(:save_final, SaveFinalWeights)
        stage(:evaluate, Evaluate)
        stage(:register_model, RegisterModel)
      end
  """

  use CrucibleKitchen.Stage

  require Logger

  @default_metrics [:accuracy, :f1, :precision, :recall]
  @default_eval_samples 500

  @impl true
  def name, do: :evaluate

  @impl true
  def execute(context) do
    if get_config(context, :skip_eval, false) do
      Logger.info("[Evaluate] Skipping evaluation (skip_eval: true)")
      {:ok, put_state(context, :eval_results, %{skipped: true})}
    else
      do_evaluate(context)
    end
  end

  @impl true
  def validate(_context) do
    # Evaluation can proceed with or without explicit eval dataset
    # We'll try to load test split from main dataset if needed
    :ok
  end

  # Perform evaluation
  defp do_evaluate(context) do
    global_step = get_state(context, :global_step, 0)
    Logger.info("[Evaluate] Running evaluation at step #{global_step}")

    case get_adapter(context, :evaluator) do
      nil ->
        # No evaluator adapter - use basic evaluation
        Logger.info("[Evaluate] No evaluator adapter, using basic evaluation")
        do_basic_evaluation(context)

      {evaluator, evaluator_opts} ->
        do_full_evaluation(context, evaluator, evaluator_opts)
    end
  end

  # Full evaluation with evaluator adapter
  defp do_full_evaluation(context, evaluator, evaluator_opts) do
    with {:ok, eval_dataset} <- load_eval_dataset(context),
         {:ok, results} <- run_evaluation(evaluator, evaluator_opts, context, eval_dataset),
         {:ok, report} <- maybe_generate_report(evaluator, evaluator_opts, context, results) do
      emit_eval_event(context, results)

      context =
        context
        |> put_state(:eval_results, results)
        |> put_state(:eval_report, report)
        |> record_metric(:eval_accuracy, Map.get(results, :accuracy, 0))
        |> record_metric(:eval_f1, Map.get(results, :f1, 0))
        |> record_metric(:eval_run, 1)

      Logger.info("[Evaluate] Evaluation complete: #{format_results(results)}")
      {:ok, context}
    else
      {:error, :no_eval_dataset} ->
        Logger.warning("[Evaluate] No evaluation dataset available, skipping")
        {:ok, put_state(context, :eval_results, %{skipped: true, reason: :no_dataset})}

      {:error, reason} ->
        Logger.error("[Evaluate] Evaluation failed: #{inspect(reason)}")
        {:error, {:evaluation_failed, reason}}
    end
  end

  # Basic evaluation without adapter (placeholder metrics)
  defp do_basic_evaluation(context) do
    global_step = get_state(context, :global_step, 0)

    results = %{
      step: global_step,
      evaluated: true,
      accuracy: 0.0,
      f1: 0.0,
      precision: 0.0,
      recall: 0.0,
      note: "Basic evaluation - no evaluator adapter configured"
    }

    emit_eval_event(context, results)

    context =
      context
      |> put_state(:eval_results, results)
      |> record_metric(:eval_run, 1)

    {:ok, context}
  end

  # Load evaluation dataset
  defp load_eval_dataset(context) do
    # Try to get pre-loaded eval dataset
    case get_state(context, :eval_dataset_handle) || get_state(context, :eval_dataset) do
      nil ->
        # Try to load test split from dataset store
        load_test_split(context)

      dataset ->
        {:ok, dataset}
    end
  end

  # Load test split from dataset store
  defp load_test_split(context) do
    case get_adapter(context, :dataset_store) do
      nil ->
        {:error, :no_eval_dataset}

      {dataset_store, store_opts} ->
        dataset_name = get_config(context, :dataset)
        eval_samples = get_config(context, :eval_samples, @default_eval_samples)

        if dataset_name do
          load_opts = Keyword.merge(store_opts, split: "test", limit: eval_samples)
          dataset_store.load(load_opts, dataset_name)
        else
          {:error, :no_eval_dataset}
        end
    end
  end

  # Run evaluation with adapter
  defp run_evaluation(evaluator, base_opts, context, eval_dataset) do
    metrics = get_config(context, :eval_metrics, @default_metrics)
    model = get_state(context, :session) || get_state(context, :trained_model)

    opts = Keyword.merge(base_opts, metrics: metrics)
    evaluator.evaluate(opts, model, eval_dataset)
  end

  # Maybe generate evaluation report
  defp maybe_generate_report(evaluator, base_opts, context, results) do
    format = get_config(context, :eval_report_format, :markdown)

    if get_config(context, :generate_eval_report, true) do
      opts = Keyword.merge(base_opts, format: format)
      evaluator.generate_report(opts, results)
    else
      {:ok, nil}
    end
  end

  # Emit telemetry event
  defp emit_eval_event(context, results) do
    measurements =
      results
      |> Map.take([:accuracy, :f1, :precision, :recall, :loss, :perplexity])
      |> Enum.reject(fn {_, v} -> is_nil(v) end)
      |> Enum.into(%{})

    metadata = %{
      model: get_config(context, :model),
      dataset: get_config(context, :dataset),
      step: get_state(context, :global_step, 0),
      sample_count: Map.get(results, :sample_count, 0)
    }

    :telemetry.execute(
      [:crucible_kitchen, :eval, :complete],
      measurements,
      metadata
    )
  end

  # Format results for logging
  defp format_results(results) do
    results
    |> Map.take([:accuracy, :f1, :precision, :recall])
    |> Enum.map_join(" ", fn {k, v} -> "#{k}=#{format_value(v)}" end)
  end

  defp format_value(v) when is_float(v), do: Float.round(v, 4)
  defp format_value(v), do: inspect(v)
end
