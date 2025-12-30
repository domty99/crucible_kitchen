defmodule CrucibleKitchen.Ports.Evaluator do
  @moduledoc """
  Port for model evaluation.

  Implementations handle running evaluations against trained models,
  computing metrics, and generating reports.

  ## Implementation Example

      defmodule MyApp.Adapters.Evaluator do
        @behaviour CrucibleKitchen.Ports.Evaluator

        @impl true
        def evaluate(opts, model, dataset) do
          # Run evaluation and compute metrics
          {:ok, %{accuracy: 0.95, f1: 0.93, ...}}
        end

        @impl true
        def generate_report(opts, results) do
          # Generate evaluation report
          {:ok, "# Evaluation Report\\n..."}
        end
      end
  """

  @type opts :: keyword()
  @type model :: term()
  @type dataset :: term()
  @type results :: %{
          optional(:accuracy) => float(),
          optional(:f1) => float(),
          optional(:precision) => float(),
          optional(:recall) => float(),
          optional(:loss) => float(),
          optional(:perplexity) => float(),
          optional(atom()) => number()
        }
  @type report :: String.t()

  @doc """
  Evaluate a model against a dataset.

  ## Parameters

  - `opts` - Adapter-specific options:
    - `:metrics` - List of metrics to compute (e.g., `[:accuracy, :f1]`)
    - `:batch_size` - Evaluation batch size
    - `:limit` - Maximum samples to evaluate
  - `model` - Model to evaluate (session handle or model reference)
  - `dataset` - Dataset to evaluate against

  ## Returns

  - `{:ok, results}` - Map of computed metrics
  - `{:error, reason}` - Evaluation failure
  """
  @callback evaluate(opts(), model(), dataset()) :: {:ok, results()} | {:error, term()}

  @doc """
  Generate an evaluation report.

  ## Parameters

  - `opts` - Adapter-specific options:
    - `:format` - Report format (`:markdown`, `:html`, `:json`)
  - `results` - Evaluation results from `evaluate/3`

  ## Returns

  - `{:ok, report}` - Generated report string
  - `{:error, reason}` - Report generation failure
  """
  @callback generate_report(opts(), results()) :: {:ok, report()} | {:error, term()}

  @optional_callbacks []
end
