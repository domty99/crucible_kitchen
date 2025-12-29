defmodule CrucibleKitchen.Stages.Evaluate do
  @moduledoc """
  Stage for running evaluation during training.

  Placeholder for evaluation logic - can be customized per recipe.

  ## State Requirements

  - `:session` - Training session
  - `:eval_dataset_handle` - Optional eval dataset

  ## State Updates

  - `:eval_results` - Evaluation results
  """

  use CrucibleKitchen.Stage

  require Logger

  @impl true
  def name, do: :evaluate

  @impl true
  def execute(context) do
    global_step = get_state(context, :global_step, 0)

    Logger.info("[Evaluate] Running evaluation at step #{global_step}")

    # Placeholder - actual evaluation logic should be added
    # or this stage should be overridden by specific recipes
    context =
      context
      |> put_state(:eval_results, %{step: global_step, evaluated: true})
      |> record_metric(:eval_run, 1)

    {:ok, context}
  end
end
