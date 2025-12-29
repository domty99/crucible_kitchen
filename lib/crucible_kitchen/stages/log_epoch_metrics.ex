defmodule CrucibleKitchen.Stages.LogEpochMetrics do
  @moduledoc """
  Stage for logging metrics at the end of each epoch.

  ## State Requirements

  - `:current_epoch` - Current epoch number
  - `:global_step` - Current global step
  """

  use CrucibleKitchen.Stage

  require Logger

  @impl true
  def name, do: :log_epoch_metrics

  @impl true
  def execute(context) do
    current_epoch = get_state(context, :current_epoch, 0)
    global_step = get_state(context, :global_step, 0)
    num_epochs = get_config(context, :epochs, 1)

    Logger.info(
      "[Epoch #{current_epoch + 1}/#{num_epochs}] Completed. Total steps: #{global_step}"
    )

    context =
      context
      |> record_metric(:epoch_completed, current_epoch + 1)
      |> record_metric(:steps_at_epoch_end, global_step)

    {:ok, context}
  end
end
