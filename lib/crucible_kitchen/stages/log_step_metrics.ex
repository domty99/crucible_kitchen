defmodule CrucibleKitchen.Stages.LogStepMetrics do
  @moduledoc """
  Stage for logging metrics at each training step.

  Logs progress and metrics using telemetry and standard logging.

  ## State Requirements

  - `:global_step` - Current global step
  - `:current_lr` - Current learning rate (optional)
  - `:total_steps` - Total steps (optional)

  ## Configuration

  - `:log_every` - Log every N steps (default: 10)
  """

  use CrucibleKitchen.Stage

  require Logger

  @impl true
  def name, do: :log_step_metrics

  @impl true
  def execute(context) do
    global_step = get_state(context, :global_step, 0)
    total_steps = get_state(context, :total_steps, 0)
    current_lr = get_state(context, :current_lr, 0.0)
    log_every = get_config(context, :log_every, 10)

    if log_every > 0 and rem(global_step, log_every) == 0 do
      progress =
        if total_steps > 0, do: Float.round(global_step / total_steps * 100, 1), else: 0.0

      Logger.info(
        "[Step #{global_step}/#{total_steps}] Progress: #{progress}% | LR: #{Float.round(current_lr, 8)}"
      )
    end

    context = record_metric(context, :step, global_step)
    {:ok, context}
  end
end
