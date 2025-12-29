defmodule CrucibleKitchen.Stages.OptimStep do
  @moduledoc """
  Stage for running the optimizer step.

  Applies gradients accumulated from forward-backward pass.
  Computes the current learning rate based on schedule.

  ## State Requirements

  - `:session` - Training session
  - `:global_step` - Current global step

  ## Configuration

  - `:learning_rate` - Base learning rate
  - `:lr_schedule` - Learning rate schedule (:constant, :linear, :cosine)
  - `:total_steps` - Total training steps (for schedule)

  ## State Updates

  - `:optim_future` - Future for the optim step result
  - `:current_lr` - Current learning rate used
  """

  use CrucibleKitchen.Stage

  alias CrucibleTrain.Ports.TrainingClient

  require Logger

  @impl true
  def name, do: :optim_step

  @impl true
  def execute(context) do
    session = get_state(context, :session)
    global_step = get_state(context, :global_step, 0)
    total_steps = get_state(context, :total_steps, 1)

    base_lr = get_config(context, :learning_rate, 2.0e-5)
    schedule = get_config(context, :lr_schedule, :linear)

    # Compute current learning rate
    lr = compute_lr(base_lr, global_step, total_steps, schedule)

    ports = get_train_ports(context)

    case TrainingClient.optim_step(ports, session, lr) do
      {:error, reason} ->
        Logger.error("[OptimStep] Failed: #{inspect(reason)}")
        {:error, {:optim_step_failed, reason}}

      future ->
        context =
          context
          |> put_state(:optim_future, future)
          |> put_state(:current_lr, lr)
          |> put_state(:global_step, global_step + 1)

        {:ok, context}
    end
  end

  defp compute_lr(base_lr, step, total_steps, schedule) do
    progress = step / max(total_steps - 1, 1)

    case schedule do
      :constant -> base_lr
      :linear -> base_lr * (1.0 - progress)
      :cosine -> base_lr * 0.5 * (1.0 + :math.cos(:math.pi() * progress))
      _ -> base_lr
    end
  end
end
