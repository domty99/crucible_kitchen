defmodule CrucibleKitchen.LRSchedule do
  @moduledoc """
  Learning rate scheduling functions for training.

  This module provides common learning rate schedules used in deep learning,
  including linear decay, cosine annealing, and warmup strategies.

  ## Schedule Types

  - `:constant` - Fixed learning rate throughout training
  - `:linear` - Linear decay from initial to final LR
  - `:cosine` - Cosine annealing with optional restarts
  - `:exponential` - Exponential decay by a factor each step

  ## Warmup

  All schedules support optional warmup, which linearly ramps the LR from
  a fraction of the initial value up to the full value over a specified
  number of steps.

  ## Examples

      # Create a cosine schedule with warmup
      schedule = LRSchedule.new(:cosine,
        initial_lr: 5.0e-4,
        total_steps: 10000,
        warmup_steps: 500
      )

      # Get LR at different steps
      LRSchedule.get_lr(schedule, 0)     # ~0 (start of warmup)
      LRSchedule.get_lr(schedule, 500)   # 5.0e-4 (end of warmup)
      LRSchedule.get_lr(schedule, 5000)  # ~2.5e-4 (midpoint)
      LRSchedule.get_lr(schedule, 10000) # ~0 (end)

  ## Usage with Training

      for step <- 1..total_steps do
        lr = LRSchedule.get_lr(schedule, step)
        # Update optimizer with current LR
      end
  """

  @type schedule_type :: :constant | :linear | :cosine | :exponential | :polynomial

  @type t :: %__MODULE__{
          type: schedule_type(),
          initial_lr: float(),
          final_lr: float(),
          total_steps: pos_integer(),
          warmup_steps: non_neg_integer(),
          warmup_ratio: float(),
          # For exponential
          decay_rate: float(),
          # For polynomial
          power: float(),
          # For cosine restarts
          num_cycles: float()
        }

  defstruct type: :cosine,
            initial_lr: 5.0e-4,
            final_lr: 0.0,
            total_steps: 1000,
            warmup_steps: 0,
            warmup_ratio: 0.0,
            decay_rate: 0.9,
            power: 1.0,
            num_cycles: 0.5

  @doc """
  Create a new learning rate schedule.

  ## Parameters

  - `type` - Schedule type: `:constant`, `:linear`, `:cosine`, `:exponential`, `:polynomial`
  - `opts` - Options:
    - `:initial_lr` - Starting learning rate (default: 5.0e-4)
    - `:final_lr` - Ending learning rate (default: 0.0)
    - `:total_steps` - Total training steps (required for most schedules)
    - `:warmup_steps` - Number of warmup steps (default: 0)
    - `:warmup_ratio` - Warmup as ratio of total_steps (alternative to warmup_steps)
    - `:decay_rate` - For exponential: decay factor per step (default: 0.9)
    - `:power` - For polynomial: power of decay (default: 1.0 = linear)
    - `:num_cycles` - For cosine: number of cosine cycles (default: 0.5)

  ## Examples

      # Constant LR
      LRSchedule.new(:constant, initial_lr: 1.0e-4)

      # Linear decay with warmup
      LRSchedule.new(:linear,
        initial_lr: 5.0e-4,
        total_steps: 10000,
        warmup_steps: 500
      )

      # Cosine annealing with warmup ratio
      LRSchedule.new(:cosine,
        initial_lr: 5.0e-4,
        total_steps: 10000,
        warmup_ratio: 0.05
      )
  """
  @spec new(schedule_type(), keyword()) :: t()
  def new(type, opts \\ []) do
    total_steps = Keyword.get(opts, :total_steps, 1000)
    warmup_ratio = Keyword.get(opts, :warmup_ratio, 0.0)

    warmup_steps =
      case Keyword.get(opts, :warmup_steps) do
        nil when warmup_ratio > 0 -> round(total_steps * warmup_ratio)
        nil -> 0
        steps -> steps
      end

    %__MODULE__{
      type: type,
      initial_lr: Keyword.get(opts, :initial_lr, 5.0e-4),
      final_lr: Keyword.get(opts, :final_lr, 0.0),
      total_steps: total_steps,
      warmup_steps: warmup_steps,
      warmup_ratio: warmup_ratio,
      decay_rate: Keyword.get(opts, :decay_rate, 0.9),
      power: Keyword.get(opts, :power, 1.0),
      num_cycles: Keyword.get(opts, :num_cycles, 0.5)
    }
  end

  @doc """
  Get the learning rate at a given step.

  ## Parameters

  - `schedule` - The schedule struct
  - `step` - Current training step (0-indexed)

  ## Returns

  The learning rate as a float.
  """
  @spec get_lr(t(), non_neg_integer()) :: float()
  def get_lr(%__MODULE__{} = schedule, step) do
    # Handle warmup first
    if step < schedule.warmup_steps do
      warmup_lr(schedule, step)
    else
      decay_lr(schedule, step)
    end
  end

  @doc """
  Get a list of learning rates for all steps.

  Useful for visualization or precomputing the schedule.
  """
  @spec get_lr_schedule(t()) :: [float()]
  def get_lr_schedule(%__MODULE__{} = schedule) do
    for step <- 0..(schedule.total_steps - 1) do
      get_lr(schedule, step)
    end
  end

  @doc """
  Create a constant schedule (no decay).
  """
  @spec constant(keyword()) :: t()
  def constant(opts \\ []) do
    new(:constant, opts)
  end

  @doc """
  Create a linear decay schedule.

  Linear interpolation from initial_lr to final_lr over total_steps.
  """
  @spec linear(keyword()) :: t()
  def linear(opts \\ []) do
    new(:linear, opts)
  end

  @doc """
  Create a cosine annealing schedule.

  Follows a cosine curve from initial_lr to final_lr.
  The `num_cycles` option controls how many complete cosine cycles occur.
  Default is 0.5 (half cosine, smooth decay).
  """
  @spec cosine(keyword()) :: t()
  def cosine(opts \\ []) do
    new(:cosine, opts)
  end

  @doc """
  Create an exponential decay schedule.

  LR decays by decay_rate each step: lr = initial_lr * (decay_rate ^ step)
  """
  @spec exponential(keyword()) :: t()
  def exponential(opts \\ []) do
    new(:exponential, opts)
  end

  @doc """
  Create a polynomial decay schedule.

  LR decays as: initial_lr * (1 - progress)^power + final_lr * progress^power
  When power=1.0, this is equivalent to linear decay.
  """
  @spec polynomial(keyword()) :: t()
  def polynomial(opts \\ []) do
    new(:polynomial, opts)
  end

  @doc """
  Create a warmup-only schedule that linearly ramps to initial_lr.

  After warmup, maintains constant LR.
  """
  @spec warmup_constant(keyword()) :: t()
  def warmup_constant(opts \\ []) do
    warmup_steps = Keyword.get(opts, :warmup_steps, 100)
    new(:constant, Keyword.put(opts, :warmup_steps, warmup_steps))
  end

  @doc """
  Create a warmup + linear decay schedule.

  Common schedule for fine-tuning: warmup, then linear decay to final_lr.
  """
  @spec warmup_linear(keyword()) :: t()
  def warmup_linear(opts \\ []) do
    warmup_steps = Keyword.get(opts, :warmup_steps, 100)
    new(:linear, Keyword.put(opts, :warmup_steps, warmup_steps))
  end

  @doc """
  Create a warmup + cosine decay schedule.

  Popular schedule for transformers: warmup, then cosine annealing.
  """
  @spec warmup_cosine(keyword()) :: t()
  def warmup_cosine(opts \\ []) do
    warmup_steps = Keyword.get(opts, :warmup_steps, 100)
    new(:cosine, Keyword.put(opts, :warmup_steps, warmup_steps))
  end

  # ==========================================================================
  # Private Helpers
  # ==========================================================================

  defp warmup_lr(%__MODULE__{} = schedule, step) do
    # Linear warmup from 0 to initial_lr
    progress = step / max(schedule.warmup_steps, 1)
    schedule.initial_lr * progress
  end

  defp decay_lr(%__MODULE__{type: :constant} = schedule, _step) do
    schedule.initial_lr
  end

  defp decay_lr(%__MODULE__{type: :linear} = schedule, step) do
    progress = decay_progress(schedule, step)
    schedule.initial_lr + (schedule.final_lr - schedule.initial_lr) * progress
  end

  defp decay_lr(%__MODULE__{type: :cosine} = schedule, step) do
    progress = decay_progress(schedule, step)
    # Cosine annealing formula
    cos_value = :math.cos(:math.pi() * schedule.num_cycles * 2.0 * progress)
    range = schedule.initial_lr - schedule.final_lr
    schedule.final_lr + range * 0.5 * (1.0 + cos_value)
  end

  defp decay_lr(%__MODULE__{type: :exponential} = schedule, step) do
    decay_step = step - schedule.warmup_steps
    schedule.initial_lr * :math.pow(schedule.decay_rate, decay_step)
  end

  defp decay_lr(%__MODULE__{type: :polynomial} = schedule, step) do
    progress = decay_progress(schedule, step)
    decay_factor = :math.pow(1.0 - progress, schedule.power)
    schedule.initial_lr * decay_factor + schedule.final_lr * (1.0 - decay_factor)
  end

  defp decay_progress(%__MODULE__{} = schedule, step) do
    # Progress through the decay phase (0.0 to 1.0)
    decay_step = step - schedule.warmup_steps
    decay_total = schedule.total_steps - schedule.warmup_steps
    min(1.0, max(0.0, decay_step / max(decay_total, 1)))
  end
end
