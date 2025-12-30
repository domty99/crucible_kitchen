# CrucibleControl: Full-Stack Automated Experiment Control

**Date:** 2025-12-29
**Status:** Design Proposal
**Goal:** Close the loop from observation to automated action

---

## The Missing Piece

The crucible ecosystem has excellent **observability** but no **actuation**:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         CURRENT STATE                                    │
│                                                                          │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐          │
│  │ Training │───▶│ Metrics  │───▶│ Triggers │───▶│ Dashboard│          │
│  │ (Kitchen)│    │(Telemetry)    │(Feedback)│    │   (UI)   │          │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘          │
│                                                        │                 │
│                                                        ▼                 │
│                                                  ┌──────────┐           │
│                                                  │  HUMAN   │           │
│                                                  │ DECISION │           │
│                                                  └──────────┘           │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                         TARGET STATE                                     │
│                                                                          │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐          │
│  │ Training │───▶│ Metrics  │───▶│ Triggers │───▶│ CONTROL  │          │
│  │ (Kitchen)│    │(Telemetry)    │(Feedback)│    │  ENGINE  │          │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘          │
│       ▲                                               │                  │
│       │                                               ▼                  │
│       │          ┌──────────┐    ┌──────────┐    ┌──────────┐          │
│       └──────────│ Actions  │◀───│ Policies │◀───│  State   │          │
│                  │(Actuators)    │(Decisions)    │ Machine  │          │
│                  └──────────┘    └──────────┘    └──────────┘          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Architecture Overview

### New Package: `crucible_control`

```elixir
# Mix.exs
def project do
  [
    app: :crucible_control,
    version: "0.1.0",
    deps: [
      {:crucible_kitchen, "~> 0.1"},
      {:crucible_feedback, "~> 0.1"},
      {:crucible_model_registry, "~> 0.2"},
      {:crucible_telemetry, "~> 0.3"},
      {:oban, "~> 2.17"}  # Job scheduling
    ]
  ]
end
```

---

## Core Components

### 1. Experiment State Machine

```elixir
defmodule CrucibleControl.ExperimentFSM do
  @moduledoc """
  State machine for experiment lifecycle management.
  """

  use GenStateMachine, callback_mode: :state_functions

  # States
  @type state :: :pending | :queued | :running | :evaluating |
                 :promoting | :deployed | :failed | :cancelled

  # Events
  @type event :: :queue | :start | :complete | :fail | :cancel |
                 :evaluate | :promote | :deploy | :rollback

  defstruct [
    :experiment_id,
    :recipe,
    :config,
    :run_id,
    :metrics,
    :model_version,
    :started_at,
    :history
  ]

  # State transitions
  def pending(:queue, _from, data) do
    {:next_state, :queued, data, [{:reply, :ok}]}
  end

  def queued(:start, _from, data) do
    {:next_state, :running, %{data | started_at: DateTime.utc_now()}}
  end

  def running(:complete, _from, %{metrics: metrics} = data) do
    case evaluate_promotion_policy(metrics, data) do
      :promote -> {:next_state, :promoting, data}
      :evaluate -> {:next_state, :evaluating, data}
      :hold -> {:next_state, :pending, data}
    end
  end

  def evaluating(:pass, _from, data) do
    {:next_state, :promoting, data}
  end

  def promoting(:deploy, _from, data) do
    {:next_state, :deployed, data}
  end

  # Failure handling
  def running(:fail, _from, data) do
    {:next_state, :failed, %{data | history: [:failed | data.history]}}
  end

  # Any state can be cancelled
  def handle_event(:cancel, state, data) when state not in [:deployed, :cancelled] do
    {:next_state, :cancelled, data}
  end
end
```

### 2. Policy DSL

```elixir
defmodule CrucibleControl.Policy do
  @moduledoc """
  Declarative policies for automated experiment decisions.
  """

  defmacro policy(name, do: block) do
    quote do
      def unquote(name)() do
        unquote(block)
      end
    end
  end

  # Example policies
  defmodule DefaultPolicies do
    use CrucibleControl.Policy

    @doc "Auto-promote if accuracy > 0.9 and no regression"
    policy :auto_promote do
      %{
        conditions: [
          {:metric, :accuracy, :>, 0.9},
          {:metric, :f1, :>, 0.85},
          {:regression, :loss, :<, 0.05}  # < 5% regression
        ],
        actions: [:promote_to_staging],
        fallback: :hold_for_review
      }
    end

    @doc "Auto-retrain on drift detection"
    policy :drift_retrain do
      %{
        triggers: [:drift_detected],
        conditions: [
          {:drift_score, :>, 0.15},
          {:last_retrain, :older_than, hours: 24}
        ],
        actions: [
          {:curate_dataset, strategy: :hard_examples, count: 1000},
          {:queue_experiment, recipe: :supervised, priority: :high}
        ]
      }
    end

    @doc "Canary deployment policy"
    policy :canary_deploy do
      %{
        stages: [
          %{name: :canary_1pct, traffic: 0.01, duration: hours: 1},
          %{name: :canary_10pct, traffic: 0.10, duration: hours: 4},
          %{name: :full_rollout, traffic: 1.0}
        ],
        rollback_on: [
          {:metric, :error_rate, :>, 0.05},
          {:metric, :latency_p99, :>, :ms(500)}
        ]
      }
    end

    @doc "Cost-aware scheduling"
    policy :cost_aware_schedule do
      %{
        constraints: [
          {:daily_budget, :<=, :usd(100)},
          {:concurrent_runs, :<=, 3}
        ],
        priority_by: [
          {:metric_improvement_potential, :desc},
          {:estimated_cost, :asc}
        ]
      }
    end
  end
end
```

### 3. Control Engine

```elixir
defmodule CrucibleControl.Engine do
  @moduledoc """
  Central control loop that evaluates triggers and executes actions.
  """

  use GenServer

  alias CrucibleControl.{PolicyEvaluator, ActionExecutor, ExperimentFSM}
  alias CrucibleFeedback.Triggers
  alias CrucibleKitchen

  defstruct [
    :policies,
    :experiments,     # Map of experiment_id => FSM pid
    :action_queue,    # Pending actions
    :metrics_cache    # Recent metrics for decision making
  ]

  # Control loop - runs every N seconds
  def handle_info(:control_loop, state) do
    state
    |> check_triggers()
    |> evaluate_policies()
    |> execute_actions()
    |> schedule_next_loop()
  end

  defp check_triggers(state) do
    active_experiments = Map.keys(state.experiments)

    triggers =
      active_experiments
      |> Enum.flat_map(fn exp_id ->
        Triggers.check_all(exp_id, state.metrics_cache[exp_id])
      end)

    %{state | pending_triggers: triggers}
  end

  defp evaluate_policies(%{pending_triggers: triggers, policies: policies} = state) do
    actions =
      triggers
      |> Enum.flat_map(fn trigger ->
        PolicyEvaluator.evaluate(trigger, policies, state)
      end)

    %{state | action_queue: state.action_queue ++ actions}
  end

  defp execute_actions(%{action_queue: [action | rest]} = state) do
    case ActionExecutor.execute(action, state) do
      {:ok, result} ->
        Logger.info("Action executed: #{inspect(action)} -> #{inspect(result)}")
        emit_telemetry(:action_executed, action, result)

      {:error, reason} ->
        Logger.error("Action failed: #{inspect(action)} -> #{inspect(reason)}")
        emit_telemetry(:action_failed, action, reason)
    end

    %{state | action_queue: rest}
  end
end
```

### 4. Action Executors (Actuators)

```elixir
defmodule CrucibleControl.Actions do
  @moduledoc """
  Executable actions that the control engine can take.
  """

  # Training actions
  def queue_experiment(recipe, config, opts \\ []) do
    priority = Keyword.get(opts, :priority, :normal)

    Oban.insert(%Oban.Job{
      worker: CrucibleControl.Workers.TrainingWorker,
      args: %{recipe: recipe, config: config},
      priority: priority_to_int(priority),
      queue: :training
    })
  end

  def cancel_experiment(experiment_id) do
    with {:ok, fsm} <- get_fsm(experiment_id),
         :ok <- ExperimentFSM.cancel(fsm) do
      CrucibleKitchen.cancel(experiment_id)
    end
  end

  # Model registry actions
  def promote_to_staging(model_version_id) do
    CrucibleModelRegistry.promote(model_version_id, :staging)
  end

  def promote_to_production(model_version_id, opts \\ []) do
    if Keyword.get(opts, :canary, false) do
      start_canary_deployment(model_version_id, opts)
    else
      CrucibleModelRegistry.promote(model_version_id, :production)
    end
  end

  def rollback(model_version_id) do
    CrucibleModelRegistry.rollback(model_version_id)
  end

  # Data curation actions
  def curate_dataset(strategy, opts) do
    CrucibleFeedback.Curation.curate(strategy, opts)
  end

  # Deployment actions
  def start_canary_deployment(model_version_id, policy) do
    CrucibleDeployment.Canary.start(model_version_id, policy)
  end

  # Notification actions
  def notify(channel, message) do
    CrucibleControl.Notifier.send(channel, message)
  end
end
```

### 5. Scheduler (Oban-based)

```elixir
defmodule CrucibleControl.Scheduler do
  @moduledoc """
  Job scheduling for training runs and periodic tasks.
  """

  use Oban.Worker, queue: :training, max_attempts: 3

  @impl Oban.Worker
  def perform(%Oban.Job{args: %{"recipe" => recipe, "config" => config}}) do
    adapters = build_adapters(config)

    case CrucibleKitchen.run(recipe, config, adapters: adapters) do
      {:ok, result} ->
        # Update state machine
        ExperimentFSM.complete(result.experiment_id, result.metrics)
        :ok

      {:error, reason} ->
        ExperimentFSM.fail(result.experiment_id, reason)
        {:error, reason}
    end
  end

  # Periodic jobs
  defmodule PeriodicJobs do
    use Oban.Worker, queue: :control

    @impl true
    def perform(%{args: %{"job" => "check_drift"}}) do
      CrucibleControl.Engine.check_all_drift()
    end

    @impl true
    def perform(%{args: %{"job" => "cleanup_old_checkpoints"}}) do
      CrucibleControl.Maintenance.cleanup_checkpoints(older_than: days: 30)
    end
  end
end

# Oban config in application.ex
config :crucible_control, Oban,
  repo: CrucibleControl.Repo,
  queues: [training: 3, control: 1, notifications: 5],
  plugins: [
    {Oban.Plugins.Cron, crontab: [
      {"*/5 * * * *", CrucibleControl.PeriodicJobs, args: %{job: "check_drift"}},
      {"0 2 * * *", CrucibleControl.PeriodicJobs, args: %{job: "cleanup_old_checkpoints"}}
    ]}
  ]
```

### 6. UI Actuation Layer

```elixir
defmodule CrucibleUIWeb.ExperimentLive.Show do
  use CrucibleUIWeb, :live_view

  # Add action buttons
  def render(assigns) do
    ~H"""
    <div class="experiment-controls">
      <.button phx-click="start" disabled={@experiment.status != :pending}>
        Start Training
      </.button>

      <.button phx-click="cancel" disabled={@experiment.status in [:completed, :cancelled]}>
        Cancel
      </.button>

      <.button phx-click="promote" disabled={@experiment.status != :completed}>
        Promote to Staging
      </.button>

      <.button phx-click="retrain" disabled={@experiment.status != :completed}>
        Retrain with New Data
      </.button>

      <.button phx-click="rollback" disabled={!@can_rollback}>
        Rollback
      </.button>
    </div>

    <!-- Real-time metrics -->
    <div id="live-metrics" phx-hook="MetricsChart">
      <canvas id="loss-chart"></canvas>
    </div>

    <!-- Trigger status -->
    <div class="triggers">
      <h3>Active Triggers</h3>
      <%= for trigger <- @active_triggers do %>
        <div class={"trigger #{trigger.status}"}>
          <span><%= trigger.type %></span>
          <span><%= trigger.value %></span>
          <.button phx-click="act_on_trigger" phx-value-id={trigger.id}>
            Execute Policy
          </.button>
        </div>
      <% end %>
    </div>
    """
  end

  def handle_event("start", _, socket) do
    CrucibleControl.Actions.queue_experiment(
      socket.assigns.experiment.recipe,
      socket.assigns.experiment.config
    )
    {:noreply, assign(socket, :experiment, %{socket.assigns.experiment | status: :queued})}
  end

  def handle_event("promote", _, socket) do
    CrucibleControl.Actions.promote_to_staging(socket.assigns.experiment.model_version_id)
    {:noreply, socket}
  end

  def handle_event("act_on_trigger", %{"id" => trigger_id}, socket) do
    CrucibleControl.Engine.execute_trigger_policy(trigger_id)
    {:noreply, socket}
  end
end
```

---

## Integration Points

### Connecting the Pieces

```elixir
# In application.ex
def start(_type, _args) do
  children = [
    # Existing components
    CrucibleKitchen.Supervisor,
    CrucibleTelemetry.Supervisor,
    CrucibleFeedback.Supervisor,
    CrucibleModelRegistry.Supervisor,

    # New control layer
    {Oban, oban_config()},
    CrucibleControl.Engine,
    CrucibleControl.ExperimentRegistry,

    # UI with actions
    CrucibleUIWeb.Endpoint
  ]

  Supervisor.start_link(children, strategy: :one_for_one)
end
```

### Telemetry Integration

```elixir
defmodule CrucibleControl.TelemetryHandler do
  def attach do
    :telemetry.attach_many(
      "crucible-control-handler",
      [
        [:crucible_kitchen, :workflow, :stop],
        [:crucible_kitchen, :eval, :complete],
        [:crucible_feedback, :trigger, :fired],
        [:crucible_model_registry, :model, :promoted]
      ],
      &handle_event/4,
      nil
    )
  end

  def handle_event([:crucible_kitchen, :workflow, :stop], measurements, metadata, _) do
    CrucibleControl.Engine.on_workflow_complete(metadata.experiment_id, measurements)
  end

  def handle_event([:crucible_feedback, :trigger, :fired], _, metadata, _) do
    CrucibleControl.Engine.on_trigger(metadata.trigger)
  end
end
```

---

## Implementation Roadmap

### Phase 1: Core Control (Week 1-2)
- [ ] ExperimentFSM state machine
- [ ] Basic policy evaluator
- [ ] Action executors (queue, cancel, promote)
- [ ] Oban integration for scheduling

### Phase 2: Policy DSL (Week 3)
- [ ] Policy macro system
- [ ] Condition evaluators (metrics, time, regression)
- [ ] Default policies (auto-promote, drift-retrain)

### Phase 3: UI Actuation (Week 4)
- [ ] Action buttons in experiment views
- [ ] Real-time trigger display
- [ ] Manual policy execution
- [ ] Confirmation modals for destructive actions

### Phase 4: Advanced Automation (Week 5-6)
- [ ] Canary deployment integration
- [ ] Cost-aware scheduling
- [ ] Multi-experiment dependencies
- [ ] A/B test statistical decision automation

---

## Success Metrics

| Metric | Target |
|--------|--------|
| Time from trigger to action | < 5 minutes (automated) |
| Manual interventions needed | < 10% of experiments |
| Failed rollouts caught by canary | > 95% |
| Dashboard action response time | < 500ms |

---

## Summary

**What exists:** Excellent observation (metrics, artifacts, triggers, dashboards)

**What's needed:** A thin control layer that:
1. Maintains experiment state machines
2. Evaluates declarative policies
3. Executes actions (queue, promote, rollback, notify)
4. Exposes controls in the UI
5. Schedules periodic checks

**Estimated effort:** 4-6 weeks for full implementation

**Dependencies:** Oban for job scheduling, existing crucible_* packages
