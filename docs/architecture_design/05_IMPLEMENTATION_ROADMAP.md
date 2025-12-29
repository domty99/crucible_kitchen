# Implementation Roadmap

**Purpose:** Phased delivery plan for crucible_kitchen.

---

## Overview

```
Phase 0 (Day 1)      → Project setup, core scaffolding
Phase 1 (Days 2-3)   → Port definitions, Context, basic Runner
Phase 2 (Days 4-5)   → Workflow DSL, built-in stages
Phase 3 (Days 6-7)   → Supervised workflow, telemetry
Phase 4 (Days 8-9)   → tinkex_cookbook migration
Phase 5 (Days 10-12) → RL/DPO/Distillation workflows
Phase 6 (Days 13-14) → Documentation, polish, release
```

---

## Phase 0: Project Setup (Day 1)

### Deliverables
- [x] Create `commercial_kitchen` design directory
- [ ] Create GitHub repo `crucible_kitchen`
- [ ] Initialize Elixir project with proper structure
- [ ] Set up CI/CD (GitHub Actions)
- [ ] Configure mix.exs with dependencies

### Tasks

```bash
# Create GitHub repo
gh repo create North-Shore-AI/crucible_kitchen \
  --public \
  --description "Industrial ML training orchestration - backend-agnostic workflows for supervised, reinforcement, and preference learning" \
  --homepage "https://github.com/North-Shore-AI/crucible_kitchen"

# Initialize Elixir project
mix new crucible_kitchen --sup
cd crucible_kitchen

# Add dependencies (mix.exs)
defp deps do
  [
    {:crucible_train, "~> 0.2.0"},
    {:crucible_ir, "~> 0.2.0"},
    {:telemetry, "~> 1.2"},
    {:jason, "~> 1.4"},

    # Dev/test
    {:ex_doc, "~> 0.31", only: :dev},
    {:dialyxir, "~> 1.4", only: [:dev, :test], runtime: false},
    {:credo, "~> 1.7", only: [:dev, :test]},
    {:mox, "~> 1.1", only: :test}
  ]
end
```

### Directory Structure

```
crucible_kitchen/
├── lib/
│   ├── crucible_kitchen.ex
│   └── crucible_kitchen/
│       ├── application.ex
│       └── (empty, to be filled in later phases)
├── test/
│   ├── test_helper.exs
│   └── crucible_kitchen_test.exs
├── mix.exs
├── README.md
├── CHANGELOG.md
└── .github/
    └── workflows/
        └── ci.yml
```

---

## Phase 1: Core Infrastructure (Days 2-3)

### Deliverables
- [ ] Port integrations (CrucibleTrain + CrucibleTelemetry)
- [ ] Noop adapters for train/telemetry ports (plus Completer)
- [ ] Context module
- [ ] Stage behaviour
- [ ] Basic stage runner (no workflow DSL yet)

### Files to Create

```
lib/crucible_kitchen/
├── ports/
│   └── completer.ex
├── adapters/
│   └── noop/
│       ├── training_client.ex
│       ├── dataset_store.ex
│       ├── blob_store.ex
│       ├── hub_client.ex
│       ├── metrics_store.ex
│       ├── completer.ex
│       └── tokenizer_client.ex # adapter extension, not a port
├── stage/
│   ├── stage.ex          # Behaviour
│   └── context.ex        # Flowing state
└── runner/
    └── stage_runner.ex   # Execute single stage
```

### Tests

```elixir
# test/crucible_kitchen/ports/training_client_test.exs
defmodule CrucibleTrain.Ports.TrainingClientTest do
  use ExUnit.Case

  test "noop adapter implements all callbacks" do
    adapter = CrucibleKitchen.Adapters.Noop.TrainingClient
    assert CrucibleKitchen.Adapters.implements?(adapter, CrucibleTrain.Ports.TrainingClient)
  end
end

# test/crucible_kitchen/stage/context_test.exs
defmodule CrucibleKitchen.Stage.ContextTest do
  use ExUnit.Case

  test "context flows state through stages" do
    ctx = Context.new(%{}, %{training_client: NoopAdapter})
    ctx = Context.put_state(ctx, :foo, :bar)
    assert Context.get_state(ctx, :foo) == :bar
  end
end
```

---

## Phase 2: Workflow DSL (Days 4-5)

### Deliverables
- [ ] Workflow behaviour and DSL macros
- [ ] Workflow builder (DSL -> IR)
- [ ] Workflow runner
- [ ] Control flow: `stage`, `loop`, `conditional`, `parallel`

### Files to Create

```
lib/crucible_kitchen/
├── workflow/
│   ├── workflow.ex       # Behaviour + use macro
│   ├── dsl.ex            # DSL macros
│   ├── builder.ex        # Compile to IR
│   └── runner.ex         # Execute workflow
└── stage/
    └── builtins/
        └── noop_stage.ex # For testing
```

### Tests

```elixir
defmodule TestWorkflow do
  use CrucibleKitchen.Workflow

  workflow do
    stage :step1, NoopStage
    stage :step2, NoopStage

    loop :items, over: fn ctx -> ctx.config[:items] || [] end do
      stage :process, NoopStage
    end

    conditional fn ctx -> ctx.config[:do_final] end do
      stage :final, NoopStage
    end
  end
end

defmodule CrucibleKitchen.Workflow.RunnerTest do
  use ExUnit.Case

  test "executes stages in order" do
    ctx = Context.new(%{items: [1,2,3], do_final: true}, noop_adapters())
    {:ok, result} = Workflow.Runner.run(TestWorkflow, ctx)
    assert result.state[:stages_executed] == [:step1, :step2, :process, :process, :process, :final]
  end
end
```

---

## Phase 3: Supervised Workflow (Days 6-7)

### Deliverables
- [ ] All supervised training stages
- [ ] `Workflows.Supervised` workflow
- [ ] Telemetry integration
- [ ] Public `CrucibleKitchen.run/3` API

### Files to Create

```
lib/crucible_kitchen/
├── stage/
│   └── builtins/
│       ├── load_dataset.ex
│       ├── init_session.ex
│       ├── init_tokenizer.ex
│       ├── build_supervised_dataset.ex
│       ├── set_epoch.ex
│       ├── get_batch.ex
│       ├── forward_backward.ex
│       ├── await_future.ex
│       ├── optim_step.ex
│       ├── log_step_metrics.ex
│       ├── log_epoch_metrics.ex
│       ├── save_checkpoint.ex
│       ├── evaluate.ex
│       ├── save_final_weights.ex
│       └── cleanup.ex
├── workflow/
│   └── builtins/
│       └── supervised.ex
├── telemetry/
│   ├── events.ex
│   └── handlers/
│       ├── console.ex
│       └── jsonl.ex
└── crucible_kitchen.ex   # Public facade
```

### Integration Test

```elixir
defmodule CrucibleKitchen.Integration.SupervisedTest do
  use ExUnit.Case

  @tag :integration
  test "supervised workflow runs end-to-end with noop adapters" do
    config = %{
      model: "test-model",
      epochs: 2,
      batch_size: 4,
      learning_rate: 1.0e-4
    }

    {:ok, result} = CrucibleKitchen.run(:supervised, config,
      adapters: CrucibleKitchen.Adapters.noop())

    assert result.context.state[:final_weights_saved]
    assert length(result.metrics) > 0
  end
end
```

---

## Phase 4: tinkex_cookbook Migration (Days 8-9)

### Deliverables
- [ ] Move `sl_basic` to use `CrucibleKitchen.run/3`
- [ ] Tinker adapter implements Kitchen ports
- [ ] tinkex_cookbook LOC < 2,000
- [ ] End-to-end test with live Tinker

### Changes to tinkex_cookbook

```elixir
# BEFORE (sl_basic.ex - 400 lines of training loop)
def run_training(config, opts) do
  # Hand-rolled training loop
  case TinkexAdapter.start_session(...) do
    {:ok, session} ->
      # ... 300+ lines of training logic
  end
end

# AFTER (sl_basic.ex - 50 lines)
defmodule TinkexCookbook.Recipes.SlBasic do
  use CrucibleKitchen.Recipe

  def name, do: :sl_basic
  def description, do: "Basic supervised fine-tuning"
  def workflow, do: CrucibleKitchen.Workflows.Supervised

  def defaults do
    %{
      model: "meta-llama/Llama-3.1-8B",
      epochs: 1,
      batch_size: 128,
      learning_rate: 2.0e-4,
      lora_rank: 32
    }
  end
end

# Usage
CrucibleKitchen.run(TinkexCookbook.Recipes.SlBasic, config,
  adapters: %{
    training_client: {CrucibleKitchen.Adapters.Tinkex.TrainingClient, []},
    dataset_store: {CrucibleKitchen.Adapters.HfDatasets.DatasetStore, []}
  })
```

### Tinker Adapter

```elixir
defmodule CrucibleKitchen.Adapters.Tinkex.TrainingClient do
  @behaviour CrucibleTrain.Ports.TrainingClient

  @impl true
  def start_session(_opts, config) do
    Tinkex.TrainingClient.start(...)
  end

  @impl true
  def forward_backward(_opts, session, datums) do
    # Convert CrucibleTrain types to Tinkex types
    tinkex_datums = Enum.map(datums, &convert_datum/1)
    Tinkex.TrainingClient.forward_backward_async(session, tinkex_datums)
  end

  # ... implement all callbacks
end
```

---

## Phase 5: RL/DPO/Distillation (Days 10-12)

### Deliverables
- [ ] RL-specific stages (rollouts, advantages, PPO)
- [ ] `Workflows.Reinforcement` workflow
- [ ] DPO-specific stages
- [ ] `Workflows.Preference` workflow
- [ ] Distillation workflow

### Files to Create

```
lib/crucible_kitchen/
├── stage/
│   └── builtins/
│       ├── rl/
│       │   ├── load_environments.ex
│       │   ├── collect_rollouts.ex
│       │   ├── compute_advantages.ex
│       │   └── forward_backward_ppo.ex
│       ├── dpo/
│       │   ├── load_preference_dataset.ex
│       │   ├── init_reference_model.ex
│       │   └── forward_backward_dpo.ex
│       └── distillation/
│           ├── init_teacher.ex
│           └── forward_backward_kl.ex
└── workflow/
    └── builtins/
        ├── reinforcement.ex
        ├── preference.ex
        └── distillation.ex
```

---

## Phase 6: Polish & Release (Days 13-14)

### Deliverables
- [ ] Complete documentation (moduledocs, guides)
- [ ] API reference published to HexDocs
- [ ] README with examples
- [ ] CHANGELOG
- [ ] v0.1.0 release to Hex

### Documentation

```
docs/
├── guides/
│   ├── getting_started.md
│   ├── custom_workflows.md
│   ├── custom_stages.md
│   ├── adapters.md
│   └── telemetry.md
├── architecture/
│   └── (copy from commercial_kitchen/docs)
└── api/
    └── (generated by ExDoc)
```

### Release Checklist

- [ ] All tests passing
- [ ] Dialyzer clean
- [ ] Credo clean
- [ ] Documentation complete
- [ ] CHANGELOG updated
- [ ] Version bumped
- [ ] GitHub release created
- [ ] Published to Hex

---

## Success Metrics

| Metric | Phase 3 | Phase 4 | Phase 6 |
|--------|---------|---------|---------|
| crucible_kitchen LOC | ~2,000 | ~3,000 | ~4,000 |
| tinkex_cookbook LOC | 13,633 | <2,000 | <2,000 |
| Test coverage | >80% | >85% | >90% |
| Workflows | 1 (SL) | 1 (SL) | 4 (all) |
| Documentation | Minimal | Good | Complete |

---

## Dependencies Between Phases

```
Phase 0 ──→ Phase 1 ──→ Phase 2 ──→ Phase 3 ──→ Phase 4
                                        │
                                        └──→ Phase 5 ──→ Phase 6
```

- Phase 1 requires Phase 0 (project exists)
- Phase 2 requires Phase 1 (ports, context, stage behaviour)
- Phase 3 requires Phase 2 (workflow DSL)
- Phase 4 requires Phase 3 (supervised workflow works)
- Phase 5 can run parallel to Phase 4 (after Phase 3)
- Phase 6 requires Phases 4 & 5 complete

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Workflow DSL complexity | Start simple; iterate based on real needs |
| Port interface gaps | Design ports from tinkex_cookbook usage patterns |
| Telemetry overhead | Make telemetry opt-in with zero overhead when disabled |
| tinkex_cookbook migration breaks | Maintain backward compatibility during transition |
| RL/DPO complexity | Defer advanced features; MVP first |

---

## Open Questions

1. **Should `crucible_kitchen` own the ChzEx config schemas?**
   - Option A: Yes, provide base schemas
   - Option B: No, let cookbooks define their own

2. **How to handle multi-GPU/distributed training?**
   - Defer to Phase 7 or leave to adapters

3. **Should there be a `crucible_kitchen_ui` for dashboards?**
   - Separate project, not in initial scope

4. **Integration with `crucible_harness` for batch experiments?**
   - Design for interop but implement later
