# Ecosystem Integration Plan

**Purpose:** Define how crucible_kitchen integrates with the FULL North-Shore-AI ecosystem.

---

## The Complete Ecosystem

crucible_kitchen is the orchestration layer that composes ALL crucible components:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           CRUCIBLE KITCHEN                                   │
│                    (Orchestration & Composition)                             │
└─────────────────────────────────────────────────────────────────────────────┘
        │
        ├── CORE INFRASTRUCTURE
        │   ├── crucible_ir           → Experiment specs, Training.Config
        │   ├── crucible_framework    → Pipeline runner, Stage behaviour
        │   └── crucible_bench        → Statistical testing, power analysis
        │
        ├── TRAINING INFRASTRUCTURE
        │   └── crucible_train        → Types, Renderers, Training loops, Ports
        │
        ├── MLOPS LAYER
        │   ├── crucible_model_registry → Artifact storage, versioning, lineage
        │   ├── crucible_deployment     → vLLM/Ollama/TGI, canary, blue-green
        │   └── crucible_feedback       → Production monitoring, drift, curation
        │
        ├── OBSERVABILITY
        │   ├── crucible_telemetry    → Metrics collection, streaming
        │   ├── crucible_trace        → Causal reasoning chains
        │   ├── crucible_harness      → Batch experiments, reports
        │   └── crucible_datasets     → Dataset management (GSM8K, HumanEval, MMLU)
        │
        ├── RELIABILITY
        │   ├── crucible_ensemble     → Multi-model voting strategies
        │   ├── crucible_hedging      → Request hedging strategies
        │   ├── crucible_adversary    → Adversarial testing
        │   └── crucible_xai          → Explainability (LIME, SHAP, PDP)
        │
        ├── HUGGINGFACE
        │   ├── hf_hub_ex             → Model/dataset download
        │   └── hf_datasets_ex        → Dataset loading
        │
        ├── EVALUATION & CONFIG
        │   ├── eval_ex               → inspect-ai parity
        │   └── chz_ex                → Dataclass config schemas
        │
        ├── TOKENIZATION
        │   └── tiktoken_ex           → Tiktoken bindings
        │
        ├── PYTHON BRIDGE
        │   └── snakebridge              → gRPC bridge to Python (sympy, pylatexenc)
        │
        └── BACKEND SDKs
            └── tinkex                → Tinker ML platform SDK
```

---

## Integration Categories

### Tier 1: Core (Always Required)

| Package | Integration Point | Kitchen Usage |
|---------|------------------|---------------|
| `crucible_ir` | `CrucibleIR.Experiment`, `Training.Config` | Workflow specs, serialization |
| `crucible_framework` | `Crucible.Pipeline.Runner` | Optional pipeline execution mode |
| `crucible_train` | Types, Renderers, Ports | Core training primitives |

### Tier 2: Observability (Default Enabled)

| Package | Integration Point | Kitchen Usage |
|---------|------------------|---------------|
| `crucible_telemetry` | Telemetry handlers | Metrics collection, streaming |
| `crucible_trace` | Trace.log/2 | Causal chain logging |
| `crucible_harness` | Harness.run/2 | Batch experiment orchestration |
| `crucible_datasets` | Dataset loaders | GSM8K, HumanEval, MMLU, etc. |

### Tier 3: MLOps (Optional)

| Package | Integration Point | Kitchen Usage |
|---------|------------------|---------------|
| `crucible_model_registry` | Registry.register/2 | Artifact versioning |
| `crucible_deployment` | Deploy.deploy/2 | Model serving |
| `crucible_feedback` | Feedback.ingest/2 | Production signals |

### Tier 4: Reliability (Optional)

| Package | Integration Point | Kitchen Usage |
|---------|------------------|---------------|
| `crucible_ensemble` | Ensemble.vote/2 | Multi-model inference |
| `crucible_hedging` | Hedging.request/2 | Latency reduction |
| `crucible_adversary` | Adversary.attack/2 | Robustness testing |
| `crucible_xai` | XAI.explain/2 | Model explanations |
| `crucible_bench` | Bench.compare/2 | Statistical significance |

### Tier 5: Data & Config

| Package | Integration Point | Kitchen Usage |
|---------|------------------|---------------|
| `hf_hub_ex` | HfHub.download/2 | Model downloads |
| `hf_datasets_ex` | Datasets.load/2 | Dataset loading |
| `eval_ex` | Eval.run/2 | Benchmark evaluation |
| `chz_ex` | Config schemas | CLI argument parsing |
| `tiktoken_ex` | Tokenizer | Token counting |
| `snakebridge` | Python bridge | Math verification, etc. |

---

## Kitchen Module Structure

Each ecosystem component maps to a Kitchen module:

```
lib/crucible_kitchen/
├── integrations/
│   ├── crucible_ir.ex         # IR type conversions
│   ├── crucible_framework.ex  # Pipeline interop
│   ├── crucible_train.ex      # Training primitives
│   ├── crucible_telemetry.ex  # Telemetry wiring
│   ├── crucible_harness.ex    # Batch experiments
│   ├── crucible_registry.ex   # Model registry
│   ├── crucible_deployment.ex # Deployment
│   ├── crucible_feedback.ex   # Feedback loop
│   ├── crucible_ensemble.ex   # Ensemble voting
│   ├── crucible_hedging.ex    # Request hedging
│   ├── hf_hub.ex              # HuggingFace Hub
│   ├── hf_datasets.ex         # HuggingFace Datasets
│   ├── eval_ex.ex             # Evaluation
│   ├── chz_ex.ex              # Config schemas
│   ├── tiktoken.ex            # Tokenization
│   └── snakebridge.ex            # Python bridge
```

---

## Port-to-Package Mapping

Kitchen Ports map to ecosystem packages:

| Port | Primary Implementation | Package |
|------|----------------------|---------|
| `TrainingClient` | Tinkex adapter | tinkex |
| `DatasetStore` | HfDatasets adapter | hf_datasets_ex |
| `BlobStore` | Local/S3 | crucible_model_registry |
| `HubClient` | HfHub adapter | hf_hub_ex |
| `MetricsStore` | Telemetry adapter | crucible_telemetry |
| `Completer` | Sampling adapter | tinkex |
| `VectorStore` | Chroma adapter | (external) |
| `EmbeddingClient` | HfHub adapter | hf_hub_ex |
| `PythonBridge` | Snakepit adapter | snakebridge |

Tokenizer access is adapter-specific; `tiktoken_ex` is used by tokenizer-capable adapters and stages.

---

## Stage-to-Package Mapping

Built-in stages use specific packages:

| Stage | Primary Package | Usage |
|-------|----------------|-------|
| `LoadDataset` | hf_datasets_ex | Load HuggingFace datasets |
| `InitSession` | tinkex (via port) | Start training session |
| `InitTokenizer` | training adapter / tiktoken_ex | Load tokenizer |
| `BuildSupervisedDataset` | crucible_train | Dataset construction |
| `ForwardBackward` | tinkex (via port) | Gradient computation |
| `OptimStep` | tinkex (via port) | Parameter update |
| `Evaluate` | eval_ex | Benchmark evaluation |
| `SaveCheckpoint` | crucible_model_registry | Artifact storage |
| `LogMetrics` | crucible_telemetry | Metrics streaming |
| `TraceReasoning` | crucible_trace | Causal logging |

---

## Workflow-to-Package Mapping

Workflows compose packages:

### Supervised Workflow
```
crucible_train.Types + Renderers
  → hf_datasets_ex (load data)
  → tiktoken_ex (tokenize)
  → crucible_train.Supervised.Dataset (batch)
  → tinkex (train via port)
  → crucible_telemetry (log)
  → crucible_model_registry (checkpoint)
```

### RL Workflow
```
crucible_train.RL.Env
  → tinkex (rollouts via port)
  → crucible_train.RL.Advantages (compute)
  → tinkex (PPO via port)
  → snakebridge (math verification for reward)
  → crucible_telemetry (log)
```

### Evaluation Workflow
```
eval_ex (benchmark definitions)
  → crucible_datasets (load benchmarks)
  → crucible_harness (batch orchestration)
  → crucible_bench (statistical analysis)
  → crucible_telemetry (results streaming)
```

### MLOps Workflow
```
crucible_model_registry (artifact versioning)
  → crucible_deployment (model serving)
  → crucible_feedback (production monitoring)
  → crucible_ensemble (A/B testing)
```

---

## Implementation Phases (Revised)

### Phase 1: Core Integration (Days 1-3)
- [ ] Wire crucible_ir for experiment specs
- [ ] Wire crucible_train for types, renderers, ports
- [ ] Wire crucible_framework for optional pipeline mode
- [ ] Wire crucible_bench for statistical testing

### Phase 2: Data Integration (Days 4-5)
- [ ] Wire hf_datasets_ex for dataset loading
- [ ] Wire hf_hub_ex for model downloads
- [ ] Wire tiktoken_ex for tokenization
- [ ] Wire chz_ex for config schemas

### Phase 3: Observability Integration (Days 6-7)
- [ ] Wire crucible_telemetry for metrics
- [ ] Wire crucible_trace for causal logging
- [ ] Wire crucible_harness for batch experiments
- [ ] Wire crucible_datasets for benchmark datasets

### Phase 4: Training Workflows (Days 8-10)
- [ ] Complete Supervised workflow with all integrations
- [ ] Complete RL workflow with snakebridge for math verification
- [ ] Complete DPO workflow
- [ ] Complete Distillation workflow

### Phase 5: MLOps Integration (Days 11-12)
- [ ] Wire crucible_model_registry for artifacts
- [ ] Wire crucible_deployment for serving
- [ ] Wire crucible_feedback for production loop

### Phase 6: Reliability Integration (Days 13-14)
- [ ] Wire crucible_ensemble for multi-model
- [ ] Wire crucible_hedging for latency reduction
- [ ] Wire crucible_adversary for robustness
- [ ] Wire crucible_xai for explainability

### Phase 7: Evaluation Integration (Days 15-16)
- [ ] Wire eval_ex for inspect-ai parity
- [ ] Create benchmark evaluation workflows
- [ ] Integrate with crucible_harness for batch runs

### Phase 8: tinkex_cookbook Migration (Days 17-20)
- [ ] Replace tinkex_cookbook training logic with Kitchen
- [ ] Implement Tinker adapters for all ports
- [ ] Migrate recipes to use Kitchen workflows
- [ ] Verify LOC < 2,000

---

## Dependency Graph

```
                              crucible_kitchen
                                    │
           ┌────────────────────────┼────────────────────────┐
           │                        │                        │
           ▼                        ▼                        ▼
    ┌──────────────┐        ┌──────────────┐        ┌──────────────┐
    │crucible_train│        │ crucible_ir  │        │crucible_     │
    │              │        │              │        │ framework    │
    └──────────────┘        └──────────────┘        └──────────────┘
           │                        │                        │
           │                        │                        │
    ┌──────┴──────┐          ┌──────┴──────┐          ┌──────┴──────┐
    │             │          │             │          │             │
    ▼             ▼          ▼             ▼          ▼             ▼
crucible_     hf_         chz_ex       eval_ex    crucible_    crucible_
telemetry   datasets_ex                          harness      bench
    │             │                                  │
    │             │                                  │
    ▼             ▼                                  ▼
crucible_    hf_hub_ex                          crucible_
trace                                           datasets
    │
    ▼
snakebridge ──────► tiktoken_ex
```

---

## Success Metrics

| Metric | Target |
|--------|--------|
| Ecosystem packages integrated | 20+ |
| Port implementations | 10+ |
| Built-in stages using integrations | 20+ |
| tinkex_cookbook LOC after migration | < 2,000 |
| Test coverage across integrations | > 85% |

---

## Notes

1. **Path-based deps for development** - All NSAI packages use `path:` deps during development
2. **Optional dependencies** - MLOps and Reliability tiers are optional
3. **Graceful degradation** - Kitchen works with minimal deps (just crucible_train + crucible_ir)
4. **Backend-agnostic** - Tinkex is a reference implementation, not a hard dependency
