# Tinker-Cookbook Core Architecture Analysis

## Executive Summary

Tinker is a Python SDK for distributed ML training. Key innovations for crucible_kitchen:
1. **Async-first futures pattern** - Unified sync/async via APIFuture
2. **Intelligent request chunking** - Automatic batching (1024 items or 5MB)
3. **Sequenced execution** - Request ID + turn-based barrier ensures order
4. **Typed configuration hierarchy** - Pydantic validation at every layer

## 1. Core Modules & Responsibilities

### Client Architecture
```
ServiceClient (entry point)
├─ TrainingClient (~900 LOC)
│  ├─ forward_backward(data, loss_fn) → APIFuture
│  ├─ optim_step(adam_params) → APIFuture
│  └─ save_weights_and_get_sampling_client(name) → SamplingClient
├─ SamplingClient (text generation)
└─ RestClient (generic REST)
```

### Type System (~1600 LOC, 73 types)
| Type | Purpose |
|------|---------|
| Datum | Training example (model_input + loss_fn_inputs) |
| ModelInput | Tokenized sequence with chunks |
| TensorData | Binary tensor with shape/dtype |
| LoraConfig | LoRA training configuration |
| AdamParams | Optimizer parameters |
| Checkpoint | Checkpoint metadata |

## 2. Training Loop Patterns

### forward_backward Flow
```
Input: [Datum, Datum, ...] (arbitrary length)
   ↓ Chunking: Split by max 1024 items or ~5MB
   ↓ Request ID allocation (for ordering)
   ↓ Turn-based execution (async barrier)
   ↓ HTTP POST /api/v1/forward_backward
   ↓ Server returns: {loss, logprobs, per_token_metrics}
   ↓ Combine results via _CombinedAPIFuture
   ↓ Return: APIFuture[ForwardBackwardOutput]
```

### Critical Pattern: Chunking
```python
for datum in data:
    if len(chunk) >= 1024 or chunk_bytes + datum_bytes > 5_000_000:
        yield chunk  # Flush
        chunk = []
    chunk.append(datum)
```

### Critical Pattern: Request Sequencing
Turn-based barrier ensures FIFO order despite async execution.

## 3. Configuration Patterns

**Layered configuration:**
1. ServiceClient init (api_key, base_url, timeout)
2. create_lora_training_client(base_model, rank, seed)
3. forward_backward(data, loss_fn, loss_fn_config)
4. optim_step(AdamParams)

## 4. Data Pipeline

### Datum Construction
```python
datum = Datum(
    model_input=ModelInput.from_ints([1, 2, 3, 4]),
    loss_fn_inputs={
        "target_tokens": TensorData.from_torch(tensor)
    }
)
```

### LossFnInputs Keys
- target_tokens (int64)
- weights (float32)
- advantages (float32, for RL)
- logprobs (float32)

## 5. What Must Be Ported

| Item | Effort |
|------|--------|
| TrainingClient behaviour | Medium |
| Datum/ModelInput/TensorData types | Done |
| Chunking logic | Low |
| Request sequencing | Medium |
| Future/Task bridge | Low |

## 6. What Should Be Improved

| Tinker Limitation | Crucible Enhancement |
|-------------------|---------------------|
| Implicit session | Explicit create/close |
| Basic telemetry | Research-grade instrumentation |
| Path parsing only | Checkpoint versioning |
| HTTP exceptions | Structured errors |

## 7. File Locations

**Tinker Core:**
- `training_client.py` (~900 LOC) - Core training orchestration
- `types/datum.py`, `model_input.py`, `tensor_data.py` - Data types
- `types/lora_config.py`, `sampling_params.py` - Config types

**CrucibleKitchen (already in place):**
- `ports/training_client.ex` - TrainingClient behaviour
- `types.ex` - Type definitions (partial)
- `adapters/noop/training_client.ex` - Noop implementation

## 8. Data Flow Diagram

```
User Application
       ↓
CrucibleKitchen.run(:supervised, config, adapters)
       ↓
┌─────────────────────────────────────┐
│ Stage: LoadDataset → Datum[]       │
│ Stage: InitSession → session_id    │
│ Loop: Epoch 0..N                   │
│  ├─ TrainBatch → forward_backward  │
│  └─ Checkpoint → save_weights      │
└─────────────────────────────────────┘
       ↓
Ports.TrainingClient (Behaviour)
       ↓
  ┌────┴────┬────────────┐
  ↓         ↓            ↓
Tinkex   Fireworks    LocalNx
Adapter   Adapter     Adapter
```

---

**Key Insight:** Tinker's patterns (chunking, sequencing, futures) are solid and should be ported. Crucible should add explicit session management and checkpoint versioning.
