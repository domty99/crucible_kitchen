# HuggingFace Libraries Analysis: Placement & Integration

## Executive Summary

The crucible ecosystem has two HuggingFace integration libraries:
1. **hf_hub_ex** (`:hf_hub`) - Low-level Hub API client
2. **hf_datasets_ex** - Mid-level Datasets library port

**Recommendation**: Keep as standalone Hex packages; crucible_kitchen integrates via adapters.

## 1. Current State

### hf_hub_ex (v0.1.1)
- Hub API client (model_info, dataset_info, downloads)
- LRU cache with GenServer
- Resume-capable streaming downloads
- Token-based authentication

### hf_datasets_ex (v0.1.1)
- Depends on hf_hub
- Dataset loading with config/split support
- Streaming Parquet/JSONL
- Built-in loaders: MMLU, HumanEval, GSM8K, etc.

## 2. Placement Decision

**Keep as standalone packages because:**
1. Published to Hex.pm independently
2. Designed for Python HuggingFace library parity
3. Used by multiple NSAI projects
4. Clean separation of concerns

## 3. Integration Architecture

```
crucible_kitchen (integration hub)
├── Ports (abstract interfaces)
│   ├── HubClient
│   └── DatasetStore
├── Adapters
│   ├── HfHub.HubClient ← hf_hub_ex
│   └── HfDatasets.DatasetStore ← hf_datasets_ex
```

## 4. Gaps vs tinkex_cookbook

| Feature | tinkex_cookbook | crucible_kitchen |
|---------|-----------------|------------------|
| HfHub adapter | Present | Noop only - **NEEDED** |
| HfDatasets adapter | Present | Noop only - **NEEDED** |
| Model registry HF | Not used | Stubbed - **NEEDED** |

## 5. Implementation Roadmap

### Phase 1: crucible_kitchen Adapters
```
lib/crucible_kitchen/adapters/hf_hub/hub_client.ex
lib/crucible_kitchen/adapters/hf_datasets/dataset_store.ex
```

### Phase 2: crucible_model_registry HF Storage
Implement actual `HuggingFace.Client` using hf_hub.

### Phase 3: Integration Tests
End-to-end: load dataset → train → register → upload

## 6. Configuration

```elixir
config :hf_hub,
  token: System.get_env("HF_TOKEN"),
  cache_dir: "~/.cache/huggingface"

config :crucible_kitchen,
  adapters: [
    hub_client: CrucibleKitchen.Adapters.HfHub.HubClient,
    dataset_store: CrucibleKitchen.Adapters.HfDatasets.DatasetStore
  ]
```
