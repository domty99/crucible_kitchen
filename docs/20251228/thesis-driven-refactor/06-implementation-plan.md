# Master Implementation Plan

## Goal
Refactor tinkex_cookbook to use crucible_kitchen as the core, with minimal code in tinkex_cookbook.

## Implementation Phases

### Phase 1: Create Adapters in crucible_kitchen (3-4 hours)

1. **Tinkex Training Adapter** (CRITICAL)
   - Implement in crucible_kitchen/adapters/tinkex
   - Implement CrucibleTrain.Ports.TrainingClient
   - Add session lifecycle management
   - Port chunking and sequencing logic

2. **HfHub Adapter**
   - Implement CrucibleTrain.Ports.HubClient
   - Delegate to hf_hub_ex
   - Add telemetry

3. **HfDatasets Adapter**
   - Implement CrucibleTrain.Ports.DatasetStore
   - Delegate to hf_datasets_ex
   - Schema conversion

### Phase 2: Refactor tinkex_cookbook (2-3 hours)

1. Update sl_basic recipe to use CrucibleKitchen.Recipe
2. Remove redundant adapters (keep Tinkex-specific only)
3. Remove Runtime.Manifests (use app config)
4. Update configuration pattern

### Phase 3: Integration Testing (2 hours)

1. Test sl_basic recipe end-to-end
2. Verify all adapters work
3. Check noop fallbacks

### Phase 4: Validation (1 hour)

1. Run all tests
2. Check for warnings
3. Run dialyzer
4. Run credo --strict

## Key Files to Create

```
crucible_kitchen/lib/crucible_kitchen/adapters/
├── tinkex/
│   └── training_client.ex    # From tinkex_cookbook
├── hf_hub/
│   └── hub_client.ex         # New
├── hf_datasets/
│   └── dataset_store.ex      # New
└── local/
    └── blob_store.ex         # Existing or from tinkex_cookbook
```

## Database Decision for MVP

**Decision: NO database required for MVP**

Rationale:
- Training recipes are ephemeral (single-run)
- Results stored via blob_store (files)
- Metrics exported via telemetry
- crucible_framework/model_registry/feedback have opt-in database

## Success Criteria

- [ ] sl_basic recipe runs via `CrucibleKitchen.run/3`
- [ ] All 208+ crucible_kitchen tests pass
- [ ] tinkex_cookbook tests pass
- [ ] No compiler warnings
- [ ] No dialyzer issues
- [ ] No credo --strict issues
