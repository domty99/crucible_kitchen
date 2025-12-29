# Crucible Kitchen: The Better ML Kitchen

## Core Thesis

**crucible_kitchen IS the hexagonal core** - a manifest-driven, adapter-based orchestration layer that unifies the crucible_* ecosystem while remaining lightweight itself.

### Design Principles

1. **Hexagonal Architecture**: Ports define interfaces, adapters implement backends
2. **Manifest-Driven Config**: Recipe configuration drives adapter selection
3. **Minimal Core**: crucible_kitchen stays lean; integrations live in crucible_* siblings
4. **Better Than tinker-cookbook**: Built-in experiment tracking, lineage, quality feedback loops

### Architecture Layers

```
┌─────────────────────────────────────────────────────────────┐
│                     tinkex_cookbook                         │
│              (Recipes, Application Config)                  │
├─────────────────────────────────────────────────────────────┤
│                     crucible_kitchen                        │
│  (Workflow DSL, Stage Behaviour, Orchestration Core)        │
├─────────────────────────────────────────────────────────────┤
│ crucible_train │ crucible_datasets │ crucible_model_registry│
│ crucible_feedback │ crucible_framework │ crucible_deployment │
├─────────────────────────────────────────────────────────────┤
│       tinkex      │    hf_hub/hf_datasets_ex   │   chz_ex   │
│     (Tinker API)  │   (HuggingFace Ecosystem)  │ (Configs)  │
└─────────────────────────────────────────────────────────────┘
```

### Key Integrations

| Integration | Location | Purpose |
|-------------|----------|---------|
| tinkex | Adapter in crucible_train | Training backend |
| hf_hub | Used by crucible_model_registry | Model storage |
| hf_datasets_ex | Used by crucible_datasets | Dataset loading |
| chz_ex | crucible_kitchen config layer | Typed config composition |

### MVP Scope for tinkex_cookbook

1. First recipe from tinker-cookbook working end-to-end
2. Workflow DSL defining stages
3. Tinker backend for training
4. Basic experiment tracking (crucible_framework)
5. Database optional for MVP (in-memory/noop adapters)

### Success Criteria

- [ ] All tests passing
- [ ] No compiler warnings
- [ ] No dialyzer issues
- [ ] No credo --strict issues
- [ ] First recipe executes successfully
- [ ] Documentation complete

---

## Work Log

### 2025-12-28 Session Start
- Dynamic repo refactor completed for crucible_framework, crucible_model_registry, crucible_feedback
- Beginning thesis-driven multi-agent refactor
