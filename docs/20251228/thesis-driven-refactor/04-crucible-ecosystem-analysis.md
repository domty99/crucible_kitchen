# Crucible Ecosystem State Analysis

## Executive Summary

The Crucible ecosystem is a **modern, modular ML orchestration platform** with hexagonal architecture.
**MVP Readiness**: 8/10 - Core pieces exist; adapters and integration testing needed.

## Library Overview

| Library | Version | Status | Test Files | LOC |
|---------|---------|--------|-----------|-----|
| crucible_kitchen | 0.1.0 | Development | 9 | ~2,500 |
| crucible_train | 0.2.0 | Stable | 34 | ~4,000 |
| crucible_datasets | 0.5.3 | Stable | 16 | ~2,200 |
| crucible_model_registry | 0.2.0 | Stable | 14 | Ecto |
| crucible_feedback | 0.2.0 | Stable | 22 | Ecto |
| crucible_framework | 0.5.1 | Stable | 14 | Ecto |
| crucible_deployment | 0.2.0 | Stable | 24 | - |
| crucible_ir | 0.2.1 | Stable | 19 | Pure data |

## crucible_kitchen (Core)

**Current Capabilities:**
- Workflow DSL (stage, loop, conditional, parallel)
- 6 pluggable ports
- Context threading (Phoenix-style assigns)
- Recipe system
- 4 built-in workflows

**Missing for MVP:**
- Real adapter implementations (only noop)
- Preprocessing/postprocessing stages
- Full checkpoint/resume logic

## crucible_train

**Current Capabilities:**
- Renderers for major models (Llama, Qwen, etc.)
- Training loops: Supervised, DPO, RL, Distillation
- Logging: Console, JSONL, W&B, Neptune

**Missing for MVP:**
- Backend adapters (only noop)
- Model registry integration
- Distributed training

## crucible_datasets

**Current Capabilities:**
- Loaders: MMLU, HumanEval, GSM8K, NoRobots
- Metrics: Exact match, F1, BLEU, ROUGE
- Sampling: Random, stratified, k-fold

**Missing for MVP:**
- Remote dataset sources (BigQuery, cloud)
- Active learning
- Schema validation

## crucible_model_registry

**Current Capabilities:**
- Ecto-backed metadata
- S3/HuggingFace/Local storage
- Lineage tracking with libgraph
- Query DSL

**Missing for MVP:**
- Model comparison
- Access control
- Deployment integration

## crucible_feedback

**Current Capabilities:**
- Batch ingestion + PII sanitization
- User signals (thumbs, edit, etc.)
- Quality assessment
- Drift detection
- Export (JSONL, preference pairs)

**Missing for MVP:**
- Embedding client (only noop)
- Advanced analytics
- Human-in-the-loop UI

## Integration Matrix

```
                Kitchen  Train  Datasets  Registry  Feedback  Deploy
Kitchen            X       X       X         X         X        X
Train              X       X       X         X         -        -
Datasets           X       X       X         -         X        -
Registry           X       X       -         X         -        X
Feedback           X       X       X         -         X        -
Deploy             X       -       -         X         -        X
```

## MVP Roadmap

### Week 1: Core Connection
- Wire crucible_train stages into Kitchen workflows
- Verify describe/1 contract compliance

### Week 2: Data & Training
- Integrate CrucibleDatasets loader
- Implement Tinkex training adapter
- Test end-to-end supervised loop

### Week 3: Model Lifecycle
- Auto-register trained models
- Wire deployment triggers
- Health check feedback loop

### Week 4: Production Readiness
- Database migrations
- Error handling
- Performance profiling
