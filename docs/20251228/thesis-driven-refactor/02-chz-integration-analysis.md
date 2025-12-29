# ChzEx Integration Analysis for CrucibleKitchen

## Executive Summary

**chz_ex is production-ready (v0.1.2) and should be the configuration backbone for crucible_kitchen's recipes.**

## 1. Current State

### chz_ex (Elixir Port)
- **Version**: 0.1.2
- **Feature Parity**: Full with Python chz v0.4.0
- **Dependencies**: Ecto-based, pure Elixir
- **Test Coverage**: Comprehensive

### Core Modules
| Module | Purpose |
|--------|---------|
| `ChzEx.Schema` | Macro-based schema definition |
| `ChzEx.Blueprint` | Lazy construction with arg maps |
| `ChzEx.Parser` | CLI argument parsing |
| `ChzEx.Factory` | Polymorphic type resolution |
| `ChzEx.Validator` | Field + schema validation |

### Key Features
- Typed configuration schemas (compile + runtime)
- CLI argument parsing (key=value, nested, wildcards, refs)
- Polymorphic factories (Standard, Subclass, Function)
- Blueprint-driven lazy construction
- Ecto changeset validation

## 2. Integration Points for crucible_kitchen

### A. Recipe Configuration Schema (MVP Priority)
```elixir
defmodule CrucibleKitchen.Recipes.SupervisedFinetuning do
  use CrucibleKitchen.Recipe

  defmodule Config do
    use ChzEx.Schema
    chz_schema do
      field :model, :string, doc: "Model name/ID"
      field :epochs, :integer, default: 1
      field :batch_size, :integer, default: 32
      field :learning_rate, :float, default: 2.0e-5
    end
  end

  def config_schema, do: Config
end
```

### B. Context Validation
```elixir
def new(config, adapters) do
  with :ok <- validate_recipe_config(config, recipe) do
    # continue
  end
end
```

### C. CLI Generation
```elixir
defmodule CrucibleKitchen.CLI do
  def run_recipe(name, argv \\ System.argv()) do
    {:ok, recipe} = resolve_recipe(name)
    {:ok, config} = ChzEx.entrypoint(recipe.config_schema(), argv)
    CrucibleKitchen.run(name, config, adapters: get_adapters())
  end
end
```

## 3. Gap Analysis: chz_ex vs chz

| Feature | Python chz | chz_ex | Status |
|---------|-----------|--------|--------|
| Immutability | Frozen dataclass | Struct | Done |
| Type checking | Runtime | Optional | Done |
| CLI parsing | argparse | ChzEx.Parser | Done |
| Polymorphism | subclass/function | Factories | Done |
| References | `@=field` | Blueprint.Reference | Done |
| Wildcards | `...activation=` | ChzEx.Wildcard | Done |
| Validation | Custom | Changesets | Done |

**Missing (Nice-to-have)**:
- Confex bridge (use manual merging)
- YAML/JSON loaders (use external libs)

## 4. MVP Integration Plan

### Week 1: Recipe Schema
- [ ] Add ChzEx.Schema to 3-5 built-in recipes
- [ ] Implement config_schema/0 callbacks
- [ ] Add validation bridge in Context.new/2

### Week 2: CLI Integration
- [ ] Build CrucibleKitchen.CLI module
- [ ] Test wildcard/reference parsing
- [ ] Document patterns

### Week 3 (Optional): Manifest Composition
- [ ] Define manifest schema
- [ ] Add loader & validator

## 5. Recommendation

**Integrate chz_ex into crucible_kitchen for MVP** with focus on:
1. Recipe schemas (type safety)
2. CLI parsing (scripting/automation)
3. Validation (pre-execution checks)

**Why**:
- Already in mix.exs dependency
- Mature and tested
- Gradual adoption per-recipe
- Backward compatible with map configs

**Effort**: ~80-100 hours
**Value**: Type-safe config, CLI generation, validation
