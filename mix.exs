defmodule CrucibleKitchen.MixProject do
  use Mix.Project

  @version "0.1.0"
  @source_url "https://github.com/North-Shore-AI/crucible_kitchen"

  def project do
    [
      app: :crucible_kitchen,
      version: @version,
      elixir: "~> 1.15",
      elixirc_paths: elixirc_paths(Mix.env()),
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      aliases: aliases(),

      # Hex
      description: description(),
      package: package(),

      # Docs
      name: "CrucibleKitchen",
      source_url: @source_url,
      homepage_url: @source_url,
      docs: docs()
    ]
  end

  def application do
    [
      extra_applications: [:logger],
      mod: {CrucibleKitchen.Application, []}
    ]
  end

  defp elixirc_paths(:test), do: ["lib", "test/support"]
  defp elixirc_paths(_), do: ["lib"]

  # Path to North-Shore-AI repos (development mode)
  @nsai_path "../"

  defp deps do
    [
      # ==========================================================================
      # CRUCIBLE CORE - Foundational infrastructure
      # ==========================================================================
      {:crucible_ir, path: "#{@nsai_path}crucible_ir", override: true},
      {:crucible_framework, path: "#{@nsai_path}crucible_framework", override: true},
      {:crucible_bench, path: "#{@nsai_path}crucible_bench", override: true},

      # ==========================================================================
      # CRUCIBLE TRAINING - ML training infrastructure
      # ==========================================================================
      {:crucible_train, path: "#{@nsai_path}crucible_train", override: true},

      # ==========================================================================
      # CRUCIBLE MLOPS - Model lifecycle management
      # ==========================================================================
      {:crucible_model_registry, path: "#{@nsai_path}crucible_model_registry", override: true},
      {:crucible_deployment, path: "#{@nsai_path}crucible_deployment", override: true},
      {:crucible_feedback, path: "#{@nsai_path}crucible_feedback", override: true},

      # ==========================================================================
      # CRUCIBLE OBSERVABILITY - Telemetry, tracing, harness
      # ==========================================================================
      {:crucible_telemetry, path: "#{@nsai_path}crucible_telemetry", override: true},
      {:crucible_trace, path: "#{@nsai_path}crucible_trace", override: true},
      {:crucible_harness, path: "#{@nsai_path}crucible_harness", override: true},
      {:crucible_datasets, path: "#{@nsai_path}crucible_datasets", override: true},

      # ==========================================================================
      # CRUCIBLE RELIABILITY - Ensemble, hedging, adversarial
      # ==========================================================================
      {:crucible_ensemble, path: "#{@nsai_path}crucible_ensemble", override: true},
      {:crucible_hedging, path: "#{@nsai_path}crucible_hedging", override: true},
      {:crucible_adversary, path: "#{@nsai_path}crucible_adversary", override: true},
      {:crucible_xai, path: "#{@nsai_path}crucible_xai", override: true},

      # ==========================================================================
      # HUGGINGFACE INTEGRATION
      # ==========================================================================
      {:hf_hub, path: "#{@nsai_path}hf_hub_ex", override: true},
      {:hf_datasets_ex, path: "#{@nsai_path}hf_datasets_ex", override: true},

      # ==========================================================================
      # EVALUATION & CONFIG
      # ==========================================================================
      {:eval_ex, path: "#{@nsai_path}eval_ex", override: true},
      {:chz_ex, path: "#{@nsai_path}chz_ex", override: true},

      # ==========================================================================
      # TOKENIZATION
      # ==========================================================================
      {:tiktoken_ex, path: "#{@nsai_path}tiktoken_ex", override: true},

      # ==========================================================================
      # PYTHON BRIDGE (stable, from hex)
      # ==========================================================================
      {:snakebridge, "~> 0.6.0"},

      # ==========================================================================
      # TINKEX SDK (Backend adapter reference implementation)
      # ==========================================================================
      {:tinkex, path: "#{@nsai_path}tinkex", override: true},

      # ==========================================================================
      # EXTERNAL DEPENDENCIES
      # ==========================================================================
      {:telemetry, "~> 1.2"},
      {:telemetry_metrics, "~> 1.0", override: true},
      {:jason, "~> 1.4"},

      # Dev/test
      {:ex_doc, "~> 0.31", only: :dev, runtime: false},
      {:dialyxir, "~> 1.4", only: [:dev, :test], runtime: false},
      {:credo, "~> 1.7", only: [:dev, :test], runtime: false},
      {:mox, "~> 1.1", only: :test}
    ]
  end

  defp aliases do
    [
      quality: ["format --check-formatted", "credo --strict", "dialyzer"]
    ]
  end

  defp description do
    """
    Industrial ML training orchestration - backend-agnostic workflow engine
    for supervised, reinforcement, and preference learning.
    """
  end

  defp package do
    [
      name: "crucible_kitchen",
      licenses: ["MIT"],
      links: %{
        "GitHub" => @source_url,
        "Changelog" => "#{@source_url}/blob/main/CHANGELOG.md"
      },
      maintainers: ["North-Shore-AI"],
      files:
        ~w(lib assets docs/guides docs/00_EXECUTIVE_SUMMARY.md docs/01_ARCHITECTURE_PATTERNS.md docs/02_COMPONENT_DESIGN.md docs/03_WORKFLOW_ENGINE.md docs/04_API_SURFACE.md docs/05_IMPLEMENTATION_ROADMAP.md docs/06_ECOSYSTEM_INTEGRATION.md .formatter.exs mix.exs README.md LICENSE CHANGELOG.md)
    ]
  end

  defp docs do
    [
      main: "readme",
      source_ref: "v#{@version}",
      assets: %{"assets" => "assets"},
      logo: "assets/crucible_kitchen.svg",
      extras: [
        "README.md",
        "CHANGELOG.md",
        # Guides
        "docs/guides/getting_started.md",
        "docs/guides/custom_workflows.md",
        "docs/guides/adapters.md",
        "docs/guides/telemetry.md",
        # Architecture
        "docs/00_EXECUTIVE_SUMMARY.md",
        "docs/01_ARCHITECTURE_PATTERNS.md",
        "docs/02_COMPONENT_DESIGN.md",
        "docs/03_WORKFLOW_ENGINE.md",
        "docs/04_API_SURFACE.md",
        "docs/05_IMPLEMENTATION_ROADMAP.md",
        "docs/06_ECOSYSTEM_INTEGRATION.md"
      ],
      groups_for_extras: [
        Introduction: ["README.md", "CHANGELOG.md"],
        Guides: [
          "docs/guides/getting_started.md",
          "docs/guides/custom_workflows.md",
          "docs/guides/adapters.md",
          "docs/guides/telemetry.md"
        ],
        Architecture: [
          "docs/00_EXECUTIVE_SUMMARY.md",
          "docs/01_ARCHITECTURE_PATTERNS.md",
          "docs/02_COMPONENT_DESIGN.md",
          "docs/03_WORKFLOW_ENGINE.md",
          "docs/04_API_SURFACE.md",
          "docs/05_IMPLEMENTATION_ROADMAP.md",
          "docs/06_ECOSYSTEM_INTEGRATION.md"
        ]
      ],
      groups_for_modules: [
        Core: [
          CrucibleKitchen,
          CrucibleKitchen.Context,
          CrucibleKitchen.Recipe,
          CrucibleKitchen.Stage,
          CrucibleKitchen.Workflow
        ],
        Ports: [
          CrucibleKitchen.Ports,
          CrucibleKitchen.Ports.Completer
        ],
        "Built-in Workflows": [
          CrucibleKitchen.Workflows.Supervised,
          CrucibleKitchen.Workflows.Reinforcement,
          CrucibleKitchen.Workflows.Preference,
          CrucibleKitchen.Workflows.Distillation
        ],
        Telemetry: [
          CrucibleKitchen.Telemetry,
          CrucibleKitchen.Telemetry.Handlers.Console,
          CrucibleKitchen.Telemetry.Handlers.JSONL
        ]
      ]
    ]
  end
end
