defmodule JadedGateway.MixProject do
  use Mix.Project

  def project do
    [
      app: :jaded_gateway,
      version: "0.1.0",
      elixir: "~> 1.17",
      elixirc_paths: elixirc_paths(Mix.env()),
      start_permanent: Mix.env() == :prod,
      aliases: aliases(),
      deps: deps(),
      releases: releases()
    ]
  end

  # Configuration for the OTP application.
  def application do
    [
      mod: {JadedGateway.Application, []},
      extra_applications: [:logger, :runtime_tools]
    ]
  end

  # Specifies which paths to compile per environment.
  defp elixirc_paths(:test), do: ["lib", "test/support"]
  defp elixirc_paths(_), do: ["lib"]

  # Dependencies specification
  defp deps do
    [
      # Phoenix Framework
      {:phoenix, "~> 1.7.14"},
      {:phoenix_ecto, "~> 4.6"},
      {:ecto_sql, "~> 3.12"},
      {:postgrex, "~> 0.19"},
      {:phoenix_html, "~> 4.1"},
      {:phoenix_live_reload, "~> 1.5", only: :dev},
      {:phoenix_live_view, "~> 1.0"},
      {:floki, ">= 0.30.0", only: :test},
      {:phoenix_live_dashboard, "~> 0.8.5"},
      {:esbuild, "~> 0.8", runtime: Mix.env() == :dev},
      {:tailwind, "~> 0.2", runtime: Mix.env() == :dev},
      {:swoosh, "~> 1.17"},
      {:finch, "~> 0.19"},
      {:telemetry_metrics, "~> 1.0"},
      {:telemetry_poller, "~> 1.1"},
      {:gettext, "~> 0.26"},
      {:jason, "~> 1.4"},
      {:dns_cluster, "~> 0.1.3"},
      {:bandit, "~> 1.5"},

      # HTTP Client
      {:req, "~> 0.5"},
      {:hackney, "~> 1.20"},

      # Authentication & Authorization
      {:guardian, "~> 2.3"},
      {:comeonin, "~> 5.4"},
      {:bcrypt_elixir, "~> 3.1"},

      # Rate Limiting & Throttling
      {:hammer, "~> 6.2"},
      {:hammer_backend_redis, "~> 6.1"},

      # WebSocket & Real-time
      {:phoenix_pubsub, "~> 2.1"},
      {:presence, "~> 0.2"},

      # Circuit Breaker & Resilience
      {:fuse, "~> 2.5"},
      {:retry, "~> 0.18"},

      # Caching
      {:nebulex, "~> 2.6"},
      {:shards, "~> 1.1"},
      {:decorator, "~> 1.4"},
      {:telemetry, "~> 1.3"},

      # Background Jobs
      {:oban, "~> 2.18"},

      # Metrics & Monitoring
      {:prometheus_ex, "~> 3.1"},
      {:prometheus_plugs, "~> 1.1"},
      {:prometheus_phoenix, "~> 1.3"},

      # Configuration & Secrets
      {:vault, "~> 0.1"},
      {:configparser_ex, "~> 4.0"},

      # Validation & Parsing
      {:norm, "~> 0.13"},
      {:ex_json_schema, "~> 0.10"},

      # Development & Testing
      {:credo, "~> 1.7", only: [:dev, :test], runtime: false},
      {:dialyxir, "~> 1.4", only: [:dev, :test], runtime: false},
      {:ex_doc, "~> 0.35", only: :dev, runtime: false},
      {:mix_test_watch, "~> 1.2", only: [:dev, :test], runtime: false}
    ]
  end

  # Aliases
  defp aliases do
    [
      setup: ["deps.get", "ecto.setup", "assets.setup", "assets.build"],
      "ecto.setup": ["ecto.create", "ecto.migrate", "run priv/repo/seeds.exs"],
      "ecto.reset": ["ecto.drop", "ecto.setup"],
      test: ["ecto.create --quiet", "ecto.migrate --quiet", "test"],
      "assets.setup": ["tailwind.install --if-missing", "esbuild.install --if-missing"],
      "assets.build": ["tailwind jaded_gateway", "esbuild jaded_gateway"],
      "assets.deploy": [
        "tailwind jaded_gateway --minify",
        "esbuild jaded_gateway --minify",
        "phx.digest"
      ]
    ]
  end

  # Release configuration
  defp releases do
    [
      jaded_gateway: [
        include_executables_for: [:unix],
        applications: [runtime_tools: :permanent],
        steps: [:assemble, :tar]
      ]
    ]
  end
end