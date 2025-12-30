defmodule CrucibleKitchen.Stages.BuildEnvGroup do
  @moduledoc """
  Builds environment group for RL rollout collection.

  Creates an environment group builder that produces parallel environments
  for trajectory collection. The environments implement the `CrucibleTrain.RL.Env`
  behaviour.

  ## Context Requirements

  **Input:**
  - Config: `:env` - Environment type (atom) or module
  - Config: `:group_size` - Number of environments per group (default: 4)
  - Config: `:groups_per_batch` - Number of groups per batch (default: 100)
  - State: `:raw_dataset` - Dataset for environment prompts (optional)

  **Output:**
  - State: `:env_group` - Environment group builder
  - State: `:env_config` - Environment configuration

  ## Example

      stage(:build_env_group, BuildEnvGroup)
  """

  use CrucibleKitchen.Stage

  alias CrucibleKitchen.Context

  require Logger

  @impl true
  def name, do: :build_env_group

  @impl true
  def execute(context) do
    env_type = Context.get_config(context, :env, :noop)
    group_size = Context.get_config(context, :group_size, 4)
    groups_per_batch = Context.get_config(context, :groups_per_batch, 100)
    raw_dataset = Context.get_state(context, :raw_dataset)

    Logger.info(
      "Building env group: type=#{inspect(env_type)} " <>
        "group_size=#{group_size} groups_per_batch=#{groups_per_batch}"
    )

    env_config = %{
      env_type: env_type,
      group_size: group_size,
      groups_per_batch: groups_per_batch,
      dataset: raw_dataset
    }

    env_group = build_env_group(env_type, env_config)

    emit_telemetry(env_type, group_size, groups_per_batch)

    context
    |> Context.put_state(:env_group, env_group)
    |> Context.put_state(:env_config, env_config)
    |> then(&{:ok, &1})
  end

  @impl true
  def validate(_context), do: :ok

  defp build_env_group(:noop, config) do
    # Noop environment group for testing
    %{
      type: :noop,
      config: config,
      make_envs: fn _opts ->
        Enum.map(1..config.group_size, fn i ->
          %{
            id: "noop_env_#{i}",
            observation: "Initial observation #{i}",
            done: false
          }
        end)
      end
    }
  end

  defp build_env_group(module, config) when is_atom(module) do
    # Custom environment module
    if Code.ensure_loaded?(module) and function_exported?(module, :make_envs, 1) do
      %{
        type: :custom,
        module: module,
        config: config,
        make_envs: fn opts -> module.make_envs(opts) end
      }
    else
      Logger.warning("Environment module #{module} not found, using noop")
      build_env_group(:noop, config)
    end
  end

  defp emit_telemetry(env_type, group_size, groups_per_batch) do
    :telemetry.execute(
      [:crucible_kitchen, :rl, :env_group_built],
      %{group_size: group_size, groups_per_batch: groups_per_batch},
      %{env_type: env_type}
    )
  end
end
