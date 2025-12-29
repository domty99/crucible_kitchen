defmodule CrucibleKitchen.RLEnv do
  @moduledoc """
  Reinforcement learning environment abstractions for LLM training.

  This module provides a consistent interface for RL environments used in
  techniques like RLHF (Reinforcement Learning from Human Feedback),
  DPO (Direct Preference Optimization), and GRPO (Group Relative Policy Optimization).

  ## Environment Types

  - **Bandit** - Single-step environment (generate completion, get reward)
  - **Episodic** - Multi-turn conversation environment
  - **Comparison** - Two-completion environment for preference learning

  ## Reward Sources

  - Reward models (trained from human preferences)
  - Rule-based rewards (code execution, math verification)
  - Human feedback (online annotation)
  - Composite rewards (weighted combination)

  ## Examples

      # Create a bandit environment with a reward model
      env = RLEnv.new(:bandit,
        reward_fn: fn prompt, completion -> RewardModel.score(prompt, completion) end
      )

      # Step through the environment
      {:ok, state} = RLEnv.reset(env, prompt)
      {:ok, next_state, reward, done} = RLEnv.step(env, state, completion)

      # Generate trajectories for training
      trajectories = RLEnv.collect_trajectories(env, prompts, policy, n_samples: 4)

  ## Integration with Training

  The RLEnv abstractions can be used with various RL algorithms:

      # PPO-style training
      for batch <- training_batches do
        trajectories = RLEnv.collect_trajectories(env, batch.prompts, policy)
        loss = PPO.compute_loss(policy, trajectories)
        # update policy...
      end
  """

  @type state :: map()
  @type action :: String.t()
  @type reward :: float()
  @type done :: boolean()
  @type info :: map()

  @type step_result :: {:ok, state(), reward(), done(), info()} | {:error, term()}

  @type env_type :: :bandit | :episodic | :comparison

  @type reward_fn :: (String.t(), String.t() -> float())
  @type comparison_fn :: (String.t(), String.t(), String.t() -> float())

  @type t :: %__MODULE__{
          type: env_type(),
          reward_fn: reward_fn() | nil,
          comparison_fn: comparison_fn() | nil,
          max_steps: pos_integer() | nil,
          discount: float(),
          normalize_rewards: boolean(),
          reward_clip: {float(), float()} | nil
        }

  defstruct type: :bandit,
            reward_fn: nil,
            comparison_fn: nil,
            max_steps: nil,
            discount: 1.0,
            normalize_rewards: false,
            reward_clip: nil

  # ==========================================================================
  # Construction
  # ==========================================================================

  @doc """
  Create a new RL environment.

  ## Parameters

  - `type` - Environment type (:bandit, :episodic, :comparison)
  - `opts` - Options:
    - `:reward_fn` - Function (prompt, completion) -> reward
    - `:comparison_fn` - Function (prompt, chosen, rejected) -> preference score
    - `:max_steps` - Maximum steps per episode (for episodic)
    - `:discount` - Discount factor gamma (default: 1.0)
    - `:normalize_rewards` - Whether to normalize rewards (default: false)
    - `:reward_clip` - Tuple {min, max} to clip rewards
  """
  @spec new(env_type(), keyword()) :: t()
  def new(type, opts \\ []) do
    %__MODULE__{
      type: type,
      reward_fn: Keyword.get(opts, :reward_fn),
      comparison_fn: Keyword.get(opts, :comparison_fn),
      max_steps: Keyword.get(opts, :max_steps),
      discount: Keyword.get(opts, :discount, 1.0),
      normalize_rewards: Keyword.get(opts, :normalize_rewards, false),
      reward_clip: Keyword.get(opts, :reward_clip)
    }
  end

  @doc """
  Create a bandit environment (single-step, reward-based).
  """
  @spec bandit(reward_fn(), keyword()) :: t()
  def bandit(reward_fn, opts \\ []) do
    new(:bandit, Keyword.put(opts, :reward_fn, reward_fn))
  end

  @doc """
  Create an episodic environment (multi-turn).
  """
  @spec episodic(reward_fn(), keyword()) :: t()
  def episodic(reward_fn, opts \\ []) do
    new(:episodic, Keyword.put(opts, :reward_fn, reward_fn))
  end

  @doc """
  Create a comparison environment for preference learning.
  """
  @spec comparison(comparison_fn(), keyword()) :: t()
  def comparison(comparison_fn, opts \\ []) do
    new(:comparison, Keyword.put(opts, :comparison_fn, comparison_fn))
  end

  # ==========================================================================
  # Core Environment Interface
  # ==========================================================================

  @doc """
  Reset the environment with a new prompt.

  ## Returns

  Initial state containing the prompt and any context.
  """
  @spec reset(t(), String.t() | [map()]) :: {:ok, state()}
  def reset(%__MODULE__{} = _env, prompt) when is_binary(prompt) do
    {:ok,
     %{
       prompt: prompt,
       messages: [],
       step: 0,
       done: false,
       total_reward: 0.0
     }}
  end

  def reset(%__MODULE__{} = _env, messages) when is_list(messages) do
    {:ok,
     %{
       prompt: nil,
       messages: messages,
       step: 0,
       done: false,
       total_reward: 0.0
     }}
  end

  @doc """
  Take a step in the environment.

  ## Parameters

  - `env` - The environment
  - `state` - Current state
  - `action` - The completion/response to take

  ## Returns

  `{:ok, next_state, reward, done, info}` or `{:error, reason}`
  """
  @spec step(t(), state(), action()) :: step_result()
  def step(%__MODULE__{type: :bandit} = env, state, action) do
    reward = compute_reward(env, state.prompt, action)
    processed_reward = process_reward(env, reward)

    next_state = %{
      state
      | step: state.step + 1,
        done: true,
        total_reward: state.total_reward + processed_reward
    }

    {:ok, next_state, processed_reward, true, %{raw_reward: reward}}
  end

  def step(%__MODULE__{type: :episodic} = env, state, action) do
    reward = compute_reward(env, state.prompt, action)
    processed_reward = process_reward(env, reward)

    next_step = state.step + 1
    done = next_step >= (env.max_steps || 1)

    next_state = %{
      state
      | step: next_step,
        done: done,
        messages: state.messages ++ [%{role: "assistant", content: action}],
        total_reward: state.total_reward + processed_reward
    }

    {:ok, next_state, processed_reward, done, %{raw_reward: reward}}
  end

  def step(%__MODULE__{type: :comparison} = _env, _state, _action) do
    {:error, :use_compare_instead}
  end

  @doc """
  Compare two completions for preference learning.

  ## Parameters

  - `env` - Comparison environment
  - `prompt` - The prompt
  - `chosen` - The preferred completion
  - `rejected` - The non-preferred completion

  ## Returns

  Preference score (higher means chosen is more preferred).
  """
  @spec compare(t(), String.t(), String.t(), String.t()) :: {:ok, float()} | {:error, term()}
  def compare(%__MODULE__{type: :comparison} = env, prompt, chosen, rejected) do
    case env.comparison_fn do
      nil ->
        {:error, :no_comparison_fn}

      fun ->
        score = fun.(prompt, chosen, rejected)
        {:ok, score}
    end
  end

  def compare(%__MODULE__{}, _prompt, _chosen, _rejected) do
    {:error, :not_comparison_env}
  end

  # ==========================================================================
  # Trajectory Collection
  # ==========================================================================

  @doc """
  Collect trajectories by sampling from a policy.

  ## Parameters

  - `env` - The environment
  - `prompts` - List of prompts to sample from
  - `policy_fn` - Function (prompt) -> completion
  - `opts` - Options:
    - `:n_samples` - Number of samples per prompt (default: 1)
    - `:return_logprobs` - Include logprobs in trajectories

  ## Returns

  List of trajectory maps.
  """
  @spec collect_trajectories(t(), [String.t()], (String.t() -> String.t()), keyword()) :: [map()]
  def collect_trajectories(%__MODULE__{} = env, prompts, policy_fn, opts \\ []) do
    n_samples = Keyword.get(opts, :n_samples, 1)

    prompts
    |> Enum.flat_map(fn prompt ->
      1..n_samples
      |> Enum.map(fn _i ->
        completion = policy_fn.(prompt)
        {:ok, state} = reset(env, prompt)
        {:ok, final_state, reward, _done, info} = step(env, state, completion)

        %{
          prompt: prompt,
          completion: completion,
          reward: reward,
          raw_reward: info[:raw_reward],
          total_reward: final_state.total_reward
        }
      end)
    end)
  end

  @doc """
  Collect comparison pairs for preference learning.

  ## Parameters

  - `env` - Comparison environment
  - `prompts` - List of prompts
  - `policy_fn` - Function (prompt) -> completion
  - `opts` - Options:
    - `:n_pairs` - Number of pairs per prompt (default: 1)

  ## Returns

  List of comparison pair maps with chosen/rejected.
  """
  @spec collect_comparisons(t(), [String.t()], (String.t() -> String.t()), keyword()) :: [map()]
  def collect_comparisons(%__MODULE__{type: :comparison} = env, prompts, policy_fn, opts \\ []) do
    n_pairs = Keyword.get(opts, :n_pairs, 1)

    prompts
    |> Enum.flat_map(fn prompt ->
      Enum.map(1..n_pairs, fn _i ->
        generate_comparison_pair(env, prompt, policy_fn)
      end)
    end)
  end

  defp generate_comparison_pair(env, prompt, policy_fn) do
    completion_a = policy_fn.(prompt)
    completion_b = policy_fn.(prompt)

    {:ok, score} = compare(env, prompt, completion_a, completion_b)

    {chosen, rejected} =
      if score >= 0, do: {completion_a, completion_b}, else: {completion_b, completion_a}

    %{
      prompt: prompt,
      chosen: chosen,
      rejected: rejected,
      preference_score: abs(score)
    }
  end

  # ==========================================================================
  # Reward Utilities
  # ==========================================================================

  @doc """
  Create a composite reward function from multiple sources.

  ## Parameters

  - `reward_fns` - List of {weight, reward_fn} tuples

  ## Returns

  A combined reward function.
  """
  @spec composite_reward([{float(), reward_fn()}]) :: reward_fn()
  def composite_reward(reward_fns) do
    fn prompt, completion ->
      reward_fns
      |> Enum.map(fn {weight, fn_} -> weight * fn_.(prompt, completion) end)
      |> Enum.sum()
    end
  end

  @doc """
  Create a reward function that checks if output contains expected answer.
  """
  @spec exact_match_reward() :: reward_fn()
  def exact_match_reward do
    fn expected, completion ->
      if String.contains?(String.downcase(completion), String.downcase(expected)) do
        1.0
      else
        0.0
      end
    end
  end

  @doc """
  Create a reward function based on length penalty.

  Penalizes outputs that are too short or too long.
  """
  @spec length_penalty_reward(non_neg_integer(), non_neg_integer(), float()) :: reward_fn()
  def length_penalty_reward(min_length, max_length, penalty \\ 0.1) do
    fn _prompt, completion ->
      len = String.length(completion)

      cond do
        len < min_length -> -penalty * (min_length - len) / min_length
        len > max_length -> -penalty * (len - max_length) / max_length
        true -> 0.0
      end
    end
  end

  @doc """
  Compute returns (discounted cumulative rewards) from a trajectory.
  """
  @spec compute_returns([float()], float()) :: [float()]
  def compute_returns(rewards, discount \\ 1.0) do
    rewards
    |> Enum.reverse()
    |> Enum.reduce({[], 0.0}, fn reward, {returns, cumulative} ->
      new_cumulative = reward + discount * cumulative
      {[new_cumulative | returns], new_cumulative}
    end)
    |> elem(0)
  end

  @doc """
  Normalize rewards to zero mean and unit variance.
  """
  @spec normalize_rewards([float()]) :: [float()]
  def normalize_rewards(rewards) when length(rewards) <= 1, do: rewards

  def normalize_rewards(rewards) do
    mean = Enum.sum(rewards) / length(rewards)

    variance =
      rewards
      |> Enum.map(fn r -> (r - mean) * (r - mean) end)
      |> Enum.sum()
      |> Kernel./(length(rewards))

    std = :math.sqrt(variance + 1.0e-8)

    Enum.map(rewards, fn r -> (r - mean) / std end)
  end

  @doc """
  Compute advantages using GAE (Generalized Advantage Estimation).
  """
  @spec compute_gae([float()], [float()], float(), float()) :: [float()]
  def compute_gae(rewards, values, gamma \\ 0.99, lambda \\ 0.95) do
    # Add terminal value of 0
    values_with_terminal = values ++ [0.0]

    # Compute TD residuals
    deltas =
      Enum.zip([rewards, values, tl(values_with_terminal)])
      |> Enum.map(fn {r, v, v_next} -> r + gamma * v_next - v end)

    # Compute GAE
    deltas
    |> Enum.reverse()
    |> Enum.reduce({[], 0.0}, fn delta, {advantages, gae} ->
      new_gae = delta + gamma * lambda * gae
      {[new_gae | advantages], new_gae}
    end)
    |> elem(0)
  end

  # ==========================================================================
  # Private Helpers
  # ==========================================================================

  defp compute_reward(%__MODULE__{reward_fn: nil}, _prompt, _action), do: 0.0

  defp compute_reward(%__MODULE__{reward_fn: fun}, prompt, action) do
    fun.(prompt, action)
  end

  defp process_reward(%__MODULE__{} = env, reward) do
    reward
    |> maybe_clip(env.reward_clip)
  end

  defp maybe_clip(reward, nil), do: reward
  defp maybe_clip(reward, {min, max}), do: max(min, min(max, reward))
end
