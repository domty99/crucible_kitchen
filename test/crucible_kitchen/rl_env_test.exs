defmodule CrucibleKitchen.RLEnvTest do
  use ExUnit.Case, async: true

  alias CrucibleKitchen.RLEnv

  describe "new/2" do
    test "creates environment with defaults" do
      env = RLEnv.new(:bandit)

      assert env.type == :bandit
      assert env.discount == 1.0
      assert env.normalize_rewards == false
    end

    test "accepts options" do
      env = RLEnv.new(:episodic, max_steps: 5, discount: 0.99)

      assert env.type == :episodic
      assert env.max_steps == 5
      assert env.discount == 0.99
    end
  end

  describe "bandit/2" do
    test "creates bandit environment with reward function" do
      reward_fn = fn _prompt, completion -> String.length(completion) / 100 end
      env = RLEnv.bandit(reward_fn)

      assert env.type == :bandit
      assert env.reward_fn != nil
    end
  end

  describe "episodic/2" do
    test "creates episodic environment" do
      reward_fn = fn _prompt, _completion -> 1.0 end
      env = RLEnv.episodic(reward_fn, max_steps: 3)

      assert env.type == :episodic
      assert env.max_steps == 3
    end
  end

  describe "comparison/2" do
    test "creates comparison environment" do
      comparison_fn = fn _prompt, _chosen, _rejected -> 0.8 end
      env = RLEnv.comparison(comparison_fn)

      assert env.type == :comparison
      assert env.comparison_fn != nil
    end
  end

  describe "reset/2" do
    test "resets with string prompt" do
      env = RLEnv.new(:bandit)
      {:ok, state} = RLEnv.reset(env, "Hello")

      assert state.prompt == "Hello"
      assert state.step == 0
      assert state.done == false
      assert state.total_reward == 0.0
    end

    test "resets with message list" do
      env = RLEnv.new(:episodic)
      messages = [%{role: "user", content: "Hi"}]
      {:ok, state} = RLEnv.reset(env, messages)

      assert state.messages == messages
      assert state.step == 0
    end
  end

  describe "step/3 with bandit" do
    test "completes in single step" do
      reward_fn = fn _prompt, _completion -> 1.5 end
      env = RLEnv.bandit(reward_fn)

      {:ok, state} = RLEnv.reset(env, "test")
      {:ok, next_state, reward, done, info} = RLEnv.step(env, state, "response")

      assert done == true
      assert reward == 1.5
      assert next_state.done == true
      assert info.raw_reward == 1.5
    end

    test "clips rewards when configured" do
      reward_fn = fn _prompt, _completion -> 10.0 end
      env = RLEnv.bandit(reward_fn, reward_clip: {-1.0, 1.0})

      {:ok, state} = RLEnv.reset(env, "test")
      {:ok, _next_state, reward, _done, _info} = RLEnv.step(env, state, "response")

      assert reward == 1.0
    end
  end

  describe "step/3 with episodic" do
    test "allows multiple steps" do
      reward_fn = fn _prompt, _completion -> 1.0 end
      env = RLEnv.episodic(reward_fn, max_steps: 3)

      {:ok, state} = RLEnv.reset(env, "test")

      {:ok, state1, _reward, done1, _info} = RLEnv.step(env, state, "step 1")
      assert done1 == false

      {:ok, state2, _reward, done2, _info} = RLEnv.step(env, state1, "step 2")
      assert done2 == false

      {:ok, state3, _reward, done3, _info} = RLEnv.step(env, state2, "step 3")
      assert done3 == true
      assert state3.total_reward == 3.0
    end

    test "accumulates messages" do
      reward_fn = fn _prompt, _completion -> 0.0 end
      env = RLEnv.episodic(reward_fn, max_steps: 2)

      {:ok, state} = RLEnv.reset(env, "test")
      {:ok, state1, _r, _d, _i} = RLEnv.step(env, state, "response 1")

      assert length(state1.messages) == 1
      assert Enum.at(state1.messages, 0).content == "response 1"
    end
  end

  describe "compare/4" do
    test "returns preference score" do
      comparison_fn = fn _prompt, _chosen, _rejected -> 0.7 end
      env = RLEnv.comparison(comparison_fn)

      {:ok, score} = RLEnv.compare(env, "prompt", "chosen", "rejected")

      assert score == 0.7
    end

    test "errors on non-comparison env" do
      env = RLEnv.bandit(fn _, _ -> 1.0 end)

      result = RLEnv.compare(env, "prompt", "chosen", "rejected")

      assert result == {:error, :not_comparison_env}
    end
  end

  describe "collect_trajectories/4" do
    test "collects trajectories from policy" do
      reward_fn = fn prompt, completion ->
        if String.contains?(completion, prompt), do: 1.0, else: 0.0
      end

      env = RLEnv.bandit(reward_fn)
      prompts = ["a", "b", "c"]
      policy = fn prompt -> "response to #{prompt}" end

      trajectories = RLEnv.collect_trajectories(env, prompts, policy)

      assert length(trajectories) == 3
      assert Enum.all?(trajectories, &(&1.reward == 1.0))
    end

    test "samples multiple per prompt" do
      reward_fn = fn _p, _c -> 1.0 end
      env = RLEnv.bandit(reward_fn)

      trajectories = RLEnv.collect_trajectories(env, ["a"], fn _ -> "x" end, n_samples: 3)

      assert length(trajectories) == 3
    end
  end

  describe "collect_comparisons/4" do
    test "collects comparison pairs" do
      # Prefer longer completions
      comparison_fn = fn _prompt, chosen, rejected ->
        String.length(chosen) - String.length(rejected)
      end

      env = RLEnv.comparison(comparison_fn)

      # Alternate between short and long responses
      counter = :counters.new(1, [])

      policy = fn _prompt ->
        :counters.add(counter, 1, 1)

        if rem(:counters.get(counter, 1), 2) == 1 do
          "short"
        else
          "much longer response"
        end
      end

      pairs = RLEnv.collect_comparisons(env, ["p1"], policy, n_pairs: 1)

      assert length(pairs) == 1
      pair = Enum.at(pairs, 0)
      # Longer one should be chosen
      assert String.length(pair.chosen) >= String.length(pair.rejected)
    end
  end

  describe "composite_reward/1" do
    test "combines multiple reward functions" do
      fn1 = fn _prompt, _completion -> 1.0 end
      fn2 = fn _prompt, _completion -> 2.0 end

      combined = RLEnv.composite_reward([{0.5, fn1}, {0.5, fn2}])

      result = combined.("prompt", "completion")

      assert result == 1.5
    end
  end

  describe "exact_match_reward/0" do
    test "returns 1.0 when completion contains expected" do
      reward_fn = RLEnv.exact_match_reward()

      assert reward_fn.("42", "The answer is 42.") == 1.0
      assert reward_fn.("42", "The answer is 43.") == 0.0
    end

    test "is case insensitive" do
      reward_fn = RLEnv.exact_match_reward()

      assert reward_fn.("YES", "yes, that's correct") == 1.0
    end
  end

  describe "length_penalty_reward/3" do
    test "returns 0 for valid length" do
      reward_fn = RLEnv.length_penalty_reward(10, 100)

      assert reward_fn.("prompt", String.duplicate("a", 50)) == 0.0
    end

    test "penalizes too short" do
      reward_fn = RLEnv.length_penalty_reward(10, 100, 0.1)

      result = reward_fn.("prompt", "short")
      assert result < 0
    end

    test "penalizes too long" do
      reward_fn = RLEnv.length_penalty_reward(10, 100, 0.1)

      result = reward_fn.("prompt", String.duplicate("a", 150))
      assert result < 0
    end
  end

  describe "compute_returns/2" do
    test "computes undiscounted returns" do
      rewards = [1.0, 2.0, 3.0]

      returns = RLEnv.compute_returns(rewards, 1.0)

      assert returns == [6.0, 5.0, 3.0]
    end

    test "computes discounted returns" do
      rewards = [1.0, 1.0, 1.0]

      returns = RLEnv.compute_returns(rewards, 0.9)

      # R_2 = 1
      # R_1 = 1 + 0.9*1 = 1.9
      # R_0 = 1 + 0.9*1.9 = 2.71
      assert_in_delta Enum.at(returns, 2), 1.0, 0.01
      assert_in_delta Enum.at(returns, 1), 1.9, 0.01
      assert_in_delta Enum.at(returns, 0), 2.71, 0.01
    end
  end

  describe "normalize_rewards/1" do
    test "normalizes to zero mean and unit variance" do
      rewards = [1.0, 2.0, 3.0, 4.0, 5.0]

      normalized = RLEnv.normalize_rewards(rewards)

      mean = Enum.sum(normalized) / length(normalized)
      assert_in_delta mean, 0.0, 0.01

      variance =
        normalized
        |> Enum.map(fn r -> r * r end)
        |> Enum.sum()
        |> Kernel./(length(normalized))

      assert_in_delta variance, 1.0, 0.01
    end

    test "handles single element" do
      assert RLEnv.normalize_rewards([1.0]) == [1.0]
    end

    test "handles empty list" do
      assert RLEnv.normalize_rewards([]) == []
    end
  end

  describe "compute_gae/4" do
    test "computes generalized advantage estimation" do
      rewards = [1.0, 1.0, 1.0]
      values = [0.5, 0.5, 0.5]

      advantages = RLEnv.compute_gae(rewards, values, 0.99, 0.95)

      assert length(advantages) == 3
      # Each advantage should be positive since rewards > values
      assert Enum.all?(advantages, &(&1 > 0))
    end
  end
end
