defmodule CrucibleKitchen.LRScheduleTest do
  use ExUnit.Case, async: true

  alias CrucibleKitchen.LRSchedule

  describe "new/2" do
    test "creates schedule with defaults" do
      schedule = LRSchedule.new(:cosine)

      assert schedule.type == :cosine
      assert schedule.initial_lr == 5.0e-4
      assert schedule.final_lr == 0.0
      assert schedule.total_steps == 1000
      assert schedule.warmup_steps == 0
    end

    test "accepts custom options" do
      schedule =
        LRSchedule.new(:linear,
          initial_lr: 1.0e-3,
          final_lr: 1.0e-5,
          total_steps: 5000,
          warmup_steps: 500
        )

      assert schedule.type == :linear
      assert schedule.initial_lr == 1.0e-3
      assert schedule.final_lr == 1.0e-5
      assert schedule.total_steps == 5000
      assert schedule.warmup_steps == 500
    end

    test "computes warmup_steps from warmup_ratio" do
      schedule =
        LRSchedule.new(:cosine,
          total_steps: 10_000,
          warmup_ratio: 0.1
        )

      assert schedule.warmup_steps == 1000
    end

    test "explicit warmup_steps takes precedence" do
      schedule =
        LRSchedule.new(:cosine,
          total_steps: 10_000,
          warmup_ratio: 0.1,
          warmup_steps: 500
        )

      assert schedule.warmup_steps == 500
    end
  end

  describe "constant schedule" do
    test "returns same LR throughout" do
      schedule = LRSchedule.constant(initial_lr: 1.0e-4, total_steps: 1000)

      assert LRSchedule.get_lr(schedule, 0) == 1.0e-4
      assert LRSchedule.get_lr(schedule, 500) == 1.0e-4
      assert LRSchedule.get_lr(schedule, 999) == 1.0e-4
    end
  end

  describe "linear schedule" do
    test "decays linearly from initial to final" do
      schedule =
        LRSchedule.linear(
          initial_lr: 1.0e-3,
          final_lr: 0.0,
          total_steps: 1000
        )

      assert_in_delta LRSchedule.get_lr(schedule, 0), 1.0e-3, 1.0e-8
      assert_in_delta LRSchedule.get_lr(schedule, 500), 0.5e-3, 1.0e-8
      assert_in_delta LRSchedule.get_lr(schedule, 1000), 0.0, 1.0e-8
    end

    test "decays to non-zero final LR" do
      schedule =
        LRSchedule.linear(
          initial_lr: 1.0e-3,
          final_lr: 1.0e-4,
          total_steps: 1000
        )

      assert_in_delta LRSchedule.get_lr(schedule, 0), 1.0e-3, 1.0e-8
      # Midpoint: 1.0e-3 + (1.0e-4 - 1.0e-3) * 0.5 = 5.5e-4
      assert_in_delta LRSchedule.get_lr(schedule, 500), 5.5e-4, 1.0e-8
      assert_in_delta LRSchedule.get_lr(schedule, 1000), 1.0e-4, 1.0e-8
    end
  end

  describe "cosine schedule" do
    test "follows cosine curve" do
      schedule =
        LRSchedule.cosine(
          initial_lr: 1.0e-3,
          final_lr: 0.0,
          total_steps: 1000
        )

      # At start: should be initial_lr
      assert_in_delta LRSchedule.get_lr(schedule, 0), 1.0e-3, 1.0e-8

      # At midpoint with num_cycles=0.5: cos(pi/2) = 0, so (1 + 0)/2 = 0.5
      assert_in_delta LRSchedule.get_lr(schedule, 500), 0.5e-3, 1.0e-8

      # At end with num_cycles=0.5: cos(pi) = -1, so (1 + -1)/2 = 0
      assert_in_delta LRSchedule.get_lr(schedule, 1000), 0.0, 1.0e-8
    end

    test "supports custom num_cycles" do
      schedule =
        LRSchedule.new(:cosine,
          initial_lr: 1.0e-3,
          final_lr: 0.0,
          total_steps: 1000,
          num_cycles: 1.0
        )

      # With 1 full cycle, at step 500 we should be back near initial
      lr_mid = LRSchedule.get_lr(schedule, 500)
      # cos(2*pi*1.0*0.5) = cos(pi) = -1, so (1+-1)/2 = 0
      assert_in_delta lr_mid, 0.0, 1.0e-8
    end
  end

  describe "exponential schedule" do
    test "decays exponentially" do
      schedule =
        LRSchedule.exponential(
          initial_lr: 1.0,
          decay_rate: 0.9,
          total_steps: 100
        )

      assert_in_delta LRSchedule.get_lr(schedule, 0), 1.0, 1.0e-8
      # 1.0 * 0.9^10
      assert_in_delta LRSchedule.get_lr(schedule, 10), :math.pow(0.9, 10), 1.0e-8
    end
  end

  describe "polynomial schedule" do
    test "with power=1 equals linear" do
      linear =
        LRSchedule.linear(
          initial_lr: 1.0e-3,
          final_lr: 1.0e-4,
          total_steps: 1000
        )

      poly =
        LRSchedule.polynomial(
          initial_lr: 1.0e-3,
          final_lr: 1.0e-4,
          total_steps: 1000,
          power: 1.0
        )

      for step <- [0, 250, 500, 750, 999] do
        assert_in_delta LRSchedule.get_lr(linear, step), LRSchedule.get_lr(poly, step), 1.0e-10
      end
    end

    test "higher power decays faster initially" do
      poly1 =
        LRSchedule.polynomial(
          initial_lr: 1.0e-3,
          final_lr: 0.0,
          total_steps: 1000,
          power: 1.0
        )

      poly2 =
        LRSchedule.polynomial(
          initial_lr: 1.0e-3,
          final_lr: 0.0,
          total_steps: 1000,
          power: 2.0
        )

      # At 25% progress, higher power decays faster so has lower LR
      # power=1: (1-0.25)^1 = 0.75, lr = 0.75e-3
      # power=2: (1-0.25)^2 = 0.5625, lr = 0.5625e-3
      assert LRSchedule.get_lr(poly2, 250) < LRSchedule.get_lr(poly1, 250)
    end
  end

  describe "warmup" do
    test "linearly increases during warmup" do
      schedule =
        LRSchedule.warmup_cosine(
          initial_lr: 1.0e-3,
          total_steps: 1000,
          warmup_steps: 100
        )

      # At step 0, LR should be 0
      assert_in_delta LRSchedule.get_lr(schedule, 0), 0.0, 1.0e-8

      # At step 50, should be halfway through warmup
      assert_in_delta LRSchedule.get_lr(schedule, 50), 0.5e-3, 1.0e-8

      # At step 100, should reach initial_lr
      assert_in_delta LRSchedule.get_lr(schedule, 100), 1.0e-3, 1.0e-8
    end

    test "transitions smoothly to decay phase" do
      schedule =
        LRSchedule.warmup_linear(
          initial_lr: 1.0e-3,
          final_lr: 0.0,
          total_steps: 1000,
          warmup_steps: 100
        )

      # Just before warmup ends
      lr_99 = LRSchedule.get_lr(schedule, 99)
      # Just at warmup end
      lr_100 = LRSchedule.get_lr(schedule, 100)
      # Just after warmup
      lr_101 = LRSchedule.get_lr(schedule, 101)

      # Should be continuous
      assert lr_100 >= lr_99
      assert lr_100 >= lr_101
    end

    test "warmup_constant maintains LR after warmup" do
      schedule =
        LRSchedule.warmup_constant(
          initial_lr: 1.0e-3,
          warmup_steps: 100,
          total_steps: 1000
        )

      assert_in_delta LRSchedule.get_lr(schedule, 50), 0.5e-3, 1.0e-8
      assert_in_delta LRSchedule.get_lr(schedule, 100), 1.0e-3, 1.0e-8
      assert_in_delta LRSchedule.get_lr(schedule, 500), 1.0e-3, 1.0e-8
      assert_in_delta LRSchedule.get_lr(schedule, 999), 1.0e-3, 1.0e-8
    end
  end

  describe "get_lr_schedule/1" do
    test "returns list of LRs for all steps" do
      schedule =
        LRSchedule.linear(
          initial_lr: 1.0e-3,
          final_lr: 0.0,
          total_steps: 10
        )

      lrs = LRSchedule.get_lr_schedule(schedule)

      assert length(lrs) == 10
      assert_in_delta List.first(lrs), 1.0e-3, 1.0e-8
      assert_in_delta List.last(lrs), 1.0e-4, 1.0e-8
    end
  end

  describe "convenience constructors" do
    test "constant/1 creates constant schedule" do
      schedule = LRSchedule.constant(initial_lr: 1.0e-4)
      assert schedule.type == :constant
    end

    test "linear/1 creates linear schedule" do
      schedule = LRSchedule.linear()
      assert schedule.type == :linear
    end

    test "cosine/1 creates cosine schedule" do
      schedule = LRSchedule.cosine()
      assert schedule.type == :cosine
    end

    test "exponential/1 creates exponential schedule" do
      schedule = LRSchedule.exponential()
      assert schedule.type == :exponential
    end

    test "polynomial/1 creates polynomial schedule" do
      schedule = LRSchedule.polynomial()
      assert schedule.type == :polynomial
    end
  end

  describe "edge cases" do
    test "handles step beyond total_steps" do
      schedule =
        LRSchedule.linear(
          initial_lr: 1.0e-3,
          final_lr: 0.0,
          total_steps: 100
        )

      # Beyond end should clamp to final
      lr = LRSchedule.get_lr(schedule, 200)
      assert_in_delta lr, 0.0, 1.0e-8
    end

    test "handles zero warmup steps" do
      schedule =
        LRSchedule.new(:cosine,
          initial_lr: 1.0e-3,
          total_steps: 1000,
          warmup_steps: 0
        )

      # Should start at initial_lr immediately
      assert_in_delta LRSchedule.get_lr(schedule, 0), 1.0e-3, 1.0e-8
    end

    test "handles single step schedule" do
      schedule = LRSchedule.constant(initial_lr: 1.0e-4, total_steps: 1)
      assert_in_delta LRSchedule.get_lr(schedule, 0), 1.0e-4, 1.0e-8
    end
  end
end
