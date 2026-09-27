"""Unit tests for game.SuperAutoPetsEnv."""
import numpy as np
import pytest
import gymnasium as gym
from gymnasium.utils.env_checker import check_env

from game import SuperAutoPetsEnv


@pytest.fixture
def env():
    return SuperAutoPetsEnv()


def test_reset_shape_and_defaults(env):
    obs, info = env.reset(seed=0)
    assert obs.shape == (13,)
    assert obs.dtype == np.float32
    assert env.observation_space.contains(obs)
    assert info["gold"] == env.MAX_GOLD
    assert info["lives"] == env.START_LIVES
    assert info["wins"] == 0
    assert info["turn"] == 0
    assert list(obs[10:]) == [0, env.START_LIVES, 0]


def test_reset_is_deterministic_with_seed():
    a, _ = SuperAutoPetsEnv().reset(seed=42)
    b, _ = SuperAutoPetsEnv().reset(seed=42)
    assert np.array_equal(a, b)


def test_step_returns_five_tuple(env):
    env.reset(seed=0)
    result = env.step(env.action_space.sample())
    assert len(result) == 5
    obs, reward, terminated, truncated, info = result
    assert env.observation_space.contains(obs)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)


def test_buy_places_pet_and_spends_gold(env):
    env.reset(seed=0)
    env.shop_pets[0] = 5
    env.team_pets[:] = 0
    gold_before = env.gold
    _, reward, *_ = env.step(0)
    assert env.gold == gold_before - env.BUY_COST
    assert 5 in env.team_pets
    assert reward > 0


def test_buy_without_gold_is_penalised_and_noop(env):
    env.reset(seed=0)
    env.gold = 0
    env.shop_pets[0] = 5
    team_before = env.team_pets.copy()
    _, reward, *_ = env.step(0)
    assert reward < 0
    assert np.array_equal(env.team_pets, team_before)


def test_buy_combines_matching_pet(env):
    env.reset(seed=0)
    env.team_pets[:] = 0
    env.team_pets[0] = 3
    env.shop_pets[0] = 3
    env.gold = env.MAX_GOLD
    env.step(0)  # buying a 3 onto an existing 3 should level it up to 4
    assert env.team_pets[0] == 4
    assert (env.team_pets[1:] == 0).all()


def test_combining_rewards_actual_strength_delta(env):
    env.reset(seed=0)
    env.team_pets[:] = [6, 0, 0, 0, 0]
    env.shop_pets[:] = [6, 0, 0]
    _, reward, *_ = env.step(0)
    assert reward == pytest.approx(env.STRENGTH_REWARD - env.STEP_PENALTY)


def test_buy_then_sell_cycle_cannot_farm_positive_reward(env):
    env.reset(seed=0)
    env.shop_pets[0] = 6
    _, buy_reward, *_ = env.step(0)
    _, sell_reward, *_ = env.step(4)
    assert not env.team_pets.any()
    assert buy_reward + sell_reward == pytest.approx(-2 * env.STEP_PENALTY)


def test_roll_costs_gold_and_changes_shop(env):
    env.reset(seed=0)
    env.shop_pets[:] = 0  # force a visibly different shop after rolling
    gold_before = env.gold
    env.step(3)
    assert env.gold == gold_before - env.ROLL_COST
    assert env.shop_pets.sum() > 0


def test_sell_weakest_frees_slot_and_gains_gold(env):
    env.reset(seed=0)
    env.team_pets[:] = 0
    env.team_pets[0] = 2
    env.team_pets[1] = 6
    env.gold = 0
    env.step(4)
    assert env.team_pets[0] == 0  # the weakest (2) is sold
    assert env.team_pets[1] == 6
    assert env.gold == env.SELL_VALUE


def test_strong_team_wins_and_advances_turn(env):
    env.reset(seed=0)
    env.team_pets[:] = 0
    env.team_pets[0] = 50  # always beats the first-round enemy
    _, reward, terminated, _, info = env.step(5)
    assert info["wins"] == 1
    assert info["turn"] == 1
    assert reward > 0
    assert not terminated


def test_empty_team_loses_life(env):
    env.reset(seed=0)
    env.team_pets[:] = 0
    _, reward, *_ , info = env.step(5)
    assert info["lives"] == env.START_LIVES - 1
    assert reward < 0


def test_run_ends_in_defeat_after_zero_lives(env):
    env.reset(seed=0)
    env.team_pets[:] = 0
    terminated = False
    for _ in range(env.START_LIVES):
        _, _, terminated, _, info = env.step(5)
        if terminated:
            break
    assert terminated
    assert info["lives"] == 0


def test_run_ends_in_victory_at_win_target(env):
    env.reset(seed=0)
    env.team_pets[:] = 0
    env.team_pets[0] = 50
    terminated = False
    for _ in range(env.WINS_TO_WIN):
        _, _, terminated, _, info = env.step(5)
        if terminated:
            break
    assert terminated
    assert info["wins"] >= env.WINS_TO_WIN


def test_truncation_guard(env):
    env.reset(seed=0)
    env.turn = env.MAX_TURNS  # already at the cap
    _, _, terminated, truncated, _ = env.step(3)  # a non-terminal action
    assert truncated
    assert not terminated


def test_invalid_action_raises(env):
    env.reset(seed=0)
    with pytest.raises(ValueError):
        env.step(99)


def test_step_penalty_applied_to_shop_actions(env):
    env.reset(seed=0)
    env.gold = env.MAX_GOLD
    env.team_pets[:] = 0
    env.shop_pets[0] = 4
    _, reward, *_ = env.step(0)  # legal buy of a tier-4 pet into an empty slot
    assert reward == pytest.approx(0.1 * 4 - env.STEP_PENALTY)


def test_end_turn_is_not_step_penalised(env):
    env.reset(seed=0)
    env.team_pets[:] = 0
    env.team_pets[0] = 50  # guaranteed win -> reward is exactly +1, no penalty
    _, reward, *_ = env.step(5)
    assert reward == pytest.approx(1.0)


def test_action_mask_in_info_and_end_turn_always_legal(env):
    _, info = env.reset(seed=0)
    mask = info["action_mask"]
    assert mask.shape == (6,)
    assert mask.dtype == bool
    assert mask[5]  # end turn always legal


def test_action_mask_blocks_unaffordable_and_useless_actions(env):
    env.reset(seed=0)
    env.gold = 0
    env.shop_pets[:] = [5, 5, 5]
    env.team_pets[:] = 0  # empty team -> selling is illegal
    mask = env._action_mask()
    assert not mask[0] and not mask[1] and not mask[2]  # no gold -> no buys
    assert not mask[3]  # no gold -> no roll
    assert not mask[4]  # empty team -> no sell
    assert mask[5]      # end turn still legal


def test_action_mask_blocks_buy_when_team_full_without_combine(env):
    env.reset(seed=0)
    env.gold = env.MAX_GOLD
    env.team_pets[:] = [1, 2, 3, 4, 5]  # full, no value equals shop's 6
    env.shop_pets[:] = [6, 6, 6]
    mask = env._action_mask()
    assert not mask[0]  # can't place (full) and can't combine a 6
    # but a matching shop pet enables the combine buy
    env.shop_pets[0] = 3
    assert env._action_mask()[0]


def test_mask_from_obs_matches_env_mask(env):
    # mask_from_obs on the env's own observation must equal the internal mask.
    obs, info = env.reset(seed=1)
    assert np.array_equal(SuperAutoPetsEnv.mask_from_obs(obs), info["action_mask"])
    for _ in range(15):
        obs, _, term, trunc, info = env.step(env.action_space.sample())
        assert np.array_equal(SuperAutoPetsEnv.mask_from_obs(obs), info["action_mask"])
        if term or trunc:
            obs, info = env.reset()


def test_mask_from_obs_standalone():
    # gold 10, empty team, one non-empty shop slot -> only that buy, roll, end turn.
    obs = np.array([10, 0, 0, 4, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    mask = SuperAutoPetsEnv.mask_from_obs(obs)
    assert list(mask) == [False, True, False, True, False, True]


def test_gymnasium_environment_contract(env):
    check_env(env, skip_render_check=True)


def test_sell_at_full_gold_remains_in_observation_space(env):
    env.reset(seed=3)
    env.team_pets[:] = [1, 2, 3, 4, 5]
    for _ in range(env.TEAM_SLOTS):
        obs, _, terminated, truncated, _ = env.step(4)
        assert env.observation_space.contains(obs)
        assert not terminated and not truncated
    assert env.gold == env.MAX_GOLD + env.TEAM_SLOTS * env.SELL_VALUE


@pytest.mark.parametrize("action", [-1, 6, True, False, 1.2, 1.0, "1", None, [1], np.array(1)])
def test_action_validation_does_not_coerce_or_mutate(env, action):
    env.reset(seed=1)
    before = env._obs()
    with pytest.raises(ValueError, match="integer"):
        env.step(action)
    np.testing.assert_array_equal(env._obs(), before)
    assert env._episode_steps == 0


def test_numpy_integer_action_is_accepted(env):
    env.reset(seed=1)
    env.step(np.int64(3))
    assert env.gold == env.MAX_GOLD - env.ROLL_COST


def test_requires_reset_before_first_step_and_after_completion(env):
    with pytest.raises(gym.error.ResetNeeded):
        env.step(5)
    env.reset(seed=0)
    for _ in range(env.START_LIVES):
        obs, _, terminated, truncated, info = env.step(5)
    assert terminated and not truncated
    assert env.observation_space.contains(obs)
    assert not info["action_mask"].any()
    assert info["episode_reason"] == "defeat"
    assert info["episode"]["l"] == env.START_LIVES
    assert info["episode"]["r"] == -env.START_LIVES - 2
    with pytest.raises(gym.error.ResetNeeded):
        env.step(5)
    _, info = env.reset(seed=0)
    assert info["episode_steps"] == 0
    env.step(5)


def test_shop_timeout_bounds_repeated_illegal_actions():
    env = SuperAutoPetsEnv(max_shop_actions=3)
    env.reset(seed=1)
    env.gold = 0
    for step in range(3):
        obs, reward, terminated, truncated, info = env.step(0)
        assert reward < 0
        assert not terminated
        assert info["action_valid"] is False
        assert truncated == (step == 2)
        assert obs[12] == step + 1
        assert env.observation_space.contains(obs)
    assert info["episode_reason"] == "shop_action_limit"
    assert info["episode"]["l"] == 3
    assert info["action_mask"][5]  # continuation value remains available
    with pytest.raises(gym.error.ResetNeeded):
        env.step(5)


def test_end_turn_resets_shop_counter_and_reports_battle(env):
    env.reset(seed=4)
    env.step(0)
    assert env.shop_actions == 1
    fought_strength = int(env.team_pets.sum())
    obs, _, _, _, info = env.step(5)
    assert obs[12] == 0
    assert info["battle"]["turn"] == 0
    assert info["battle"]["team_strength"] == fought_strength
    assert 3 <= info["battle"]["enemy_strength"] <= 6
    assert info["battle"]["outcome"] in {"win", "loss", "draw"}
    _, _, _, _, shop_info = env.step(3)
    assert "battle" not in shop_info


def test_terminal_transition_preserves_shop_and_gold(env):
    env.reset(seed=2)
    env.lives = 1
    env.step(3)
    shop, gold = env.shop_pets.copy(), env.gold
    env.step(5)
    np.testing.assert_array_equal(env.shop_pets, shop)
    assert env.gold == gold


def test_turn_limit_transition_is_valid_and_reports_truncation(env):
    env.reset(seed=0)
    env.turn = env.MAX_TURNS - 1
    env.team_pets[:] = 20
    obs, _, terminated, truncated, info = env.step(5)
    assert not terminated and truncated
    assert info["episode_reason"] == "turn_limit"
    assert env.observation_space.contains(obs)
    assert info["action_mask"][5]
    assert info["episode"]["turns"] == env.MAX_TURNS


def test_metadata_cannot_create_team_slots_or_matching_pets(env):
    env.reset(seed=0)
    env.team_pets[:] = [2, 3, 4, 5, 6]
    env.shop_pets[:] = 1
    env.wins = 1
    # wins=1 cannot act as a matching pet, shop_actions=0 is not a free slot.
    extended_mask = env.mask_from_obs(env._obs())
    legacy_mask = env.mask_from_obs(env._obs()[:10])
    np.testing.assert_array_equal(extended_mask, legacy_mask)
    assert not extended_mask[:3].any()


@pytest.mark.parametrize("obs", [np.zeros(9), np.zeros((1, 10)), [np.nan] * 10, [-1] * 10])
def test_malformed_observations_raise_clear_error(obs):
    with pytest.raises(ValueError, match="observation"):
        SuperAutoPetsEnv.mask_from_obs(obs)


def test_seeded_trajectories_and_random_observation_bounds():
    first, second = SuperAutoPetsEnv(), SuperAutoPetsEnv()
    rng = np.random.default_rng(123)
    for seed in range(10):
        first.reset(seed=seed)
        second.reset(seed=seed)
        for _ in range(first.MAX_TURNS * first.MAX_SHOP_ACTIONS):
            action = int(rng.integers(6))
            a, ar, at, ax, ai = first.step(action)
            b, br, bt, bx, bi = second.step(action)
            np.testing.assert_array_equal(a, b)
            assert (ar, at, ax) == (br, bt, bx)
            assert first.observation_space.contains(a)
            np.testing.assert_array_equal(ai["action_mask"], bi["action_mask"])
            if at or ax:
                break
        else:
            pytest.fail("environment failed to bound an episode")
