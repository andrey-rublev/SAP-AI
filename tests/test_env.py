"""Unit tests for game.SuperAutoPetsEnv."""
import numpy as np
import pytest

from game import SuperAutoPetsEnv


@pytest.fixture
def env():
    return SuperAutoPetsEnv()


def test_reset_shape_and_defaults(env):
    obs, info = env.reset(seed=0)
    assert obs.shape == (10,)
    assert obs.dtype == np.float32
    assert env.observation_space.contains(obs)
    assert info["gold"] == env.MAX_GOLD
    assert info["lives"] == env.START_LIVES
    assert info["wins"] == 0
    assert info["turn"] == 0


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
    env.team_pets[0] = 100  # always beats the scaling enemy
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
    env.team_pets[0] = 100
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
    env.team_pets[0] = 100  # guaranteed win -> reward is exactly +1, no penalty
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
