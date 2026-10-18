"""Bounded waiting for incomplete observations, using no desktop IO."""
from dataclasses import replace
from itertools import chain, repeat

import pytest

from desktop_session import DesktopSession
from desktop_state import Action, Board, DesktopPolicy, FoodSlot, PetSlot, Phase


EMPTY = PetSlot(False)
FISH = PetSlot(True, "fish", 2, 3, 1)


def shop(**changes):
    return replace(Board(Phase.SHOP, gold=3, turn=1, shop=(FISH,),
                         team=(EMPTY,) * 5), **changes)


def replay(frames, *, policy=None, **options):
    stream = iter(chain(frames, repeat(frames[-1])))
    seen, actions, events = [], [], []

    def observe():
        board = next(stream)
        seen.append(board)
        return board

    options.setdefault("max_polls", 30)
    options.setdefault("clock", lambda: 0.)
    options.setdefault("sleep", lambda _: None)
    runner = DesktopSession(observe, actions.append, policy or DesktopPolicy(),
                            event_callback=events.append, **options)
    return runner, seen, actions, events


def recovery_options():
    return dict(shop_ready=DesktopPolicy().observation_complete,
                max_incomplete_shop_polls=5)


@pytest.mark.parametrize("board", [
    shop(gold=None), shop(shop=()), shop(team=(EMPTY,) * 4),
    shop(shop=(PetSlot(None),)), shop(team=(PetSlot(None),) + (EMPTY,) * 4),
    shop(shop=(replace(FISH, attack=None),)),
    shop(team=(replace(FISH, health=None),) + (EMPTY,) * 4),
])
def test_baseline_exposes_existing_required_readings_without_inventing_them(board):
    policy = DesktopPolicy()
    assert policy.observation_complete(board) is False
    assert policy.choose_action(board) is None


def test_baseline_readiness_does_not_require_unused_identity_food_or_turn():
    board = shop(turn=None, shop=(replace(FISH, species=None, level=None),),
                 food=(FoodSlot(None),))
    policy = DesktopPolicy()
    assert policy.observation_complete(board) is True
    assert policy.choose_action(board) == Action("buy", 0, 0)
    assert policy.observation_complete(Board(Phase.BATTLE)) is False
    assert policy.observation_complete(None) is False


def test_incomplete_shop_recovers_only_after_two_matching_complete_observations():
    incomplete, ready = shop(gold=None), shop()
    runner, seen, actions, events = replay(
        [incomplete, incomplete, ready, ready], preview=True, **recovery_options())
    result = runner.run()
    assert (result.reason, result.polls, result.actions) == ("preview", 4, 0)
    assert result.proposed_action == Action("buy", 0, 0)
    assert not actions and seen[:2] == [incomplete, incomplete]
    assert seen[0].gold is None
    assert [row["poll"] for row in events if row["event"] == "proposed"] == [4]
    assert events[0] == {"event": "started", "poll": 0, "elapsed": 0.,
                         "preview": True, "max_incomplete_shop_polls": 5}
    assert {row["event"] for row in events} == {"started", "observed", "proposed", "finished"}


def test_default_direct_session_preserves_immediate_policy_stop():
    runner, _, actions, events = replay([shop(gold=None)])
    result = runner.run()
    assert (result.reason, result.polls, result.actions) == ("policy_stopped", 2, 0)
    assert not actions and "max_incomplete_shop_polls" not in events[0]


def test_zero_budget_disables_callback_without_changing_direct_session():
    def unexpected(_):
        raise AssertionError("disabled readiness callback was called")

    runner, _, _, _ = replay([shop(gold=None)], shop_ready=unexpected)
    assert runner.run().reason == "policy_stopped"


def test_changing_incomplete_fingerprints_cannot_renew_wait_budget():
    frames = [shop(gold=None, turn=turn) for turn in (1, 2, 3, 4)]
    runner, _, actions, events = replay(frames, shop_ready=DesktopPolicy().observation_complete,
                                      max_incomplete_shop_polls=3)
    result = runner.run()
    assert (result.reason, result.polls, result.actions) == ("shop_observation_timeout", 3, 0)
    assert not actions and result.pending_action is None
    assert all(row["stable_frames"] == 1 for row in events if row["event"] == "observed")
    assert not any(row["event"] == "proposed" for row in events)


def test_persistent_incomplete_shop_stops_at_explicit_bound_without_policy_calls():
    class UnexpectedPolicy:
        def choose_action(self, board):
            raise AssertionError("incomplete shop reached the policy")

    runner, _, actions, _ = replay([shop(gold=None)], policy=UnexpectedPolicy(),
                                   **recovery_options())
    result = runner.run()
    assert (result.reason, result.polls, result.actions) == ("shop_observation_timeout", 5, 0)
    assert not actions


def test_ready_custom_policy_none_is_a_real_stop_without_retry():
    class StoppingPolicy:
        calls = 0

        def choose_action(self, board):
            self.calls += 1
            return None

    policy = StoppingPolicy()
    runner, _, actions, _ = replay([shop()], policy=policy, **recovery_options())
    result = runner.run()
    assert (result.reason, result.polls, policy.calls) == ("policy_stopped", 2, 1)
    assert not actions


def test_pending_action_uses_original_timeout_without_incomplete_wait_or_reclick():
    before = shop()
    callback_boards = []

    def ready(board):
        callback_boards.append(board)
        return DesktopPolicy().observation_complete(board)

    runner, _, actions, _ = replay([before, before, shop(gold=None)],
        shop_ready=ready, max_incomplete_shop_polls=1, action_max_polls=3)
    result = runner.run()
    assert actions == [Action("buy", 0, 0)]
    assert (result.reason, result.polls, result.actions, result.acknowledgments) == (
        "action_timeout", 5, 1, 0)
    assert result.pending_action == actions[0]
    assert callback_boards == [before, before]


def test_pending_action_can_acknowledge_after_incomplete_frames_without_wait_budget():
    before = shop()
    bought = replace(before, gold=0, shop=(EMPTY,), team=(FISH,) + (EMPTY,) * 4)
    runner, _, actions, _ = replay(
        [before, before, shop(gold=None), shop(gold=None), bought, bought, bought],
        max_actions=1, shop_ready=DesktopPolicy().observation_complete,
        max_incomplete_shop_polls=1)
    result = runner.run()
    assert actions == [Action("buy", 0, 0)]
    assert (result.reason, result.polls, result.actions, result.acknowledgments) == (
        "max_actions", 7, 1, 1)
    assert result.pending_action is None


def test_nonshop_phase_resets_incomplete_episode_without_being_classified():
    incomplete, ready = shop(gold=None), shop()
    frames = [incomplete, incomplete, Board(Phase.UNKNOWN), Board(Phase.UNKNOWN),
              incomplete, incomplete, ready, ready]
    callback_phases = []

    def shop_ready(board):
        callback_phases.append(board.phase)
        return DesktopPolicy().observation_complete(board)

    runner, _, actions, _ = replay(frames, preview=True, shop_ready=shop_ready,
                                   max_incomplete_shop_polls=3)
    result = runner.run()
    assert (result.reason, result.polls) == ("preview", 8)
    assert callback_phases == [Phase.SHOP] * 6 and not actions


def test_complete_observation_resets_episode_but_still_needs_fresh_stability():
    incomplete, ready = shop(gold=None), shop()
    runner, _, actions, _ = replay(
        [incomplete, incomplete, ready, incomplete, incomplete, ready, ready],
        preview=True, shop_ready=DesktopPolicy().observation_complete,
        max_incomplete_shop_polls=3)
    result = runner.run()
    assert (result.reason, result.polls) == ("preview", 7)
    assert not actions


def test_readiness_cannot_inherit_stability_from_previously_incomplete_snapshots():
    answers = iter((False, False, True, True))
    runner, _, actions, _ = replay([shop()] * 4, preview=True,
        shop_ready=lambda _: next(answers), max_incomplete_shop_polls=3)
    result = runner.run()
    assert (result.reason, result.polls) == ("preview", 4) and not actions


def test_repeated_positive_turn_shop_cannot_pause_existing_transition_budget():
    runner, _, actions, _ = replay([Board(Phase.BATTLE), shop(gold=None)],
        max_transition_polls=3, **recovery_options())
    result = runner.run()
    assert (result.reason, result.polls) == ("transition_timeout", 3) and not actions


def test_new_wait_cannot_expand_total_poll_budget_or_override_stop():
    runner, _, actions, _ = replay([shop(gold=None)], max_polls=3, **recovery_options())
    result = runner.run()
    assert (result.reason, result.polls) == ("max_polls", 3)
    assert not actions

    runner, seen, actions, _ = replay([shop(gold=None)], **recovery_options())
    runner.should_stop = lambda: len(seen) >= 2
    result = runner.run()
    assert (result.reason, result.polls, result.actions) == ("stopped", 2, 0)
    assert not actions


def test_unknown_phase_and_transition_budgets_remain_independent():
    runner, _, actions, _ = replay([Board(Phase.UNKNOWN)], max_unknown_polls=3,
                                   **recovery_options())
    result = runner.run()
    assert (result.reason, result.polls) == ("unknown_timeout", 3)
    assert not actions

    # Battle starts the existing next-shop transition budget. Recognizing an
    # incomplete shop cannot postpone that budget with the new readiness wait.
    runner, _, actions, _ = replay([Board(Phase.BATTLE), shop(turn=None, gold=None)],
                                   max_transition_polls=3, **recovery_options())
    result = runner.run()
    assert (result.reason, result.polls) == ("transition_timeout", 3)
    assert not actions


@pytest.mark.parametrize("value", [-1, True, 1.5, "3", None])
def test_invalid_wait_budget_is_rejected(value):
    with pytest.raises(ValueError, match="max_incomplete_shop_polls"):
        replay([shop()], max_incomplete_shop_polls=value)


@pytest.mark.parametrize("callback", [None, False, "ready"])
def test_enabled_wait_requires_explicit_callable(callback):
    with pytest.raises(ValueError, match="shop_ready must be callable"):
        replay([shop()], max_incomplete_shop_polls=3, shop_ready=callback)


def test_readiness_callback_exception_is_a_policy_error_without_input():
    def broken(_):
        raise RuntimeError("readiness failed")

    runner, _, actions, _ = replay([shop()], shop_ready=broken, max_incomplete_shop_polls=3)
    result = runner.run()
    assert (result.reason, result.polls, result.actions) == ("policy_error", 1, 0)
    assert result.error == "RuntimeError: readiness failed" and not actions


@pytest.mark.parametrize("value", [None, 1, "ready", []])
def test_readiness_callback_must_return_a_boolean(value):
    runner, _, actions, _ = replay([shop()], shop_ready=lambda _: value,
                                   max_incomplete_shop_polls=3)
    result = runner.run()
    assert (result.reason, result.polls) == ("policy_error", 1)
    assert result.error == "TypeError: shop_ready must return a boolean" and not actions
