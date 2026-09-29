"""Bounded desktop control with observed acknowledgments and injectable IO.

No screen or mouse dependencies are imported here. A session treats an action
as pending until its expected effect is observed; an unresponsive game never
causes an automatic retry of the same click.
"""
from __future__ import annotations

import time
from math import isfinite
from dataclasses import asdict, dataclass
from typing import Callable

from desktop_state import Action, Board, Phase, legal_action


PHASE_ACTION_KINDS = {
    Phase.NAMING: "choose_name",
    Phase.NAMING_READY: "confirm_name",
    Phase.ROUND_RESULT: "continue_round",
    Phase.TIER_UNLOCK: "dismiss_tier",
}


@dataclass(frozen=True)
class SessionResult:
    """Counts include attempted actions, including an action that raised."""

    actions: int
    acknowledgments: int
    polls: int
    reason: str
    last_board: Board | None = None
    proposed_action: Action | None = None
    pending_action: Action | None = None
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _same_turn(before: Board, after: Board) -> bool:
    return before.turn == after.turn


def _credible_target(source, target) -> bool:
    if target.occupied is not True or target.attack is None or target.health is None:
        return False
    if target.attack < 0 or target.health <= 0:
        return False
    return not (source.species and target.species and source.species != target.species)


def _shop_purchase_observed(before, after, purchased: int) -> bool:
    """Match one removal, either leaving its gap or packing remaining offers.

    Complete survivor stats are required. Species/level must agree whenever
    both observations know them; matching stats cannot prove unknown identity.
    """
    if (len(before) != len(after) or any(pet.occupied is None for pet in (*before, *after))
            or before[purchased].occupied is not True):
        return False

    def matches(expected):
        for old, new in zip(expected, after):
            if old is None or old.occupied is False:
                if new.occupied is not False:
                    return False
                continue
            if new.occupied is not True:
                return False
            for field in ("attack", "health"):
                value = getattr(old, field)
                if value is None or value != getattr(new, field):
                    return False
            for field in ("species", "level"):
                left, right = getattr(old, field), getattr(new, field)
                if left is not None and right is not None and left != right:
                    return False
        return True

    with_gap = (*before[:purchased], None, *before[purchased + 1:])
    compacted = [pet for index, pet in enumerate(before)
                 if index != purchased and pet.occupied is True]
    compacted.extend([None] * (len(before) - len(compacted)))
    return matches(with_gap) or matches(compacted)


def action_acknowledged(before: Board, after: Board, action: Action,
                        *, previous_shop_turn: int | None = None) -> bool:
    """Recognize only action-specific evidence, never arbitrary screen changes."""
    if action.kind == "end_turn":
        return before.phase == Phase.SHOP and (after.phase in (
            Phase.NAMING, Phase.BATTLE, Phase.ROUND_RESULT,
        ) or (
            after.phase == Phase.SHOP and before.turn is not None and after.turn is not None
            and after.turn > before.turn
        ))
    if action.kind == "choose_name":
        return before.phase == Phase.NAMING and after.phase == Phase.NAMING_READY
    if action.kind == "confirm_name":
        return before.phase == Phase.NAMING_READY and after.phase in (Phase.BATTLE, Phase.ROUND_RESULT)
    if action.kind == "continue_round":
        if before.phase != Phase.ROUND_RESULT:
            return False
        if after.phase == Phase.TIER_UNLOCK:
            return True
        prior = before.turn if before.turn is not None else previous_shop_turn
        return (after.phase == Phase.SHOP and after.turn is not None
                and (prior is None or after.turn > prior))
    if action.kind == "dismiss_tier":
        if before.phase != Phase.TIER_UNLOCK or after.phase != Phase.SHOP or after.turn is None:
            return False
        if before.turn is not None:
            return after.turn == before.turn
        return previous_shop_turn is None or after.turn > previous_shop_turn
    if after.phase != Phase.SHOP or not _same_turn(before, after):
        return False
    if before.gold is None or after.gold is None:
        return False
    delta = after.gold - before.gold
    if action.kind == "roll":
        # A legitimate roll may randomly produce exactly the same shop.
        return delta == -1
    if action.kind == "sell":
        i = action.slot
        return (i is not None and i < len(before.team) and i < len(after.team)
                and before.team[i].occupied is True
                and after.team[i].occupied is False and delta == 1)
    if action.kind not in ("buy", "merge"):
        return False
    i, j = action.slot, action.target
    if (i is None or j is None or i >= len(before.shop) or i >= len(after.shop)
            or j >= len(before.team) or j >= len(after.team)):
        return False
    source, target_before, target_after = before.shop[i], before.team[j], after.team[j]
    if (delta != -3 or not _shop_purchase_observed(before.shop, after.shop, i)
            or not _credible_target(source, target_after)):
        return False
    if action.kind == "buy":
        return target_before.occupied is False
    if target_before.occupied is not True:
        return False
    return any(
        old is not None and new is not None and new > old
        for old, new in ((target_before.attack, target_after.attack),
                         (target_before.health, target_after.health),
                         (target_before.level, target_after.level))
    )


class DesktopSession:
    """Run one bounded session; every dependency is replaceable for replay tests.

    The policy provides ``choose_action(board)``. ``observe`` and ``act`` are
    zero-argument capture and single-argument execution callbacks respectively.
    Event callbacks receive JSON-serializable dictionaries. ``preview`` stops
    after the first stable, legal proposal without calling ``act``.
    ``phase_actions`` explicitly enables calibrated interstitial actions;
    unconfigured dialogs stop the session. A round transition has its own poll
    budget and never infers a new turn from a dismissed overlay.
    """

    def __init__(self, observe: Callable[[], Board], act: Callable[[Action], None], policy,
                 *, event_callback: Callable[[dict], None] | None = None,
                 preview=False, max_actions=100, max_polls=1000, poll_interval=0.2,
                 stable_frames=2, action_timeout=10.0, action_max_polls=50,
                 max_unknown_polls=25, clock=time.monotonic, sleep=time.sleep,
                 should_stop: Callable[[], bool] | None = None,
                 phase_actions=None, max_transition_polls=300):
        for name, value in (("max_actions", max_actions), ("max_polls", max_polls),
                            ("stable_frames", stable_frames),
                            ("action_max_polls", action_max_polls),
                            ("max_unknown_polls", max_unknown_polls),
                            ("max_transition_polls", max_transition_polls)):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if stable_frames < 2:
            raise ValueError("stable_frames must be at least 2")
        if (not isfinite(poll_interval) or not isfinite(action_timeout)
                or poll_interval < 0 or action_timeout <= 0):
            raise ValueError("poll_interval must be finite/nonnegative and action_timeout finite/positive")
        self.observe, self.act, self.policy = observe, act, policy
        self.event_callback = event_callback
        self.preview = preview
        self.max_actions, self.max_polls = max_actions, max_polls
        self.poll_interval, self.stable_frames = poll_interval, stable_frames
        self.action_timeout, self.action_max_polls = action_timeout, action_max_polls
        self.max_unknown_polls = max_unknown_polls
        self.clock, self.sleep, self.should_stop = clock, sleep, should_stop
        self.phase_actions = {Phase(phase): action for phase, action in (phase_actions or {}).items()}
        for phase, action in self.phase_actions.items():
            if (phase not in PHASE_ACTION_KINDS or not isinstance(action, Action)
                    or action.kind != PHASE_ACTION_KINDS[phase]
                    or action.slot is not None or action.target is not None):
                raise ValueError("phase_actions must map supported phases to their matching action")
        self.max_transition_polls = max_transition_polls

    def run(self) -> SessionResult:
        actions = acknowledgments = polls = stable = unknown = pending_polls = 0
        board = proposal = pending = before = last_fingerprint = last_acted = None
        waiting_turn = last_shop_turn = None
        transition_active, transition_polls = False, 0
        completed_transitions = set()
        started = pending_at = self.clock()
        reason, error, stage = "max_polls", None, "event"
        event_failed = False

        def emit(event, **fields):
            nonlocal event_failed
            if self.event_callback is not None:
                try:
                    self.event_callback({"event": event, "poll": polls,
                                         "elapsed": max(0.0, self.clock() - started), **fields})
                except Exception:
                    event_failed = True
                    raise

        try:
            emit("started", preview=self.preview)
            while polls < self.max_polls:
                stage = "stop_check"
                if self.should_stop is not None and self.should_stop():
                    reason = "stopped"
                    break
                if polls:
                    self.sleep(self.poll_interval)
                    if self.should_stop is not None and self.should_stop():
                        reason = "stopped"
                        break
                stage = "observation"
                board = self.observe()
                if not isinstance(board, Board):
                    raise TypeError("observe must return Board")
                polls += 1
                fingerprint = board.fingerprint()
                stable = stable + 1 if fingerprint == last_fingerprint else 1
                last_fingerprint = fingerprint
                emit("observed", board=asdict(board), stable_frames=stable)
                unknown = unknown + 1 if board.phase == Phase.UNKNOWN else 0
                if board.phase == Phase.RESULT:
                    reason = "result"
                    break
                if unknown >= self.max_unknown_polls:
                    reason = "unknown_timeout"
                    break
                if board.phase in PHASE_ACTION_KINDS or board.phase == Phase.BATTLE:
                    if not transition_active:
                        waiting_turn = last_shop_turn
                    transition_active = True
                later_shop = (board.phase == Phase.SHOP and stable >= self.stable_frames
                              and board.turn is not None
                              and (waiting_turn is None or board.turn > waiting_turn))
                if transition_active and not later_shop:
                    transition_polls += 1
                    if transition_polls >= self.max_transition_polls:
                        reason = "transition_timeout"
                        break
                if pending is not None:
                    pending_polls += 1
                    # Battles animate constantly; their phase alone confirms End Turn.
                    confirmed = action_acknowledged(before, board, pending,
                                                     previous_shop_turn=last_shop_turn)
                    if confirmed and (stable >= self.stable_frames or board.phase == Phase.BATTLE):
                        acknowledgments += 1
                        emit("acknowledged", action=asdict(pending), board=asdict(board))
                        pending = None
                        continue
                    if (pending_polls >= self.action_max_polls
                            or self.clock() - pending_at >= self.action_timeout):
                        reason = "action_timeout"
                        break
                    continue
                if actions >= self.max_actions:
                    reason = "max_actions"
                    break
                if stable < self.stable_frames:
                    continue
                if board.phase == Phase.SHOP:
                    if transition_active and not later_shop:
                        continue
                    waiting_turn = None
                    transition_active, transition_polls = False, 0
                    completed_transitions.clear()
                    last_shop_turn = board.turn
                elif board.phase in PHASE_ACTION_KINDS:
                    if board.phase not in self.phase_actions:
                        reason = "transition_disabled"
                        break
                    if board.phase in completed_transitions:
                        reason = "repeated_transition"
                        break
                else:
                    continue
                if fingerprint == last_acted:
                    continue
                stage = "policy"
                proposal = (self.policy.choose_action(board) if board.phase == Phase.SHOP
                            else self.phase_actions[board.phase])
                if proposal is None:
                    reason = "policy_stopped"
                    break
                if not legal_action(board, proposal):
                    reason = "illegal_action"
                    break
                if proposal.kind == "end_turn" and board.turn is None:
                    reason = "turn_unreadable"
                    break
                emit("proposed", action=asdict(proposal), board=asdict(board))
                if self.preview:
                    reason = "preview"
                    break
                stage = "stop_check"
                if self.should_stop is not None and self.should_stop():
                    reason = "stopped"
                    break
                stage = "action"
                # Record the attempt before IO: a failing callback may have clicked.
                actions += 1
                pending, before = proposal, board
                pending_polls = 0
                last_acted = fingerprint
                if pending.kind == "end_turn":
                    waiting_turn = board.turn
                    transition_active = True
                elif board.phase in PHASE_ACTION_KINDS:
                    completed_transitions.add(board.phase)
                self.act(pending)
                # The runtime may recheck the entire board before clicking.
                # Give the resulting change its full observation budget.
                pending_at = self.clock()
                emit("acted", action=asdict(pending))
                stable = 0
            if reason == "max_polls" and waiting_turn is not None and pending is None:
                reason = "next_shop_timeout"
        except KeyboardInterrupt:
            reason = "stopped"
        except Exception as exc:
            reason = "event_error" if event_failed else f"{stage}_error"
            error = f"{type(exc).__name__}: {exc}"
        result = SessionResult(actions, acknowledgments, polls, reason, board, proposal, pending, error)
        if not event_failed:
            try:
                emit("finished", result=result.to_dict())
            except Exception as exc:
                result = SessionResult(actions, acknowledgments, polls, "event_error", board,
                                       proposal, pending, f"{type(exc).__name__}: {exc}")
        return result
