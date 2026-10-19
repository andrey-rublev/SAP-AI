"""Bounded desktop control with observed acknowledgments and injectable IO.

No screen or mouse dependencies are imported here. A session treats an action
as pending until its expected effect is observed; an unresponsive game never
causes an automatic retry of the same click.
"""
from __future__ import annotations

import time
from math import isfinite
from dataclasses import asdict, dataclass, field, replace
from typing import Callable

from sap_ai.desktop.state import (Action, Board, BoardChangedBeforeInput, Phase,
                           legal_action, observed_sale_income)


PHASE_ACTION_KINDS = {
    Phase.MAIN_MENU: "open_play",
    Phase.PLAY_MENU: "open_arena",
    Phase.ARENA_SETUP: "start_arena",
    Phase.NAMING: "choose_name",
    Phase.NAMING_READY: "confirm_name",
    Phase.ROUND_RESULT: "continue_round",
    Phase.TIER_UNLOCK: "dismiss_tier",
    Phase.LIFE_REWARD: "dismiss_life_reward",
    Phase.END_TURN_CONFIRM: "confirm_end_turn",
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
        result = asdict(self)
        if self.last_board is not None:
            result["last_board"] = self.last_board.to_dict()
        return result


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


def _food_purchase_observed(before, after, purchased: int) -> bool:
    """Match one known food removal, retaining exact survivor observations.

    Unknown occupancy/identity cannot establish which food disappeared. Either
    a preserved gap or left compaction is accepted, without inventing a newly
    generated offer or accepting another item's disappearance as the receipt.
    """
    if (len(before) != len(after)
            or any(item.occupied is None or (item.occupied and item.food_id is None)
                   for item in (*before, *after))):
        return False
    gap = tuple(None if index == purchased or item.occupied is False else item
                for index, item in enumerate(before))
    compacted = tuple(item for index, item in enumerate(before)
                      if index != purchased and item.occupied is True)
    compacted += (None,) * (len(before) - len(compacted))
    observed = tuple(item if item.occupied else None for item in after)
    return observed == gap or observed == compacted


def _food_acknowledged(before: Board, after: Board, action: Action) -> bool:
    """Receipt for the explicit flat-stat food protocol, not inferred balance.

    Exact target gains deliberately abstain at stat caps or on ability effects.
    No other teammate, pet offer, counter, or surviving food may change. The
    native perceptor does not currently establish a food effect contract, and
    the default policy never proposes food; this path is for explicit adapters.
    """
    if (not legal_action(before, action) or before.turn is None
            or after.phase != Phase.SHOP or before.turn != after.turn
            or after.gold != before.gold - before.food[action.slot].cost
            or before.wins != after.wins or before.lives != after.lives
            or before.shop != after.shop or len(before.team) != len(after.team)
            or not _food_purchase_observed(before.food, after.food, action.slot)):
        return False
    source, target = before.food[action.slot], before.team[action.target]
    expected = replace(target, attack=target.attack + source.attack_bonus,
                       health=target.health + source.health_bonus)
    return all(new == (expected if index == action.target else old)
               for index, (old, new) in enumerate(zip(before.team, after.team)))


def action_acknowledged(before: Board, after: Board, action: Action,
                        *, previous_shop_turn: int | None = None) -> bool:
    """Recognize only action-specific evidence, never arbitrary screen changes."""
    if action.kind in {"open_play", "open_arena", "start_arena"}:
        if not legal_action(before, action):
            return False
        if action.kind == "open_play":
            return after.phase == Phase.PLAY_MENU
        if action.kind == "open_arena":
            # Arena resumes an existing run directly instead of showing setup.
            # Require readable shop counters before handing control to policy.
            return (after.phase == Phase.ARENA_SETUP or (
                after.phase == Phase.SHOP and after.turn is not None and after.turn > 0
                and after.gold is not None
            ))
        return after.phase == Phase.SHOP and after.turn == 1
    if action.kind == "end_turn":
        return before.phase == Phase.SHOP and (after.phase in (
            Phase.NAMING, Phase.BATTLE, Phase.ROUND_RESULT, Phase.END_TURN_CONFIRM,
        ) or (
            after.phase == Phase.SHOP and before.turn is not None and after.turn is not None
            and after.turn > before.turn
        ))
    if action.kind == "confirm_end_turn":
        return before.phase == Phase.END_TURN_CONFIRM and after.phase in (
            Phase.NAMING, Phase.BATTLE, Phase.ROUND_RESULT,
        )
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
    if action.kind in {"dismiss_tier", "dismiss_life_reward"}:
        if not legal_action(before, action):
            return False
        if action.kind == "dismiss_tier" and after.phase == Phase.LIFE_REWARD:
            return True
        if after.phase != Phase.SHOP or after.turn is None:
            return False
        if before.turn is not None:
            return after.turn == before.turn
        return previous_shop_turn is None or after.turn > previous_shop_turn
    if action.kind == "buy_food":
        return _food_acknowledged(before, after, action)
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
        if i is None or i >= len(before.team) or i >= len(after.team):
            return False
        sold = before.team[i]
        income = observed_sale_income(sold)
        # An unidentified level-one Pig can return two gold. Keep planning on
        # the supported one-gold minimum; acknowledge only its observed bounded
        # receipt after the selected slot is visibly empty. This does not infer
        # species, plan a bonus, or permit a sale retry.
        receipts = (1, 2) if sold.species is None and sold.level == 1 else (income,)
        return income is not None and after.team[i].occupied is False and delta in receipts
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
        return (target_before.occupied is False
                and source.attack is not None and source.health is not None
                and target_after.attack >= source.attack and target_after.health >= source.health)
    if (target_before.occupied is not True
            or (target_before.species and target_after.species
                and target_before.species != target_after.species)):
        return False
    changes = ((target_before.attack, target_after.attack),
               (target_before.health, target_after.health),
               (target_before.level, target_after.level))
    if any(old is not None and new is not None and new < old for old, new in changes):
        return False
    return any(old is not None and new is not None and new > old for old, new in changes)


_NEXT = object()  # a session stage asks for the next observation


@dataclass
class _RunState:
    """Mutable bookkeeping for one DesktopSession.run call."""

    started: float
    pending_at: float = 0.0
    stage: str = "event"
    event_failed: bool = False
    actions: int = 0
    acknowledgments: int = 0
    polls: int = 0
    stable: int = 0
    unknown: int = 0
    pending_polls: int = 0
    incomplete_shop_polls: int = 0
    transition_polls: int = 0
    board: Board | None = None
    fingerprint: tuple | None = None
    last_fingerprint: tuple | None = None
    last_acted: tuple | None = None
    proposal: Action | None = None
    pending: Action | None = None
    before: Board | None = None
    waiting_turn: int | None = None
    last_shop_turn: int | None = None
    transition_active: bool = False
    later_shop: bool = False
    initial_unknown: bool = True
    completed_transitions: set = field(default_factory=set)


class DesktopSession:
    """Run one bounded session; every dependency is replaceable for replay tests.

    The policy provides ``choose_action(board)``. ``observe`` and ``act`` are
    zero-argument capture and single-argument execution callbacks respectively.
    Event callbacks receive JSON-serializable dictionaries. ``preview`` stops
    after the first stable, legal proposal without calling ``act``.
    ``phase_actions`` explicitly enables calibrated interstitial actions;
    unconfigured dialogs stop the session. A round transition has its own poll
    budget and never infers a new turn from a dismissed overlay.
    An optional pure ``shop_ready(board)`` predicate can defer policy decisions
    for a bounded number of incomplete shop observations. Pending actions keep
    their existing acknowledgment budget and never use this extra allowance.
    """

    def __init__(self, observe: Callable[[], Board], act: Callable[[Action], None], policy,
                 *, event_callback: Callable[[dict], None] | None = None,
                 preview=False, max_actions=100, max_polls=1000, poll_interval=0.2,
                 stable_frames=2, action_timeout=10.0, action_max_polls=50,
                 max_unknown_polls=25, clock=time.monotonic, sleep=time.sleep,
                 should_stop: Callable[[], bool] | None = None,
                 phase_actions=None, max_transition_polls=300, max_initial_unknown_polls=None,
                 shop_ready: Callable[[Board], bool] | None = None,
                 max_incomplete_shop_polls=0):
        if max_initial_unknown_polls is None:
            max_initial_unknown_polls = max_unknown_polls
        for name, value in (("max_actions", max_actions), ("max_polls", max_polls),
                            ("stable_frames", stable_frames),
                            ("action_max_polls", action_max_polls),
                            ("max_unknown_polls", max_unknown_polls),
                            ("max_initial_unknown_polls", max_initial_unknown_polls),
                            ("max_transition_polls", max_transition_polls)):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(max_incomplete_shop_polls) is not int or max_incomplete_shop_polls < 0:
            raise ValueError("max_incomplete_shop_polls must be a nonnegative integer")
        if max_incomplete_shop_polls and not callable(shop_ready):
            raise ValueError("shop_ready must be callable when incomplete shop waiting is enabled")
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
        self.max_initial_unknown_polls = max_initial_unknown_polls
        self.shop_ready, self.max_incomplete_shop_polls = shop_ready, max_incomplete_shop_polls
        self.clock, self.sleep, self.should_stop = clock, sleep, should_stop
        self.phase_actions = {Phase(phase): action for phase, action in (phase_actions or {}).items()}
        for phase, action in self.phase_actions.items():
            if (phase not in PHASE_ACTION_KINDS or not isinstance(action, Action)
                    or action.kind != PHASE_ACTION_KINDS[phase]
                    or action.slot is not None or action.target is not None):
                raise ValueError("phase_actions must map supported phases to their matching action")
        self.max_transition_polls = max_transition_polls

    def run(self) -> SessionResult:
        s = _RunState(started=self.clock())
        s.pending_at = s.started
        reason, error = "max_polls", None
        try:
            readiness_options = ({"max_incomplete_shop_polls": self.max_incomplete_shop_polls}
                                 if self.max_incomplete_shop_polls else {})
            self._emit(s, "started", preview=self.preview, **readiness_options)
            while s.polls < self.max_polls:
                outcome = self._step(s)
                if isinstance(outcome, str):
                    reason = outcome
                    break
            if reason == "max_polls" and s.waiting_turn is not None and s.pending is None:
                reason = "next_shop_timeout"
        except KeyboardInterrupt:
            reason = "stopped"
        except Exception as exc:
            reason = "event_error" if s.event_failed else f"{s.stage}_error"
            error = f"{type(exc).__name__}: {exc}"
        result = SessionResult(s.actions, s.acknowledgments, s.polls, reason, s.board, s.proposal, s.pending, error)
        if not s.event_failed:
            try:
                self._emit(s, "finished", result=result.to_dict())
            except Exception as exc:
                result = SessionResult(s.actions, s.acknowledgments, s.polls, "event_error", s.board,
                                       s.proposal, s.pending, f"{type(exc).__name__}: {exc}")
        return result

    # Each stage returns None to fall through, _NEXT to take the next
    # observation, or a stop reason string that ends the session.
    def _step(self, s: _RunState):
        for stage in (self._observe_board, self._track_transition, self._await_receipt,
                      self._check_readiness, self._gate_phase, self._decide_and_act):
            outcome = stage(s)
            if outcome is not None:
                return outcome
        return None

    def _emit(self, s: _RunState, event, **fields):
        if self.event_callback is not None:
            try:
                self.event_callback({"event": event, "poll": s.polls,
                                     "elapsed": max(0.0, self.clock() - s.started), **fields})
            except Exception:
                s.event_failed = True
                raise

    def _stopped(self) -> bool:
        return self.should_stop is not None and self.should_stop()

    def _observe_board(self, s: _RunState):
        s.stage = "stop_check"
        if self._stopped():
            return "stopped"
        if s.polls:
            self.sleep(self.poll_interval)
            if self._stopped():
                return "stopped"
        s.stage = "observation"
        board = self.observe()
        if not isinstance(board, Board):
            raise TypeError("observe must return Board")
        s.board = board
        s.polls += 1
        s.fingerprint = board.fingerprint()
        s.stable = s.stable + 1 if s.fingerprint == s.last_fingerprint else 1
        s.last_fingerprint = s.fingerprint
        self._emit(s, "observed", board=board.to_dict(), stable_frames=s.stable)
        s.unknown = s.unknown + 1 if board.phase == Phase.UNKNOWN else 0
        if board.phase != Phase.UNKNOWN:
            s.initial_unknown = False
        if board.phase == Phase.RESULT:
            return "result"
        unknown_limit = self.max_initial_unknown_polls if s.initial_unknown else self.max_unknown_polls
        if s.unknown >= unknown_limit:
            return "unknown_timeout"
        return None

    def _track_transition(self, s: _RunState):
        board = s.board
        if board.phase in PHASE_ACTION_KINDS or board.phase == Phase.BATTLE:
            if not s.transition_active:
                s.waiting_turn = s.last_shop_turn
            s.transition_active = True
        s.later_shop = (board.phase == Phase.SHOP and s.stable >= self.stable_frames
                        and board.turn is not None
                        and (s.waiting_turn is None or board.turn > s.waiting_turn))
        if s.transition_active and not s.later_shop:
            s.transition_polls += 1
            if s.transition_polls >= self.max_transition_polls:
                return "transition_timeout"
        return None

    def _await_receipt(self, s: _RunState):
        if s.pending is None:
            return None
        s.incomplete_shop_polls = 0
        s.pending_polls += 1
        # Battles animate constantly; their phase alone confirms End Turn.
        confirmed = action_acknowledged(s.before, s.board, s.pending,
                                        previous_shop_turn=s.last_shop_turn)
        if confirmed and (s.stable >= self.stable_frames or s.board.phase == Phase.BATTLE):
            s.acknowledgments += 1
            self._emit(s, "acknowledged", action=asdict(s.pending), board=s.board.to_dict())
            s.pending = None
            return _NEXT
        if (s.pending_polls >= self.action_max_polls
                or self.clock() - s.pending_at >= self.action_timeout):
            return "action_timeout"
        return _NEXT

    def _check_readiness(self, s: _RunState):
        if s.actions >= self.max_actions:
            return "max_actions"
        if self.max_incomplete_shop_polls:
            if s.board.phase == Phase.SHOP:
                s.stage = "policy"
                ready = self.shop_ready(s.board)
                if type(ready) is not bool:
                    raise TypeError("shop_ready must return a boolean")
                if not ready:
                    # Count observations, not matching fingerprints:
                    # changing unreadable fields cannot renew the budget.
                    s.incomplete_shop_polls += 1
                    if s.incomplete_shop_polls >= self.max_incomplete_shop_polls:
                        return "shop_observation_timeout"
                    # Readiness must earn fresh stable observations;
                    # incomplete repeats cannot pause transition timeouts.
                    s.stable, s.last_fingerprint = 0, None
                    return _NEXT
            s.incomplete_shop_polls = 0
        if s.stable < self.stable_frames:
            return _NEXT
        return None

    def _gate_phase(self, s: _RunState):
        board = s.board
        if board.phase == Phase.SHOP:
            if s.transition_active and not s.later_shop:
                return _NEXT
            s.waiting_turn = None
            s.transition_active, s.transition_polls = False, 0
            s.completed_transitions.clear()
            s.last_shop_turn = board.turn
        elif board.phase in PHASE_ACTION_KINDS:
            if board.phase not in self.phase_actions:
                return "transition_disabled"
            if board.phase in s.completed_transitions:
                return "repeated_transition"
        else:
            return _NEXT
        if s.fingerprint == s.last_acted:
            return _NEXT
        return None

    def _decide_and_act(self, s: _RunState):
        board = s.board
        s.stage = "policy"
        s.proposal = (self.policy.choose_action(board) if board.phase == Phase.SHOP
                      else self.phase_actions[board.phase])
        if s.proposal is None:
            return "policy_stopped"
        if not legal_action(board, s.proposal):
            return "illegal_action"
        if s.proposal.kind == "end_turn" and board.turn is None:
            return "turn_unreadable"
        self._emit(s, "proposed", action=asdict(s.proposal), board=board.to_dict())
        if self.preview:
            return "preview"
        s.stage = "stop_check"
        if self._stopped():
            return "stopped"
        s.stage = "action"
        # Record the attempt before IO: a failing callback may have clicked.
        dispatch_state = (s.last_acted, s.waiting_turn, s.transition_active,
                          s.completed_transitions.copy(), s.before)
        s.actions += 1
        s.pending, s.before = s.proposal, board
        s.pending_polls = 0
        s.last_acted = s.fingerprint
        if s.pending.kind == "end_turn":
            s.waiting_turn = board.turn
            s.transition_active = True
        elif board.phase in PHASE_ACTION_KINDS:
            s.completed_transitions.add(board.phase)
        try:
            self.act(s.pending)
        except BoardChangedBeforeInput as exc:
            # This specific executor contract guarantees no input. Replan
            # from fresh stable observations, without spending an action.
            s.actions -= 1
            s.pending = None
            (s.last_acted, s.waiting_turn, s.transition_active,
             s.completed_transitions, s.before) = dispatch_state
            s.stable, s.last_fingerprint = 0, None
            evidence = ({"preflight_board": exc.preflight_board.to_dict()}
                        if exc.preflight_board is not None else {})
            self._emit(s, "deferred", action=asdict(s.proposal), reason="board_changed_before_input",
                       **evidence)
            s.proposal = None
            return _NEXT
        # The runtime may recheck the entire board before clicking.
        # Give the resulting change its full observation budget.
        s.pending_at = self.clock()
        self._emit(s, "acted", action=asdict(s.pending))
        s.stable = 0
        return None
