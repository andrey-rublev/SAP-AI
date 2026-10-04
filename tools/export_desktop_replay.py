"""Export typed observation/action evidence without desktop IO.

Only Boards, actions and relative polls enter the fixture. Screenshots, paths,
elapsed times, errors and account metadata are not copied. Replay tests must
inject observation/action callbacks; this command never imports desktop IO.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from desktop_state import Action, Board, legal_action


DESCRIPTION = (
    "Recorded Boards, actions and relative polls only; images, paths, timings and "
    "account metadata are excluded. This preserves controller evidence, not OCR "
    "accuracy, policy quality or combat performance."
)
REPLAY_REASONS = {
    "max_actions", "max_polls", "result", "unknown_timeout", "transition_timeout",
    "action_timeout", "policy_stopped", "illegal_action", "transition_disabled",
    "repeated_transition", "turn_unreadable", "preview", "stopped", "next_shop_timeout",
}


def export_session(text, *, session=-1):
    """Select one run, validate its timeline and return a sanitized replay.

    Positive session indices are one-based; negative indices count from the end.
    The newest run is selected by default, even when incomplete, so an older
    finished run cannot silently stand in for interrupted new evidence.
    """
    if type(session) is not int or session == 0:
        raise ValueError("session must be a nonzero integer")
    runs, current = [], None
    for line in text.splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError("log events must be objects")
        if row.get("event") == "started":
            current = []
            runs.append(current)
        if current is None:
            raise ValueError("each run must begin with a started event")
        current.append(row)
        if row.get("event") == "finished":
            current = None
    try:
        rows = runs[session - 1 if session > 0 else session]
    except IndexError as exc:
        raise ValueError("selected session does not exist") from exc
    if rows[-1].get("event") != "finished":
        raise ValueError("selected session is incomplete")

    boards, indices, observation_runs, actions, deferrals = [], {}, [], [], []
    poll = acknowledgments = 0
    last_board = proposal = pending = proposal_poll = last_ack_poll = last_deferral_poll = None
    for row in rows[1:-1]:
        event = row.get("event")
        if event == "observed":
            if type(row.get("poll")) is not int or row["poll"] != poll + 1:
                raise ValueError("observation polls must be consecutive from one")
            poll = row["poll"]
            last_board = Board.from_dict(row["board"])
            fingerprint = last_board.fingerprint()
            if fingerprint not in indices:
                indices[fingerprint] = len(boards)
                boards.append(last_board.to_dict())
            index = indices[fingerprint]
            if observation_runs and observation_runs[-1][0] == index:
                observation_runs[-1][1] += 1
            else:
                observation_runs.append([index, 1])
            continue
        if event not in {"proposed", "acted", "acknowledged", "deferred"}:
            raise ValueError("unsupported event in selected session")
        if type(row.get("poll")) is not int or row["poll"] != poll or last_board is None:
            raise ValueError("action events must match the latest observation poll")
        action = Action.from_dict(row["action"])
        if event == "proposed":
            if (pending is not None or proposal is not None or poll == last_ack_poll
                    or (last_deferral_poll is not None and poll < last_deferral_poll + 2)
                    or Board.from_dict(row["board"]) != last_board
                    or not legal_action(last_board, action)):
                raise ValueError("proposal contradicts the observed timeline")
            proposal, proposal_poll = action, poll
        elif event == "acted":
            if proposal != action or pending is not None or proposal_poll != poll:
                raise ValueError("dispatch needs a matching proposal and no pending action")
            actions.append({"poll": poll, "action": action.to_dict(), "acknowledged_poll": None})
            pending, proposal = action, None
        elif event == "deferred":
            if (row.get("reason") != "board_changed_before_input" or proposal != action
                    or pending is not None or proposal_poll != poll):
                raise ValueError("deferral needs its same-poll proposal, no pending action and supported reason")
            deferrals.append({"poll": poll, "action": action.to_dict()})
            last_deferral_poll = poll
            proposal = proposal_poll = None
        else:
            if (pending != action or poll <= actions[-1]["poll"]
                    or Board.from_dict(row["board"]) != last_board):
                raise ValueError("acknowledgment needs its pending action and observed board")
            actions[-1]["acknowledged_poll"] = poll
            acknowledgments += 1
            last_ack_poll = poll
            pending = None

    finished = rows[-1]
    result = finished["result"]
    if not isinstance(result, dict):
        raise ValueError("finished result must be an object")
    if result.get("error") is not None:
        raise ValueError("failed IO cannot be reproduced from saved Boards alone")
    expected = {name: result[name] for name in
                ("actions", "acknowledgments", "polls", "reason", "pending_action")}
    for field, actual in (("actions", len(actions)), ("acknowledgments", acknowledgments), ("polls", poll)):
        if type(expected[field]) is not int or expected[field] != actual:
            raise ValueError("finished counts contradict the recorded timeline")
    if type(finished.get("poll")) is not int or finished["poll"] != poll or not poll:
        raise ValueError("finished poll must match a nonempty observation timeline")
    if expected["reason"] not in REPLAY_REASONS:
        raise ValueError("finished reason must be a supported controller stop")
    recorded_pending = (Action.from_dict(expected["pending_action"])
                        if expected["pending_action"] is not None else None)
    if recorded_pending != pending or Board.from_dict(result["last_board"]) != last_board:
        raise ValueError("finished state contradicts the recorded timeline")
    expected["pending_action"] = pending.to_dict() if pending is not None else None
    replay = {"description": DESCRIPTION, "boards": boards, "observation_runs": observation_runs,
              "actions": actions, "expected": expected}
    if deferrals:
        replay["deferrals"] = deferrals
    return replay


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", required=True, help="existing JSONL log; referenced images are not opened")
    parser.add_argument("--output", required=True, help="sanitized JSON fixture to review before committing")
    parser.add_argument("--session", type=int, default=-1, help="one-based run index, or negative from the end")
    args = parser.parse_args(argv)
    try:
        output, source = Path(args.output), Path(args.log)
        if output.resolve() == source.resolve() or output.exists():
            raise ValueError("output must be a new file distinct from the source log")
        replay = export_session(source.read_text(encoding="utf-8"), session=args.session)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(replay, indent=2, allow_nan=False) + "\n")
        print(json.dumps({"actions": len(replay["actions"]), "polls": replay["expected"]["polls"]}))
    except (ValueError, TypeError, KeyError, OSError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
