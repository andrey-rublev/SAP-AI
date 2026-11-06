"""Audit saved desktop session logs without capturing or controlling the game.

Only aggregate counters are emitted. Screenshot paths, raw OCR, account text
and exception messages are never copied into the report. Poll coverage counts
repeated observations; unique-shop coverage counts each Board once per session.

    python -m sap_ai.desktop.audit --run-dir .local/desktop/autopilot/RUN
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Iterable

from sap_ai.desktop.state import Action, Board, Phase


HUD_FIELDS = ("gold", "turn", "wins", "lives")
MENU_ACTIONS = frozenset({"open_play", "open_arena"})


def _coverage(boards: Iterable[Board]) -> dict:
    hud, known = Counter(), Counter()
    occupied = species = levels = stats = shops = 0
    for board in boards:
        shops += 1
        for name in HUD_FIELDS:
            hud[name] += getattr(board, name) is None
        for pet in (*board.team, *board.shop):
            if pet.occupied is True:
                occupied += 1
                species += pet.species is not None
                levels += pet.level is not None
                stats += pet.attack is not None and pet.health is not None
            if pet.occupied is None:
                known["unknown_occupancy"] += 1
    return {"shop_boards": shops, "missing_hud": dict(hud),
            "unknown_occupancy": known["unknown_occupancy"], "occupied_pets": occupied,
            "known_species": species, "known_levels": levels, "known_stats": stats}


def audit_events(events: Iterable[dict]) -> dict:
    """Audit one session, including an unfinished one, from typed log events.

    Invalid JSON and Board/Action schemas are rejected. Timeline contradictions
    are reported as integrity errors so a failed run stays useful for diagnosis.
    This audits saved evidence; it does not establish recognition accuracy.
    """
    actions, receipts, phases, missing = Counter(), Counter(), Counter(), Counter()
    unique_shops, end_gold = {}, Counter()
    integrity = Counter()
    board = pending = None
    polls = started = finished = 0
    last_reason = "incomplete"
    known_trophies = None
    acknowledged_poll = None
    for event in events:
        if not isinstance(event, dict):
            raise ValueError("each log event must be an object")
        kind = event.get("event")
        if finished:
            integrity["events_after_finished"] += 1
        if kind == "started":
            started += 1
            if started != 1 or polls:
                raise ValueError("audit one session at a time; repeated started event")
            continue
        if not started:
            raise ValueError("session must begin with started")
        if kind == "observed":
            polls += 1
            if type(event.get("poll")) is not int or event["poll"] != polls:
                integrity["nonconsecutive_observation_poll"] += 1
            board = Board.from_dict(event["board"])
            phases[board.phase.value] += 1
            if board.phase == Phase.SHOP:
                unique_shops.setdefault(board.fingerprint(), board)
                for name in HUD_FIELDS:
                    missing[name] += getattr(board, name) is None
                if board.wins is not None:
                    known_trophies = board.wins
        elif kind in {"proposed", "deferred", "acted", "acknowledged"}:
            action = Action.from_dict(event["action"])
            if board is None or type(event.get("poll")) is not int or event["poll"] != polls:
                integrity["action_without_current_observation"] += 1
            if kind == "acted":
                actions[action.kind] += 1
                if pending is not None:
                    integrity["dispatch_while_pending"] += 1
                pending = (action, polls)
                if action.kind == "end_turn":
                    end_gold[str(board.gold) if board is not None and board.gold is not None else "unknown"] += 1
            elif kind == "acknowledged":
                receipts[action.kind] += 1
                if pending is None or pending[0] != action or polls <= pending[1]:
                    integrity["receipt_without_matching_dispatch"] += 1
                if acknowledged_poll == polls:
                    integrity["multiple_receipts_in_one_poll"] += 1
                acknowledged_poll = polls
                pending = None
        elif kind == "finished":
            finished += 1
            result = event["result"]
            last_reason = str(result["reason"])
            for field, actual in (("actions", sum(actions.values())),
                                  ("acknowledgments", sum(receipts.values())), ("polls", polls)):
                if type(result.get(field)) is not int or result[field] != actual:
                    integrity["finished_count_mismatch"] += 1
            recorded_pending = (Action.from_dict(result["pending_action"])
                                if result.get("pending_action") is not None else None)
            if recorded_pending != (pending[0] if pending else None):
                integrity["finished_pending_mismatch"] += 1
            if result.get("last_board") is not None and Board.from_dict(result["last_board"]) != board:
                integrity["finished_board_mismatch"] += 1
    if not started:
        raise ValueError("empty session log")
    gameplay = sum(n for kind, n in receipts.items() if kind not in MENU_ACTIONS)
    return {"complete_log": bool(finished), "reason": last_reason,
            "terminal_observed": bool(board is not None and board.phase == Phase.RESULT),
            "polls": polls, "phases": dict(phases),
            "actions": dict(actions), "acknowledgments": dict(receipts),
            "gameplay_acknowledgments": gameplay, "pending_action": pending[0].to_dict() if pending else None,
            "last_read_trophies": known_trophies,
            "shop_polls": {"total": phases["shop"], "missing_hud": dict(missing)},
            "unique_shop_coverage": _coverage(unique_shops.values()),
            "gold_at_end_turn": dict(end_gold), "integrity_errors": dict(integrity)}


def read_events(path):
    """Stream JSONL; a truncated/malformed line is explicit, never skipped."""
    with Path(path).open(encoding="utf-8") as source:
        for number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {number}") from exc


def audit_paths(paths: Iterable[Path]) -> dict:
    sessions = [audit_events(read_events(path)) for path in paths]
    if not sessions:
        raise ValueError("no session logs found")
    stops, actions, receipts, missing, integrity = (Counter() for _ in range(5))
    for session in sessions:
        stops[session["reason"]] += 1
        actions.update(session["actions"])
        receipts.update(session["acknowledgments"])
        missing.update(session["shop_polls"]["missing_hud"])
        integrity.update(session["integrity_errors"])
    return {"schema_version": 1, "offline": True,
            "summary": {"sessions": len(sessions), "incomplete_sessions": sum(not s["complete_log"] for s in sessions),
                        "terminal_sessions": sum(s["terminal_observed"] for s in sessions),
                        "stop_reasons": dict(stops), "actions": dict(actions), "acknowledgments": dict(receipts),
                        "missing_hud_by_shop_poll": dict(missing), "integrity_errors": dict(integrity)},
            "sessions": sessions,
            "limitations": "Recorded evidence only. Repeated polls are not independent accuracy samples. "
                           "Terminal sessions are not verified fresh arenas; missing trophies stay unknown."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--log", type=Path, help="one saved session.jsonl")
    source.add_argument("--run-dir", type=Path, help="directory containing session-*/session.jsonl")
    parser.add_argument("--output", type=Path, help="new aggregate JSON file; defaults to stdout")
    args = parser.parse_args(argv)
    try:
        paths = [args.log] if args.log is not None else sorted(args.run_dir.glob("session-*/session.jsonl"))
        if args.output is not None and args.output.exists():
            raise ValueError("output must be a new file; existing files are never overwritten")
        report = audit_paths(paths)
        encoded = json.dumps(report, indent=2, allow_nan=False) + "\n"
        if args.output is None:
            print(encoded, end="")
        else:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as target:
                target.write(encoded)
            print(json.dumps(report["summary"]))
        return report
    except (ValueError, TypeError, KeyError, OSError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
