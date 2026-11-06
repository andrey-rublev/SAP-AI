"""Saved-log diagnostics never open images, import native IO, or control SAP."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

import pytest

from sap_ai.desktop.audit import audit_events, audit_paths, main
from sap_ai.desktop.session import DesktopSession
from sap_ai.desktop.state import Action, Board, PetSlot, Phase


def recording():
    pet, empty = PetSlot(True, attack=2, health=3), PetSlot(False)
    before = Board(Phase.SHOP, gold=10, turn=1, wins=None, lives=5,
                   shop=(pet,), team=(empty,) * 5)
    after = replace(before, gold=7, shop=(empty,), team=(pet,) + (empty,) * 4)
    frames, events = iter([before, before, after, after, after]), []

    class Policy:
        def choose_action(self, board):
            return Action("buy", 0, 0)

    DesktopSession(lambda: next(frames), lambda action: None, Policy(), event_callback=events.append,
                   clock=lambda: 0., sleep=lambda seconds: None, max_actions=1, max_polls=5).run()
    return events


def save(path, events):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(e) for e in events), encoding="utf-8")
    return path


def test_real_session_evidence_counts_receipts_and_separates_unique_boards():
    events = recording()
    report = audit_events(events)
    observed = sum(e["event"] == "observed" for e in events)
    assert report["complete_log"] and report["reason"] == "max_actions"
    assert report["actions"] == report["acknowledgments"] == {"buy": 1}
    assert report["gameplay_acknowledgments"] == 1 and report["pending_action"] is None
    assert report["polls"] == observed and report["shop_polls"]["missing_hud"]["wins"] == observed
    assert report["last_read_trophies"] is None
    assert report["unique_shop_coverage"]["shop_boards"] == 2
    assert report["unique_shop_coverage"]["occupied_pets"] == 2
    assert report["unique_shop_coverage"]["known_species"] == 0
    assert report["integrity_errors"] == {}


def test_interrupted_log_keeps_pending_action_visible():
    events = recording()
    dispatch = next(i for i, e in enumerate(events) if e["event"] == "acted")
    report = audit_events(events[:dispatch + 1])
    assert not report["complete_log"] and report["reason"] == "incomplete"
    assert report["pending_action"] == Action("buy", 0, 0).to_dict()
    assert report["acknowledgments"] == {} and report["integrity_errors"] == {}


def test_reports_broken_counts_and_orphan_receipts():
    events = recording()
    events[-1]["result"]["actions"] = 999
    receipt = next(e for e in events if e["event"] == "acknowledged")
    index = next(i for i, event in enumerate(events) if event is receipt)
    events.insert(index + 1, deepcopy(receipt))
    errors = audit_events(events)["integrity_errors"]
    assert errors["finished_count_mismatch"] == 2
    assert errors["receipt_without_matching_dispatch"] == 1
    assert errors["multiple_receipts_in_one_poll"] == 1


def test_end_turn_records_unspent_gold_instead_of_inventing_spending():
    events = recording()
    action = Action("end_turn").to_dict()
    for event in events:
        if event["event"] in {"proposed", "acted", "acknowledged"}:
            event["action"] = action
    assert audit_events(events)["gold_at_end_turn"] == {"10": 1}


def test_audit_does_not_copy_private_paths_ocr_or_exception_text(tmp_path):
    events = recording()
    events[1]["frame_path"] = "PRIVATE_SCREENSHOT_DO_NOT_OPEN.png"
    events[-1].update(species_unread_titles={"PRIVATE_OCR": 3})
    events[-1]["result"]["error"] = "PRIVATE_EXCEPTION_TEXT"
    path = save(tmp_path / "session.jsonl", events)
    encoded = json.dumps(audit_paths([path]))
    assert "PRIVATE_" not in encoded and str(tmp_path) not in encoded


def test_directory_audit_includes_unfinished_sessions_and_writes_json(tmp_path, capsys):
    save(tmp_path / "run/session-0001/session.jsonl", recording())
    save(tmp_path / "run/session-0002/session.jsonl", recording()[:-1])
    output = tmp_path / "report.json"
    report = main(["--run-dir", str(tmp_path / "run"), "--output", str(output)])
    assert json.loads(output.read_text()) == report
    assert report["summary"]["sessions"] == 2 and report["summary"]["incomplete_sessions"] == 1
    assert report["summary"]["actions"] == report["summary"]["acknowledgments"] == {"buy": 2}


def test_output_never_overwrites_source_or_previous_report(tmp_path):
    source = save(tmp_path / "session.jsonl", recording())
    old = source.read_bytes()
    with pytest.raises(SystemExit):
        main(["--log", str(source), "--output", str(source)])
    assert source.read_bytes() == old


@pytest.mark.parametrize("text", ["", '{"event":"started"}\n{"event":',
                                  '{"event":"observed","board":{}}',
                                  '{"event":"started"}\n{"event":"started"}'])
def test_malformed_or_multiple_session_logs_fail_explicitly(tmp_path, text):
    path = tmp_path / "bad.jsonl"
    path.write_text(text)
    with pytest.raises(SystemExit):
        main(["--log", str(path)])


def test_empty_run_directory_is_an_error(tmp_path):
    with pytest.raises(SystemExit):
        main(["--run-dir", str(tmp_path)])


def test_auditor_has_no_native_ocr_or_training_imports(tmp_path):
    path = save(tmp_path / "session.jsonl", recording())
    script = '''import sys
from sap_ai.desktop.audit import main
main(["--log", sys.argv[1]])
for name in ("sap_ai.desktop.runtime", "sap_ai.desktop.cli", "pyautogui", "pytesseract", "torch"):
    assert name not in sys.modules, name
'''
    result = subprocess.run([sys.executable, "-c", script, str(path)], capture_output=True, text=True, check=True)
    assert json.loads(result.stdout)["summary"]["actions"] == {"buy": 1}
