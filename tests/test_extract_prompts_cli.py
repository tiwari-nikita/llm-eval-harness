import json
import sys

from extract_prompts import main


def _write_export(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")


def test_main_end_to_end_claude_style_export(tmp_path, monkeypatch, capsys):
    export = tmp_path / "conversations.json"
    _write_export(export, [
        {
            "name": "convo A",
            "chat_messages": [
                {"sender": "human", "text": "this is a long enough human message to keep"},
                {"sender": "assistant", "text": "a reply that should never be collected"},
                {"sender": "human", "text": "yes"},  # below min-words, should be dropped
            ],
        }
    ])
    out = tmp_path / "out.jsonl"
    monkeypatch.setattr(sys, "argv", ["extract_prompts.py", str(export), "-o", str(out)])

    main()

    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["text"] == "this is a long enough human message to keep"
    assert record["conversation"] == "convo A"
    assert "bucket" in record and "words" in record

    printed = capsys.readouterr().out
    assert "written to" in printed


def test_main_multiturn_only_filters_short_conversations(tmp_path, monkeypatch):
    export = tmp_path / "conversations.json"
    _write_export(export, [
        {"name": "short convo", "chat_messages": [
            {"sender": "human", "text": "only one real human turn here today"},
        ]},
        {"name": "long convo", "chat_messages": [
            {"sender": "human", "text": "human turn number one right here"},
            {"sender": "human", "text": "human turn number two right here"},
            {"sender": "human", "text": "human turn number three right here"},
            {"sender": "human", "text": "human turn number four right here"},
        ]},
    ])
    out = tmp_path / "out.jsonl"
    monkeypatch.setattr(sys, "argv",
                         ["extract_prompts.py", str(export), "-o", str(out), "--multiturn-only"])

    main()

    lines = out.read_text(encoding="utf-8").strip().splitlines()
    records = [json.loads(l) for l in lines]
    assert len(records) == 4
    assert all(r["conversation"] == "long convo" for r in records)


def test_main_exits_when_no_human_messages_found(tmp_path, monkeypatch):
    export = tmp_path / "conversations.json"
    _write_export(export, [{"name": "empty", "chat_messages": [{"sender": "assistant", "text": "hi"}]}])
    out = tmp_path / "out.jsonl"
    monkeypatch.setattr(sys, "argv", ["extract_prompts.py", str(export), "-o", str(out)])

    try:
        main()
        assert False, "expected SystemExit"
    except SystemExit as e:
        assert "no human-side messages" in str(e).lower()
