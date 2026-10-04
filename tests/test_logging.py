import json
import logging

from common.logging import JsonFormatter


def test_json_formatter_includes_severity_message_and_extras() -> None:
    record = logging.LogRecord("x", logging.INFO, "f.py", 1, "hello %s", ("world",), None)
    record.gameweek = 7
    out = json.loads(JsonFormatter().format(record))

    assert out["severity"] == "INFO"
    assert out["message"] == "hello world"
    assert out["gameweek"] == 7
