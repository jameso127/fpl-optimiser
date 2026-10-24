import json
import threading
from collections.abc import Iterator, Sequence
from typing import Any

import httpx
import pytest
import requests

from bot.handlers import FULL, HELP, reply
from bot.main import Bot, Handler, Server
from bot.webhook import SECRET_HEADER, register, webhook_secret
from common.config import Settings
from common.telegram import Button
from common.users import InMemoryUserRepository, User
from tests import fakes

CHAT = 42
TOKEN = "123456:secret-token"
SIGNED = {SECRET_HEADER: webhook_secret(TOKEN)}


class FakeEntries:
    """FPL's entry endpoint: known teams answer with a name, anything else is a 404."""

    def __init__(self, teams: dict[int, str] | None = None, down: bool = False) -> None:
        self.teams = teams or {1234567: "Haaland & Co"}
        self.down = down
        self.asked: list[int] = []

    def entry(self, team_id: int) -> dict[str, Any]:
        self.asked.append(team_id)
        if self.down:
            raise RuntimeError("FPL request failed after 6 attempts")
        if team_id not in self.teams:
            raise requests.HTTPError(response=fakes.http_response(404))
        return {"id": team_id, "name": self.teams[team_id]}


class RecordingSender:
    def __init__(self) -> None:
        self.sent: list[tuple[int, str]] = []

    def send(
        self,
        chat_id: int,
        text: str,
        *,
        buttons: Sequence[Sequence[Button]] = (),
        effect_id: str | None = None,
    ) -> None:
        self.sent.append((chat_id, text))

    def send_photo(
        self,
        chat_id: int,
        png: bytes,
        caption: str,
        *,
        buttons: Sequence[Sequence[Button]] = (),
        effect_id: str | None = None,
    ) -> None:
        self.sent.append((chat_id, caption))


def _update(text: str, chat_id: int = CHAT, chat_type: str = "private") -> dict[str, Any]:
    return {"update_id": 1, "message": {"chat": {"id": chat_id, "type": chat_type}, "text": text}}


def _say(text: str, repo: InMemoryUserRepository, fpl: FakeEntries | None = None,
         max_users: int = 50, chat_id: int = CHAT) -> str:  # fmt: skip
    answer = reply(_update(text, chat_id), repo, fpl or FakeEntries(), max_users)
    assert answer is not None
    return answer


# --- conversations ----------------------------------------------------------------------------


def test_start_registers_a_new_chat_and_asks_for_a_team_id() -> None:
    repo = InMemoryUserRepository()

    answer = _say("/start", repo)

    assert "Welcome" in answer and "team id" in answer
    user = repo.get(CHAT)
    assert user is not None and user.fpl_team_id is None and user.active
    assert repo.list_active() == []  # no team yet, so the jobs ignore it


def test_sending_a_team_id_links_it_after_checking_it_exists() -> None:
    repo, fpl = InMemoryUserRepository(), FakeEntries()
    _say("/start", repo)

    answer = _say("1234567", repo, fpl)

    assert "Haaland &amp; Co" in answer  # escaped: replies are sent as HTML
    assert fpl.asked == [1234567]
    assert [(u.chat_id, u.fpl_team_id) for u in repo.list_active()] == [(CHAT, 1234567)]


def test_a_team_id_alone_also_registers_a_chat_that_never_sent_start() -> None:
    repo = InMemoryUserRepository()

    _say("/team 1234567", repo)

    user = repo.get(CHAT)
    assert user is not None and user.fpl_team_id == 1234567


def test_an_unknown_team_is_refused_and_nothing_is_linked() -> None:
    repo = InMemoryUserRepository()
    _say("/start", repo)

    answer = _say("999", repo)

    assert "couldn't find FPL team 999" in answer
    user = repo.get(CHAT)
    assert user is not None and user.fpl_team_id is None


@pytest.mark.parametrize("text", ["/team", "/team abc", "/team 0", "/team 99999999999"])
def test_a_bad_team_id_never_reaches_the_fpl_api(text: str) -> None:
    fpl = FakeEntries()

    assert "doesn't look like a team id" in _say(text, InMemoryUserRepository(), fpl)
    assert fpl.asked == []


def test_fpl_being_down_says_try_again_and_links_nothing() -> None:
    repo = InMemoryUserRepository()

    assert "try again" in _say("1234567", repo, FakeEntries(down=True))
    assert repo.get(CHAT) is None


def test_pause_and_resume_take_a_user_out_of_and_back_into_the_jobs() -> None:
    repo = InMemoryUserRepository()
    _say("1234567", repo)

    assert "Paused" in _say("/pause", repo)
    assert repo.list_active() == []
    assert "paused" in _say("/start", repo)  # the status says so
    assert "Resumed" in _say("/resume", repo)
    assert [u.chat_id for u in repo.list_active()] == [CHAT]


def test_pause_before_registering_points_at_start() -> None:
    repo = InMemoryUserRepository()

    assert "/start" in _say("/pause", repo)
    assert repo.get(CHAT) is None


def test_delete_forgets_the_user() -> None:
    repo = InMemoryUserRepository()
    _say("1234567", repo)

    assert "forgotten" in _say("/delete", repo)
    assert repo.get(CHAT) is None and repo.count() == 0


def test_start_again_shows_the_linked_team() -> None:
    repo = InMemoryUserRepository()
    _say("1234567", repo)

    assert "following team 1234567" in _say("/start@FplBot", repo)


def test_anything_else_gets_the_help() -> None:
    assert _say("hello", InMemoryUserRepository()) == HELP
    assert _say("/help", InMemoryUserRepository()) == HELP


def test_no_one_new_is_registered_once_the_bot_is_full() -> None:
    repo = InMemoryUserRepository()
    repo.save(User(chat_id=1, fpl_team_id=10))

    assert _say("/start", repo, max_users=1) == FULL
    assert _say("1234567", repo, max_users=1) == FULL
    assert repo.count() == 1
    # ...but existing users carry on as normal.
    assert "Linked" in _say("1234567", repo, max_users=1, chat_id=1)


@pytest.mark.parametrize(
    "update",
    [
        _update("/start", chat_type="group"),  # not meant for groups
        {"update_id": 1, "message": {"chat": {"id": CHAT, "type": "private"}}},  # a sticker etc.
        {"update_id": 1, "edited_message": {"chat": {"id": CHAT}, "text": "/start"}},
        {"update_id": 1},
    ],
)
def test_updates_that_are_not_private_text_messages_are_ignored(update: dict[str, Any]) -> None:
    repo = InMemoryUserRepository()

    assert reply(update, repo, FakeEntries(), 50) is None
    assert repo.count() == 0


# --- the webhook server -----------------------------------------------------------------------


@pytest.fixture
def server() -> Iterator[tuple[str, InMemoryUserRepository, RecordingSender]]:
    repo, sender = InMemoryUserRepository(), RecordingSender()
    srv = Server(("127.0.0.1", 0), Handler)
    srv.bot = Bot(Settings(), repo, sender, webhook_secret(TOKEN))
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}/", repo, sender
    srv.shutdown()
    srv.server_close()


def test_the_webhook_answers_an_update_carrying_the_secret(
    server: tuple[str, InMemoryUserRepository, RecordingSender],
) -> None:
    url, repo, sender = server

    response = httpx.post(url, json=_update("/start"), headers=SIGNED)

    assert response.status_code == 200
    assert repo.get(CHAT) is not None
    assert len(sender.sent) == 1 and sender.sent[0][0] == CHAT


@pytest.mark.parametrize("headers", [{}, {SECRET_HEADER: "wrong"}])
def test_the_webhook_drops_requests_without_the_secret(
    server: tuple[str, InMemoryUserRepository, RecordingSender], headers: dict[str, str]
) -> None:
    url, repo, sender = server

    assert httpx.post(url, json=_update("/start"), headers=headers).status_code == 403
    assert repo.count() == 0 and sender.sent == []


def test_the_webhook_refuses_a_body_that_is_not_json(
    server: tuple[str, InMemoryUserRepository, RecordingSender],
) -> None:
    url, _, sender = server

    response = httpx.post(url, content=b"not json", headers=SIGNED)

    assert response.status_code == 400 and sender.sent == []


def test_a_failure_while_handling_still_answers_200_so_telegram_does_not_resend(
    server: tuple[str, InMemoryUserRepository, RecordingSender], monkeypatch: pytest.MonkeyPatch
) -> None:
    url, repo, _ = server

    def broken(chat_id: int) -> None:
        raise RuntimeError("database down")

    monkeypatch.setattr(repo, "get", broken)

    response = httpx.post(url, json=_update("/start"), headers=SIGNED)

    assert response.status_code == 200


# --- registering the webhook ------------------------------------------------------------------


def test_the_secret_is_stable_per_token_and_uses_telegrams_alphabet() -> None:
    secret = webhook_secret(TOKEN)

    assert secret == webhook_secret(TOKEN) != webhook_secret("654321:other")
    assert len(secret) == 64 and secret.isalnum()
    assert TOKEN not in secret


def test_registering_sets_the_webhook_with_the_secret_and_the_command_menu() -> None:
    calls: list[tuple[str, dict[str, Any]]] = []

    def telegram(request: httpx.Request) -> httpx.Response:
        calls.append((request.url.path.rsplit("/", 1)[-1], json.loads(request.content)))
        return httpx.Response(200, json={"ok": True, "result": True})

    client = httpx.Client(transport=httpx.MockTransport(telegram))
    register(TOKEN, "https://fpl-bot.example.run.app", client)

    assert [method for method, _ in calls] == ["setWebhook", "setMyCommands"]
    webhook = calls[0][1]
    assert webhook["url"] == "https://fpl-bot.example.run.app"
    assert webhook["secret_token"] == webhook_secret(TOKEN)
    assert webhook["allowed_updates"] == ["message"]
    assert "start" in [c["command"] for c in calls[1][1]["commands"]]


def test_a_refused_registration_stops_with_telegrams_reason() -> None:
    def telegram(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"ok": False, "description": "bad webhook: HTTPS required"})

    with pytest.raises(SystemExit, match="HTTPS required"):
        register(TOKEN, "https://x", httpx.Client(transport=httpx.MockTransport(telegram)))
