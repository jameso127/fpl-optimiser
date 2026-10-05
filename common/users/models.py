"""The data held about each person using the bot: the minimum needed, nothing sensitive.

A user is a Telegram chat plus a public FPL team id and a few settings. There are no names,
emails or credentials. Everything here can be deleted with the `/delete` command.
"""

import datetime as dt
import re

from pydantic import BaseModel, Field, field_validator

# Declared transfers and invites are deleted automatically (Firestore TTL) after this long.
DECLARED_TRANSFER_RETENTION = dt.timedelta(days=30)


# Invite codes become Firestore document ids, so they are restricted to a safe alphabet.
_INVITE_CODE = re.compile(r"^[A-Za-z0-9_-]{6,64}$")


def is_valid_invite_code(code: str) -> bool:
    return bool(_INVITE_CODE.fullmatch(code))


def utcnow() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


def gameweek_key(season: str, gameweek: int) -> str:
    return f"{season}:{gameweek}"


class User(BaseModel):
    chat_id: int
    fpl_team_id: int | None = None
    active: bool = True  # /pause sets this to false
    max_transfers: int | None = None  # personal override of how many transfers to consider
    free_transfers_override: int | None = None  # when our estimate of their free transfers is wrong
    created_at: dt.datetime = Field(default_factory=utcnow)
    last_notified: str | None = None  # gameweek_key of the last gameweek they were messaged for


class DeclaredTransfer(BaseModel):
    """A transfer the user says they have already made for a gameweek.

    FPL does not show a manager's changes for an upcoming gameweek until after its deadline, so
    the user tells us. The prices are recorded at the moment of declaring: the player sold
    goes at his selling price, the player bought at his market price, so the bank and the
    player's later selling price stay correct even if prices move afterwards.
    """

    season: str
    gameweek: int  # the gameweek the transfer is for
    out_id: int
    in_id: int
    out_price: int  # selling price of the player sold, in tenths of a million
    in_price: int  # market price of the player bought
    declared_at: dt.datetime = Field(default_factory=utcnow)
    expires_at: dt.datetime = Field(default_factory=lambda: utcnow() + DECLARED_TRANSFER_RETENTION)


class Invite(BaseModel):
    """A one-off (or few-use) code that lets someone register. Registration is by invitation
    so that strangers cannot add load on the FPL API or on this project's costs."""

    code: str
    uses_left: int = 1
    created_by: int | None = None
    created_at: dt.datetime = Field(default_factory=utcnow)
    expires_at: dt.datetime = Field(default_factory=lambda: utcnow() + dt.timedelta(days=7))

    @field_validator("code")
    @classmethod
    def _safe_code(cls, code: str) -> str:
        if not is_valid_invite_code(code):
            raise ValueError("invite codes are 6-64 characters: letters, digits, - and _")
        return code
