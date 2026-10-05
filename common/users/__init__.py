"""Users of the bot: models, the storage port, and implementations (in-memory, Firestore)."""

from common.config import Settings
from common.users.memory import InMemoryUserRepository
from common.users.models import DeclaredTransfer, Invite, User, gameweek_key
from common.users.repository import UserRepository

__all__ = [
    "DeclaredTransfer",
    "InMemoryUserRepository",
    "Invite",
    "User",
    "UserRepository",
    "gameweek_key",
    "get_user_repository",
]


def get_user_repository(settings: Settings) -> UserRepository:
    """Firestore for registered users; in memory for single-user mode, where the only user is
    the owner named in the settings (TELEGRAM_CHAT_ID and FPL_TEAM_ID)."""
    if settings.users_backend == "memory":
        repo = InMemoryUserRepository()
        if settings.telegram_chat_id is not None and settings.fpl_team_id is not None:
            repo.save(User(chat_id=settings.telegram_chat_id, fpl_team_id=settings.fpl_team_id))
        return repo
    from google.cloud import firestore

    from common.users.firestore import FirestoreUserRepository

    client = firestore.Client(project=settings.gcp_project_id, database=settings.firestore_database)
    return FirestoreUserRepository(client)
