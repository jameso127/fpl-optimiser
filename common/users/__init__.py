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
    """Firestore in production; in-memory for tests and local use."""
    if settings.users_backend == "memory":
        return InMemoryUserRepository()
    from google.cloud import firestore

    from common.users.firestore import FirestoreUserRepository

    client = firestore.Client(project=settings.gcp_project_id, database=settings.firestore_database)
    return FirestoreUserRepository(client)
