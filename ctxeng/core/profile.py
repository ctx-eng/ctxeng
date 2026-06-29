from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4


@dataclass
class Preference:
    key: str
    value: str
    category: str = "general"
    updated_at: str = ""


@dataclass
class UserProfile:
    user_id: str
    name: str = ""
    preferences: Dict[str, Preference] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""


class ProfileStore:
    def __init__(self) -> None:
        self._profiles: Dict[str, UserProfile] = {}

    def get(self, user_id: str) -> Optional[UserProfile]:
        return self._profiles.get(user_id)

    def get_or_create(self, user_id: str, name: str = "") -> UserProfile:
        profile = self._profiles.get(user_id)
        if profile is None:
            now = datetime.utcnow().isoformat()
            profile = UserProfile(user_id=user_id, name=name, created_at=now, updated_at=now)
            self._profiles[user_id] = profile
        return profile

    def set_preference(self, user_id: str, key: str, value: str, category: str = "general") -> Preference:
        profile = self.get_or_create(user_id)
        pref = Preference(key=key, value=value, category=category, updated_at=datetime.utcnow().isoformat())
        profile.preferences[key] = pref
        profile.updated_at = datetime.utcnow().isoformat()
        return pref

    def get_preference(self, user_id: str, key: str) -> Optional[Preference]:
        profile = self.get(user_id)
        if profile is None:
            return None
        return profile.preferences.get(key)

    def set_tags(self, user_id: str, tags: List[str]) -> None:
        profile = self.get_or_create(user_id)
        profile.tags = tags
        profile.updated_at = datetime.utcnow().isoformat()

    def to_context(self, user_id: str) -> str:
        """Format profile as a context snippet for prompt injection."""
        profile = self.get(user_id)
        if profile is None:
            return "- no profile"

        parts: List[str] = []
        if profile.name:
            parts.append(f"Name: {profile.name}")
        if profile.tags:
            parts.append(f"Tags: {', '.join(profile.tags)}")
        if profile.preferences:
            lines = [f"  {p.key}: {p.value}" for p in profile.preferences.values()]
            parts.append("Preferences:\n" + "\n".join(lines))
        return "\n".join(parts) if parts else "- no profile"
