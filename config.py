"""
AEGIS-ORBIT Configuration & Secure Credential Management
Loads settings from environment variables and .env file using pydantic-settings.
Provides credential validation and safe diagnostics with credential masking.
"""

import os
from functools import lru_cache
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Space-Track.org Credentials (Session-based, free account)
    spacetrack_username: str = Field(default="", alias="SPACETRACK_USERNAME")
    spacetrack_password: str = Field(default="", alias="SPACETRACK_PASSWORD")
    enable_spacetrack: bool = Field(default=False, alias="ENABLE_SPACETRACK")

    # Local SQLite Database
    database_url: str = Field(default="sqlite:///./aegis_orbit.db", alias="DATABASE_URL")

    # CelesTrak Configuration (Fair use: 2-6h intervals, polite User-Agent)
    celestrak_user_agent: str = Field(
        default="AEGIS-Orbit-SpaceDebrisAI/1.0 (aerospace-research-prototype; contact: contact@aegis-orbit.local)",
        alias="CELESTRAK_USER_AGENT"
    )
    celestrak_poll_interval_hours: int = Field(default=2, alias="CELESTRAK_POLL_INTERVAL_HOURS")

    # General App Settings
    app_env: str = Field(default="development", alias="APP_ENV")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    @property
    def has_spacetrack_credentials(self) -> bool:
        """Check if Space-Track username and password are provided and non-empty."""
        return bool(self.spacetrack_username.strip() and self.spacetrack_password.strip())

    @property
    def is_spacetrack_active(self) -> bool:
        """True only if feature flag is enabled AND credentials are configured."""
        return self.enable_spacetrack and self.has_spacetrack_credentials

    def get_masked_status(self) -> dict:
        """Return diagnostic dictionary masking sensitive credentials."""
        user_masked = (
            f"{self.spacetrack_username[:2]}***{self.spacetrack_username[-1]}"
            if len(self.spacetrack_username) > 3
            else ("SET" if self.spacetrack_username else "NOT_CONFIGURED")
        )
        return {
            "enable_spacetrack": self.enable_spacetrack,
            "spacetrack_configured": self.has_spacetrack_credentials,
            "spacetrack_user": user_masked,
            "spacetrack_active": self.is_spacetrack_active,
            "celestrak_user_agent": self.celestrak_user_agent,
            "database_url": self.database_url,
            "app_env": self.app_env
        }


@lru_cache()
def get_settings() -> Settings:
    """Return cached singleton application settings."""
    return Settings()
