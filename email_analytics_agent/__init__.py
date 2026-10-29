"""Meridian Email Analytics Agent proof-of-concept package."""

from .config import AgentConfig
from .models import CTDLabel, EmailMessage, ProcessResult

__all__ = ["AgentConfig", "CTDLabel", "EmailMessage", "ProcessResult"]

