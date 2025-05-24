#!/usr/bin/env python3
"""
Base Whisper backend implementation
"""
import logging
from abc import ABC, abstractmethod
from typing import Any, Iterator

from wyoming.info import Attribution

_LOGGER = logging.getLogger(__name__)


class WhisperBackend(ABC):
    """Abstract base class for Whisper backends."""
    
    @abstractmethod
    def __init__(self, model_name: str, **kwargs):
        """Initialize the backend with model and configuration."""
        pass
    
    @abstractmethod
    def transcribe(self, audio_path: str, **kwargs) -> Iterator[Any]:
        """Transcribe audio file and return segments."""
        pass
    
    @abstractmethod
    def get_supported_languages(self) -> list[str]:
        """Return list of supported language codes."""
        pass
    
    @abstractmethod
    def get_version(self) -> str:
        """Return backend version."""
        pass
    
    @abstractmethod
    def get_attribution(self) -> Attribution:
        """Return attribution information."""
        pass 