#!/usr/bin/env python3
"""
Faster-Whisper backend implementation
"""
import logging
from typing import Any, Iterator

from wyoming.info import Attribution

from .base import WhisperBackend

_LOGGER = logging.getLogger(__name__)


class FasterWhisperBackend(WhisperBackend):
    """Faster-Whisper backend implementation."""
    
    def __init__(self, model_name: str, **kwargs):
        import faster_whisper
        self.model = faster_whisper.WhisperModel(
            model_name,
            download_root=kwargs.get('download_dir'),
            device=kwargs.get('device', 'cpu'),
            compute_type=kwargs.get('compute_type', 'default'),
        )
        self._faster_whisper = faster_whisper
    
    def transcribe(self, audio_path: str, **kwargs) -> Iterator[Any]:
        segments, _info = self.model.transcribe(
            audio_path,
            beam_size=kwargs.get('beam_size', 5),
            language=kwargs.get('language'),
            initial_prompt=kwargs.get('initial_prompt'),
        )
        return segments
    
    def get_supported_languages(self) -> list[str]:
        return list(self._faster_whisper.tokenizer._LANGUAGE_CODES)
    
    def get_version(self) -> str:
        return self._faster_whisper.__version__
    
    def get_attribution(self) -> Attribution:
        return Attribution(
            name="Guillaume Klein",
            url="https://github.com/guillaumekln/faster-whisper/",
        ) 