"""Event handler for clients of the modular whisper server."""
import argparse
import asyncio
import logging
import os
import tempfile
import wave
from typing import Optional

from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioStop
from wyoming.event import Event
from wyoming.info import Describe, Info
from wyoming.server import AsyncEventHandler

from .backends import WhisperBackend

_LOGGER = logging.getLogger(__name__)


class WhisperEventHandler(AsyncEventHandler):
    """Event handler for clients using modular whisper backends."""

    def __init__(
        self,
        wyoming_info: Info,
        cli_args: argparse.Namespace,
        backend: WhisperBackend,
        model_lock: asyncio.Lock,
        *args,
        initial_prompt: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.cli_args = cli_args
        self.wyoming_info_event = wyoming_info.event()
        self.backend = backend
        self.model_lock = model_lock
        self.initial_prompt = initial_prompt
        self._language = self.cli_args.language
        self._wav_dir = tempfile.TemporaryDirectory()
        self._wav_path = os.path.join(self._wav_dir.name, "speech.wav")
        self._wav_file: Optional[wave.Wave_write] = None

    async def handle_event(self, event: Event) -> bool:
        if AudioChunk.is_type(event.type):
            chunk = AudioChunk.from_event(event)
            if self._wav_file is None:
                _LOGGER.debug(f"Starting new audio recording: rate={chunk.rate}, width={chunk.width}, channels={chunk.channels}")
                self._wav_file = wave.open(self._wav_path, "wb")
                self._wav_file.setframerate(chunk.rate)
                self._wav_file.setsampwidth(chunk.width)
                self._wav_file.setnchannels(chunk.channels)
            self._wav_file.writeframes(chunk.audio)
            return True

        if AudioStop.is_type(event.type):
            _LOGGER.debug(
                "Audio stopped. Transcribing with initial prompt=%s",
                self.initial_prompt,
            )
            assert self._wav_file is not None
            self._wav_file.close()
            
            import os
            file_size = os.path.getsize(self._wav_path)
            _LOGGER.debug(f"Audio file saved: {self._wav_path}, size: {file_size} bytes")
            
            self._wav_file = None

            async with self.model_lock:
                transcription_kwargs = {
                    'language': self._language,
                    'initial_prompt': self.initial_prompt,
                }
                
                # Add backend-specific parameters
                if hasattr(self.cli_args, 'beam_size'):
                    transcription_kwargs['beam_size'] = self.cli_args.beam_size
                
                try:
                    _LOGGER.debug(f"Starting transcription with kwargs: {transcription_kwargs}")
                    segments = self.backend.transcribe(
                        self._wav_path, 
                        **transcription_kwargs
                    )
                    
                    # Collect all segments
                    segment_texts = []
                    for segment in segments:
                        if hasattr(segment, 'text') and segment.text:
                            segment_text = segment.text.strip()
                            if segment_text:
                                segment_texts.append(segment_text)
                                _LOGGER.debug(f"Got segment: '{segment_text}'")
                    
                    text = " ".join(segment_texts)
                    _LOGGER.info(f"Final transcription result: '{text}' (from {len(segment_texts)} segments)")
                    
                except Exception as e:
                    _LOGGER.error(f"Transcription failed: {e}", exc_info=True)
                    text = ""

            await self.write_event(Transcript(text=text).event())
            _LOGGER.debug("Completed request")

            # Reset
            self._language = self.cli_args.language

            return False

        if Transcribe.is_type(event.type):
            transcribe = Transcribe.from_event(event)
            if transcribe.language:
                self._language = transcribe.language
                _LOGGER.debug("Language set to %s", transcribe.language)
            return True

        if Describe.is_type(event.type):
            await self.write_event(self.wyoming_info_event)
            _LOGGER.debug("Sent info")
            return True

        return True

    def __del__(self):
        """Cleanup temporary directory."""
        if hasattr(self, '_wav_dir'):
            self._wav_dir.cleanup()