#!/usr/bin/env python3
"""
MLX-Whisper backend implementation for Apple Silicon
"""
import logging
from typing import Any, Iterator

from wyoming.info import Attribution

from .base import WhisperBackend

_LOGGER = logging.getLogger(__name__)


class MLXWhisperBackend(WhisperBackend):
    """MLX-Whisper backend implementation for Apple Silicon."""
    
    def __init__(self, model_name: str, **kwargs):
        try:
            import mlx_whisper
            self._mlx_whisper = mlx_whisper
            
            if model_name.startswith('mlx-community/'):
                self.model_path = model_name
            elif model_name in ['tiny', 'base', 'small', 'medium', 'large']:
                self.model_path = f"mlx-community/whisper-{model_name}-mlx"
            else:
                self.model_path = "mlx-community/whisper-tiny-mlx"
                
            _LOGGER.info(f"MLX Whisper backend initialized with model: {self.model_path}")
            
        except ImportError:
            raise ImportError("MLX Whisper not available. Install with: pip install mlx-whisper")
        except Exception as e:
            _LOGGER.error(f"Failed to initialize MLX backend: {e}")
            raise
    
    def transcribe(self, audio_path: str, **kwargs) -> Iterator[Any]:
        try:
            _LOGGER.debug(f"MLX transcribing audio file: {audio_path}")
            
            # Build transcription options
            transcribe_options = {
                'path_or_hf_repo': self.model_path,
                'verbose': False,
            }
            
            if kwargs.get('language'):
                transcribe_options['language'] = kwargs['language']
                
            if kwargs.get('initial_prompt'):
                transcribe_options['initial_prompt'] = kwargs['initial_prompt']
            
            _LOGGER.debug(f"MLX transcribe options: {transcribe_options}")
            
            result = self._mlx_whisper.transcribe(
                audio_path,
                **transcribe_options
            )
            
            _LOGGER.debug(f"MLX transcription result type: {type(result)}")
            _LOGGER.debug(f"MLX transcription result keys: {result.keys() if isinstance(result, dict) else 'not a dict'}")
            
            # Convert to compatible format with segments
            class Segment:
                def __init__(self, text: str, start: float = 0.0, end: float = 0.0):
                    self.text = text
                    self.start = start
                    self.end = end
            
            # MLX whisper should return a dict with 'segments' and/or 'text'
            if isinstance(result, dict):
                if 'segments' in result and result['segments']:
                    # Process segments
                    _LOGGER.debug(f"Found {len(result['segments'])} segments")
                    for i, segment in enumerate(result['segments']):
                        text = segment.get('text', '').strip()
                        if text:
                            _LOGGER.debug(f"Segment {i}: '{text}'")
                            yield Segment(
                                text=text,
                                start=segment.get('start', 0.0),
                                end=segment.get('end', 0.0)
                            )
                elif 'text' in result:
                    # Single text result
                    text = result['text'].strip()
                    _LOGGER.debug(f"Full text result: '{text}'")
                    if text:
                        yield Segment(text)
                    else:
                        _LOGGER.warning("Empty text result from MLX")
                else:
                    _LOGGER.warning(f"Unexpected MLX result format - missing 'segments' and 'text': {result}")
            else:
                _LOGGER.warning(f"MLX returned unexpected result type: {type(result)}")
                
        except Exception as e:
            _LOGGER.error(f"MLX transcription failed: {e}", exc_info=True)
            # Yield empty result instead of crashing
            class Segment:
                def __init__(self, text: str):
                    self.text = text
            yield Segment("")
    
    def get_supported_languages(self) -> list[str]:
        return ['en', 'es', 'fr', 'de', 'it', 'pt', 'ru', 'ja', 'ko', 'zh']
    
    def get_version(self) -> str:
        return getattr(self._mlx_whisper, '__version__', '1.0.0')
    
    def get_attribution(self) -> Attribution:
        return Attribution(
            name="MLX Community",
            url="https://github.com/ml-explore/mlx-examples/",
        ) 