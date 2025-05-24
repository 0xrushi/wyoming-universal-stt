#!/usr/bin/env python3
"""
OpenAI Whisper API backend implementation
"""
import logging
import os
from typing import Any, Iterator

from wyoming.info import Attribution

from .base import WhisperBackend

_LOGGER = logging.getLogger(__name__)


class OpenAIWhisperBackend(WhisperBackend):
    """OpenAI Whisper API backend implementation."""
    
    def __init__(self, model_name: str, **kwargs):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("OpenAI package not installed. Install with: pip install openai")
        
        # Get API key from environment or kwargs
        api_key = kwargs.get('api_key') or os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY environment variable or pass api_key parameter"
            )
        
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name
        
        # Validate model name
        valid_models = ['whisper-1']
        if model_name not in valid_models:
            _LOGGER.warning(f"Model '{model_name}' not in known models {valid_models}. Using 'whisper-1'.")
            self.model_name = 'whisper-1'
    
    def transcribe(self, audio_path: str, **kwargs) -> Iterator[Any]:
        """Transcribe audio using OpenAI API."""
        
        class Segment:
            def __init__(self, text: str):
                self.text = text
        
        try:
            with open(audio_path, 'rb') as audio_file:
                # Prepare transcription parameters
                transcribe_kwargs = {
                    'model': self.model_name,
                    'file': audio_file,
                    'response_format': 'verbose_json',  # Get segments
                }
                
                # Add optional parameters
                if kwargs.get('language'):
                    transcribe_kwargs['language'] = kwargs['language']
                
                if kwargs.get('initial_prompt'):
                    transcribe_kwargs['prompt'] = kwargs['initial_prompt']
                
                _LOGGER.debug(f"Transcribing with OpenAI API: {audio_path}")
                response = self.client.audio.transcriptions.create(**transcribe_kwargs)
                
                # Handle response format
                if hasattr(response, 'segments') and response.segments:
                    # If we have segments, yield each one
                    for segment in response.segments:
                        yield Segment(segment.text)
                else:
                    # Fallback to full text if no segments
                    yield Segment(response.text)
                    
        except Exception as e:
            _LOGGER.error(f"OpenAI API transcription failed: {e}")
            raise
    
    def get_supported_languages(self) -> list[str]:
        """Return supported languages for OpenAI Whisper API."""
        # OpenAI Whisper supports these languages
        return [
            'af', 'am', 'ar', 'as', 'az', 'ba', 'be', 'bg', 'bn', 'bo', 'br', 'bs', 'ca', 'cs', 'cy', 'da', 'de', 'el',
            'en', 'es', 'et', 'eu', 'fa', 'fi', 'fo', 'fr', 'gl', 'gu', 'ha', 'haw', 'he', 'hi', 'hr', 'ht', 'hu', 'hy',
            'id', 'is', 'it', 'ja', 'jw', 'ka', 'kk', 'km', 'kn', 'ko', 'la', 'lb', 'ln', 'lo', 'lt', 'lv', 'mg', 'mi',
            'mk', 'ml', 'mn', 'mr', 'ms', 'mt', 'my', 'ne', 'nl', 'nn', 'no', 'oc', 'pa', 'pl', 'ps', 'pt', 'ro', 'ru',
            'sa', 'sd', 'si', 'sk', 'sl', 'sn', 'so', 'sq', 'sr', 'su', 'sv', 'sw', 'ta', 'te', 'tg', 'th', 'tk', 'tl',
            'tr', 'tt', 'uk', 'ur', 'uz', 'vi', 'yi', 'yo', 'zh'
        ]
    
    def get_version(self) -> str:
        """Return OpenAI API version."""
        try:
            import openai
            return f"openai-api-{openai.__version__}"
        except:
            return "openai-api-unknown"
    
    def get_attribution(self) -> Attribution:
        return Attribution(
            name="OpenAI",
            url="https://platform.openai.com/docs/guides/speech-to-text",
        ) 