#!/usr/bin/env python3
"""
Factory for creating Whisper backends
"""
import logging
import platform
from typing import Dict, Type

from .base import WhisperBackend
from .faster_whisper import FasterWhisperBackend
from .mlx_whisper import MLXWhisperBackend
from .openai_whisper_api import OpenAIWhisperBackend

_LOGGER = logging.getLogger(__name__)


class WhisperBackendFactory:
    """Factory for creating Whisper backends."""
    
    _backends: Dict[str, Type[WhisperBackend]] = {
        'faster-whisper': FasterWhisperBackend,
        'mlx-whisper': MLXWhisperBackend,
        'openai-whisper': OpenAIWhisperBackend,
    }
    
    @classmethod
    def register_backend(cls, name: str, backend_class: Type[WhisperBackend]):
        """Register a new backend."""
        cls._backends[name] = backend_class
    
    @classmethod
    def create_backend(cls, backend_name: str, model_name: str, **kwargs) -> WhisperBackend:
        """Create a backend instance."""
        if backend_name not in cls._backends:
            available = ', '.join(cls._backends.keys())
            raise ValueError(f"Unknown backend '{backend_name}'. Available: {available}")
        
        backend_class = cls._backends[backend_name]
        return backend_class(model_name, **kwargs)
    
    @classmethod
    def get_available_backends(cls) -> list[str]:
        """Get list of available backend names."""
        return list(cls._backends.keys())


def detect_optimal_backend() -> str:
    """Detect the optimal backend for the current system."""
    machine = platform.machine().lower()
    system = platform.system().lower()
    
    # Check for Apple Silicon
    if system == "darwin" and ("arm" in machine or "aarch" in machine):
        try:
            import mlx_whisper
            return "mlx-whisper"
        except ImportError:
            pass
    
    # Check for faster-whisper
    try:
        import faster_whisper
        return "faster-whisper"
    except ImportError:
        pass
    
    # Fallback to OpenAI whisper
    try:
        import whisper
        return "openai-whisper"
    except ImportError:
        raise ImportError("No Whisper backend available. Install one of: faster-whisper, mlx-whisper, or openai-whisper") 