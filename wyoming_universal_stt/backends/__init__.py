#!/usr/bin/env python3
"""
Whisper backends package
"""
from .base import WhisperBackend
from .factory import WhisperBackendFactory, detect_optimal_backend

__all__ = ['WhisperBackend', 'WhisperBackendFactory', 'detect_optimal_backend'] 