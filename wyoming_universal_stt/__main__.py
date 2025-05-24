#!/usr/bin/env python3
"""
Modular Wyoming Whisper Server supporting multiple backends
"""
import argparse
import asyncio
import logging
import platform
import re
from functools import partial

from wyoming.info import AsrModel, AsrProgram, Info
from wyoming.server import AsyncServer

from wyoming_universal_stt import __version__
from wyoming_universal_stt.handler import WhisperEventHandler
from wyoming_universal_stt.backends import WhisperBackendFactory, detect_optimal_backend

_LOGGER = logging.getLogger(__name__)


async def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        choices=WhisperBackendFactory.get_available_backends() + ["auto"],
        default="auto",
        help="Whisper backend to use (default: auto)",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Name of whisper model to use (or auto)",
    )
    parser.add_argument("--uri", required=True, help="unix:// or tcp://")
    parser.add_argument(
        "--data-dir",
        required=True,
        action="append",
        help="Data directory to check for downloaded models",
    )
    parser.add_argument(
        "--download-dir",
        help="Directory to download models into (default: first data dir)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device to use for inference (default: cpu, ignored for MLX)",
    )
    parser.add_argument(
        "--language",
        help="Default language to set for transcription",
    )
    parser.add_argument(
        "--compute-type",
        default="default",
        help="Compute type (float16, int8, etc., ignored for MLX)",
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=5,
        help="Size of beam during decoding (0 for auto, may be ignored by some backends)",
    )
    parser.add_argument(
        "--initial-prompt",
        help="Optional text to provide as a prompt for the first window",
    )
    parser.add_argument("--debug", action="store_true", help="Log DEBUG messages")
    parser.add_argument(
        "--log-format", default=logging.BASIC_FORMAT, help="Format for log messages"
    )
    parser.add_argument(
        "--version",
        action="version",
        version=__version__,
        help="Print version and exit",
    )
    args = parser.parse_args()

    if not args.download_dir:
        args.download_dir = args.data_dir[0]

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO, format=args.log_format
    )
    _LOGGER.debug(args)

    # Auto-detect backend if requested
    if args.backend == "auto":
        args.backend = detect_optimal_backend()
        _LOGGER.info("Auto-detected backend: %s", args.backend)

    # Automatic model selection
    machine = platform.machine().lower()
    is_arm = ("arm" in machine) or ("aarch" in machine)
    if args.model == "auto":
        if args.backend == "mlx-whisper":
            args.model = "mlx-community/whisper-tiny-mlx"
        else:
            args.model = "tiny-int8" if is_arm else "base-int8"
        _LOGGER.info("Model automatically selected: %s", args.model)

    # Handle beam size for ARM
    if args.beam_size <= 0:
        args.beam_size = 1 if is_arm else 5
        _LOGGER.debug("Beam size automatically selected: %s", args.beam_size)

    # Resolve model name for faster-whisper
    model_name = args.model
    if args.backend == "faster-whisper":
        match = re.match(r"^(tiny|base|small|medium)[.-]int8$", args.model)
        if match:
            model_size = match.group(1)
            model_name = f"{model_size}-int8"
            args.model = f"rhasspy/faster-whisper-{model_name}"

    if args.language == "auto":
        args.language = None

    # Create backend
    backend_kwargs = {
        'download_dir': args.download_dir,
        'device': args.device,
        'compute_type': args.compute_type,
    }
    
    try:
        whisper_backend = WhisperBackendFactory.create_backend(
            args.backend, args.model, **backend_kwargs
        )
        _LOGGER.info("Loaded %s backend with model %s", args.backend, args.model)
    except Exception as e:
        _LOGGER.error("Failed to load backend %s: %s", args.backend, e)
        return

    # Create Wyoming info
    wyoming_info = Info(
        asr=[
            AsrProgram(
                name=f"{args.backend}",
                description=f"Whisper transcription using {args.backend}",
                attribution=whisper_backend.get_attribution(),
                installed=True,
                version=__version__,
                models=[
                    AsrModel(
                        name=model_name,
                        description=model_name,
                        attribution=whisper_backend.get_attribution(),
                        installed=True,
                        languages=whisper_backend.get_supported_languages(),
                        version=whisper_backend.get_version(),
                    )
                ],
            )
        ],
    )

    server = AsyncServer.from_uri(args.uri)
    _LOGGER.info("Ready")
    model_lock = asyncio.Lock()
    await server.run(
        partial(
            WhisperEventHandler,
            wyoming_info,
            args,
            whisper_backend,
            model_lock,
            initial_prompt=args.initial_prompt,
        )
    )


def run() -> None:
    asyncio.run(main())


if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        pass
