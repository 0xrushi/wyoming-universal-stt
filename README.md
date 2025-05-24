# Wyoming Universal STT

A universal [Wyoming protocol](https://github.com/rhasspy/wyoming) server that supports multiple Whisper backends for speech-to-text transcription. This project provides a unified interface to switch between different Whisper implementations without changing your client code.

I initially explored [rhasspy/wyoming-faster-whisper](https://github.com/rhasspy/wyoming-faster-whisper), but ran into limitations with MLX and other model support—this project was created to address those gaps.


## Supported Backends

- **faster-whisper** - Optimized local inference using CTranslate2
- **mlx-whisper** - Apple Silicon optimized inference (macOS only)  
- **openai-whisper-api** - OpenAI's proprietary API service

## Quick Start

### Installation

Install with your preferred backend:

```bash
# For faster-whisper (CPU/GPU optimized)
uv sync --extra faster-whisper

# For MLX (Apple Silicon only)
uv sync --extra mlx

# For OpenAI API
uv sync --extra openai-whisper

```

## Backend Switching

### Faster-Whisper Backend
Best for most users - optimized local inference with good speed/accuracy balance.

```bash
python -m wyoming_universal_stt \
    --backend faster-whisper \
    --model tiny \
    --language en \
    --uri 'tcp://0.0.0.0:10300' \
    --data-dir ./whisper-data \
    --download-dir ./whisper-data \
    --device cpu
```

**Available models:**
- `tiny`, `tiny-int8` - Fastest, least accurate
- `base`, `base-int8` - Good balance  
- `small`, `small-int8` - Better accuracy
- `medium`, `medium-int8` - Even better accuracy
- `large-v2`, `large-v3` - Best accuracy

**Devices:** `cpu`, `cuda`, `auto`

### MLX Backend (macOS Only)
Optimized for Apple Silicon with excellent performance and energy efficiency.

```bash
python -m wyoming_universal_stt \
    --backend mlx-whisper \
    --model tiny \
    --language en \
    --uri 'tcp://0.0.0.0:10300' \
    --data-dir ./whisper-data \
    --download-dir ./whisper-data
```

**Available models:**
- `tiny`, `base`, `small`, `medium`, `large`, `large-v2`, `large-v3`
- `mlx-community/whisper-tiny-mlx` - MLX-optimized models

### OpenAI API Backend
Uses OpenAI's proprietary Whisper API - requires internet and API key.

```bash
# Set your API key
export OPENAI_API_KEY="your-api-key-here"

python -m wyoming_universal_stt \
    --backend openai-whisper \
    --model whisper-1 \
    --language en \
    --uri 'tcp://0.0.0.0:10300' \
    --data-dir ./whisper-data \
    --download-dir ./whisper-data
```

**Benefits:**
- Always latest model
- No local compute needed
- Consistent performance
- No storage requirements

**Requirements:**
- OpenAI API key
- Internet connection
- Costs money per request

## Advanced Configuration

### Platform-Specific Examples

**Linux/Windows (no MLX):**
```bash
uv sync --extra faster-whisper
python -m wyoming_universal_stt --backend faster-whisper --model base --uri 'tcp://0.0.0.0:10300' --data-dir ./whisper-data
```

**macOS with Apple Silicon:**
```bash
uv sync --extra mlx
python -m wyoming_universal_stt --backend mlx-whisper --model tiny --uri 'tcp://0.0.0.0:10300' --data-dir ./whisper-data
```

**Cloud/API Usage:**
```bash
uv sync --extra openai-whisper
export OPENAI_API_KEY="sk-..."
python -m wyoming_universal_stt --backend openai-whisper --model whisper-1 --uri 'tcp://0.0.0.0:10300' --data-dir ./whisper-data
```

## Docker Support

### CPU-Optimized (Faster-Whisper)
```bash
docker build -t wyoming-universal-stt .
docker run -p 10300:10300 wyoming-universal-stt --backend faster-whisper --model tiny
```

### GPU Support
```bash
docker run --gpus all -p 10300:10300 wyoming-universal-stt --backend faster-whisper --model base --device cuda
```

## Integration with Home Assistant

This server is compatible with Home Assistant's Wyoming integration. Configure it as a speech-to-text provider:

```yaml
# configuration.yaml
wyoming:
  - uri: tcp://your-server:10300
```

## Troubleshooting

**MLX not available:**
- MLX only works on macOS with Apple Silicon (M1/M2/M3)
- Use `--backend faster-whisper` on other platforms

**Model download failures:**
- Check your `--download-dir` permissions
- Ensure sufficient disk space
- For OpenAI backend, verify API key


## License

MIT License - see LICENSE file for details.
