#!/usr/bin/env python3
"""
Wyoming Whisper Server Test Client

Streams audio from the laptop microphone to a Wyoming Whisper ASR server and displays real-time transcriptions.

Purpose:
- Test server connectivity and transcription accuracy.
- Verify live audio is processed correctly by the server.

Use Case:
A quick diagnostic tool to confirm that your Wyoming Whisper setup is operational.
"""


import asyncio
import pyaudio
import numpy as np
import queue
import threading
import time
import collections
from typing import Optional, List, Tuple

# Wyoming library
from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioStart, AudioStop, AudioChunk
from wyoming.info import Describe, Info
from wyoming.client import AsyncTcpClient

class SimpleVAD:
    """Simple, reliable voice activity detection"""
    
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        self.energy_threshold = 500
        self.speech_frames = 0
        self.silence_frames = 0
        self.min_speech_frames = 3
        self.min_silence_frames = 30  # ~2 seconds
        self.is_speaking = False
        self.energy_history = collections.deque(maxlen=30)
        
    def calculate_energy(self, audio_data):
        """Calculate RMS energy"""
        try:
            audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
            return np.sqrt(np.mean(audio_np ** 2))
        except:
            return 0
    
    def process_frame(self, audio_data):
        """Process audio frame"""
        energy = self.calculate_energy(audio_data)
        self.energy_history.append(energy)
        
        # Simple adaptive threshold
        if len(self.energy_history) >= 10:
            avg_energy = np.mean(list(self.energy_history))
            threshold = max(self.energy_threshold, avg_energy * 2.5)
        else:
            threshold = self.energy_threshold
        
        # Debug output every 50 frames (~3 seconds)
        if len(self.energy_history) % 50 == 0:
            print(f"🔊 Audio level: {energy:.1f} (threshold: {threshold:.1f})")
        
        if energy > threshold:
            self.speech_frames += 1
            self.silence_frames = 0
            
            if not self.is_speaking and self.speech_frames >= self.min_speech_frames:
                self.is_speaking = True
                return "speech_start"
        else:
            self.silence_frames += 1
            if self.is_speaking and self.silence_frames >= self.min_silence_frames:
                self.is_speaking = False
                self.speech_frames = 0
                return "speech_end"
        
        return "speech" if self.is_speaking else "silence"


class FixedIntervalBuffer:
    """Fixed-interval audio buffer - simple and reliable"""
    
    def __init__(self, interval_seconds=5):
        self.interval_frames = int(interval_seconds * 16000 / 1024)  # 5 second intervals
        self.audio_buffer = []
        self.speech_detected = False
        self.frame_count = 0
        self.lock = threading.Lock()
        
    def add_frame(self, audio_data, is_speech=False):
        """Add audio frame"""
        with self.lock:
            self.audio_buffer.append(audio_data)
            if is_speech:
                self.speech_detected = True
            self.frame_count += 1
    
    def should_extract(self):
        """Check if we should extract current buffer"""
        with self.lock:
            return len(self.audio_buffer) >= self.interval_frames
    
    def extract_if_ready(self):
        """Extract buffer if ready and has speech"""
        with self.lock:
            if len(self.audio_buffer) >= self.interval_frames:
                if self.speech_detected or len(self.audio_buffer) >= self.interval_frames * 2:
                    # Extract buffer
                    audio_data = b''.join(self.audio_buffer)
                    had_speech = self.speech_detected
                    
                    # Keep overlap (last 25% of buffer)
                    overlap_frames = len(self.audio_buffer) // 4
                    self.audio_buffer = self.audio_buffer[-overlap_frames:] if overlap_frames > 0 else []
                    self.speech_detected = False
                    
                    return audio_data if had_speech else None
                else:
                    # No speech, just clear old data
                    self.audio_buffer = self.audio_buffer[-self.interval_frames//2:]
                    self.speech_detected = False
            
            return None


class ReliableTranscriber:
    """Reliable transcriber with fixed intervals"""
    
    def __init__(self, mic_index, host="localhost", port=10300):
        self.mic_index = mic_index
        self.host = host
        self.port = port
        
        # Components
        self.vad = SimpleVAD()
        self.audio_buffer = FixedIntervalBuffer(interval_seconds=6)  # 6 second intervals
        
        # Audio streaming
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.is_running = False
        
        # Async processing
        self.transcription_queue = asyncio.Queue(maxsize=3)
        self.processing_tasks = []
        
        # Stats
        self.segments_processed = 0
        self.start_time = time.time()
        
    async def start(self):
        """Start the reliable transcriber"""
        print("🚀 Starting Reliable Transcriber")
        
        # Test Wyoming connection
        try:
            client = AsyncTcpClient(self.host, self.port)
            await client.connect()
            await client.write_event(Describe().event())
            event = await asyncio.wait_for(client.read_event(), timeout=5)
            await client.disconnect()
            print("✓ Wyoming service verified")
        except Exception as e:
            print(f"❌ Failed to connect to Wyoming service: {e}")
            return False
        
        # Start audio streaming with error handling
        try:
            # Try different audio parameters to avoid ALSA issues
            self.stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=16000,
                input=True,
                input_device_index=self.mic_index,
                frames_per_buffer=1024,
                stream_callback=self._audio_callback
            )
            
            self.is_running = True
            self.stream.start_stream()
            print("🎤 Audio streaming started")
        except Exception as e:
            print(f"❌ Failed to start audio streaming: {e}")
            print("Audio errors (JACK/ALSA warnings) can usually be ignored if audio works")
            return False
        
        # Start transcription processors
        for i in range(2):  # Just 2 processors to avoid overload
            task = asyncio.create_task(self._transcription_processor(f"Proc-{i+1}"))
            self.processing_tasks.append(task)
        
        # Start extraction timer
        self.extraction_task = asyncio.create_task(self._extraction_timer())
        
        print("\n🎤 Listening with 6-second intervals...")
        print("Audio segments with speech will be transcribed.")
        print("Press Ctrl+C to stop.\n")
        
        return True
    
    def stop(self):
        """Stop the transcriber"""
        print(f"\n🛑 Stopping... (Processed {self.segments_processed} segments)")
        self.is_running = False
        
        if self.stream and self.stream.is_active():
            self.stream.stop_stream()
            self.stream.close()
        
        for task in self.processing_tasks:
            task.cancel()
        
        if hasattr(self, 'extraction_task'):
            self.extraction_task.cancel()
        
        self.audio.terminate()
        
        runtime = time.time() - self.start_time
        print(f"Runtime: {runtime:.1f}s")
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Audio callback"""
        if not self.is_running:
            return (in_data, pyaudio.paContinue)
        
        try:
            # Process with VAD
            vad_result = self.vad.process_frame(in_data)
            is_speech = vad_result in ["speech", "speech_start"]
            
            # Add to buffer
            self.audio_buffer.add_frame(in_data, is_speech)
            
            # Visual feedback
            if vad_result == "speech_start":
                print("🔴 Speech detected")
            elif vad_result == "speech_end":
                print("⚪ Speech ended")
            
            # Debug: show periodic stats
            if hasattr(self, '_callback_count'):
                self._callback_count += 1
            else:
                self._callback_count = 1
                
            if self._callback_count % 500 == 0:  # Every ~30 seconds
                print(f"📊 Audio callback running (count: {self._callback_count})")
                
        except Exception as e:
            print(f"Audio callback error: {e}")
        
        return (in_data, pyaudio.paContinue)
    
    async def _extraction_timer(self):
        """Extract audio at regular intervals"""
        try:
            while self.is_running:
                await asyncio.sleep(2)  # Check every 2 seconds
                
                try:
                    audio_data = self.audio_buffer.extract_if_ready()
                    if audio_data and len(audio_data) > 32000:  # At least 2 seconds
                        try:
                            self.transcription_queue.put_nowait(audio_data)
                            print("📦 Extracted segment for transcription")
                        except asyncio.QueueFull:
                            print("⚠️  Queue full")
                except Exception as e:
                    print(f"Extraction error: {e}")
                    
        except asyncio.CancelledError:
            pass
    
    async def _transcription_processor(self, processor_name):
        """Process transcriptions"""
        try:
            while self.is_running:
                try:
                    audio_data = await asyncio.wait_for(
                        self.transcription_queue.get(), timeout=3.0
                    )
                    
                    await self._transcribe_audio(audio_data, processor_name)
                    self.segments_processed += 1
                    
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    print(f"{processor_name} error: {e}")
        except asyncio.CancelledError:
            pass
    
    async def _transcribe_audio(self, audio_data, processor_name):
        """Transcribe audio"""
        try:
            client = AsyncTcpClient(self.host, self.port)
            await asyncio.wait_for(client.connect(), timeout=5)
            
            # Send transcription request
            await client.write_event(Transcribe().event())
            await client.write_event(AudioStart(rate=16000, width=2, channels=1).event())
            
            # Send audio
            chunk_size = 2048
            for i in range(0, len(audio_data), chunk_size):
                chunk_data = audio_data[i:i + chunk_size]
                chunk = AudioChunk(rate=16000, width=2, channels=1, audio=chunk_data)
                await client.write_event(chunk.event())
            
            await client.write_event(AudioStop().event())
            
            # Get result
            timeout_count = 0
            while timeout_count < 8:  # 8 second total timeout
                try:
                    event = await asyncio.wait_for(client.read_event(), timeout=1.0)
                    if event and Transcript.is_type(event.type):
                        transcript = Transcript.from_event(event)
                        text = transcript.text.strip()
                        if text:
                            duration = len(audio_data) / (16000 * 2)
                            print(f"📝 [{processor_name}] ({duration:.1f}s): {text}")
                        break
                except asyncio.TimeoutError:
                    timeout_count += 1
                    continue
            
            await client.disconnect()
            
        except Exception as e:
            print(f"[{processor_name}] Transcription failed: {e}")


async def select_microphone():
    """Select microphone with better error handling"""
    audio = pyaudio.PyAudio()
    
    print("\n🎤 Testing microphones...")
    working_mics = []
    
    for i in range(audio.get_device_count()):
        try:
            info = audio.get_device_info_by_index(i)
            if info['maxInputChannels'] > 0:
                try:
                    # Quick test
                    test_stream = audio.open(
                        format=pyaudio.paInt16,
                        channels=1,
                        rate=16000,
                        input=True,
                        input_device_index=i,
                        frames_per_buffer=1024
                    )
                    test_stream.close()
                    working_mics.append((i, info['name']))
                    print(f"✓ {info['name']} (index {i})")
                except Exception as e:
                    print(f"✗ {info['name']} - {str(e)[:50]}")
        except:
            pass
    
    audio.terminate()
    
    if not working_mics:
        print("❌ No working microphones found!")
        return None
    
    if len(working_mics) == 1:
        print(f"Using: {working_mics[0][1]}")
        return working_mics[0][0]
    
    print("\nSelect microphone:")
    for i, (idx, name) in enumerate(working_mics):
        print(f"{i + 1}. {name}")
    
    try:
        choice = int(input(f"Enter choice (1-{len(working_mics)}): ")) - 1
        if 0 <= choice < len(working_mics):
            return working_mics[choice][0]
    except:
        pass
    
    return working_mics[0][0]  # Default to first


async def main():
    print("Reliable Wyoming Client")
    print("=" * 25)
    
    # Select microphone
    mic_index = await select_microphone()
    if mic_index is None:
        return
    
    # Create transcriber
    transcriber = ReliableTranscriber(mic_index)
    
    try:
        if await transcriber.start():
            # Keep running
            while True:
                await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("\n\nShutting down...")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        transcriber.stop()


if __name__ == "__main__":
    try:
        import pyaudio
        import numpy as np
        from wyoming.client import AsyncTcpClient
        from wyoming.asr import Transcribe, Transcript
        from wyoming.audio import AudioStart, AudioStop, AudioChunk
        from wyoming.info import Describe, Info
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Install with: pip install pyaudio numpy wyoming")
        exit(1)
    
    asyncio.run(main())