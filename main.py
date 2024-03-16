#!/usr/bin/env python3
import logging
import os
import time

import sounddevice as sd
from numpy import ndarray
from scipy.io.wavfile import write

from whisper_online import (SAMPLING_RATE, FasterWhisperASR,
                            OnlineASRProcessor, TimestampedSegment,
                            add_shared_args, load_audio, load_audio_chunk)

# Initalize the ASR model

# Start a separate thread for capturing the audio
# In the main thread listen for the 'q' key press
# When 'q' is pressed, stop the audio capture and pass the audio to the ASR model for processing
# Once the ASR model has processed the audio, add the transcript to the clipboard

SAMPLE_RATE = 16000
MAX_AUDIO_CAPTURE_DURATION_SECONDS = 60
MAX_SAMPLES = int(MAX_AUDIO_CAPTURE_DURATION_SECONDS * SAMPLE_RATE)
LOG_LEVEL = logging.INFO

logging = logging.getLogger(__name__)
logging.setLevel(LOG_LEVEL)


def clip(recording: ndarray, start: float, end: float):
    beggining = int(start * SAMPLE_RATE)
    end = int(end * SAMPLE_RATE)
    return recording[beggining:end]


asr = FasterWhisperASR(
    language="en",
    model_size="tiny.en",
)
online_asr = OnlineASRProcessor(asr)


recording = sd.rec(
    MAX_SAMPLES,
    samplerate=SAMPLE_RATE,
    channels=1,
    dtype="int16",
)

start = time.time()
logging.debug("Recording started... Press Ctrl+C to stop recording")

try:
    time.sleep(MAX_AUDIO_CAPTURE_DURATION_SECONDS)
except KeyboardInterrupt:
    pass

sd.stop()
end = time.time()
duration = end - start
logging.debug(f"Recording stopped. Duration: {duration} seconds")

write("output.wav", SAMPLE_RATE, clip(recording, 0, duration))

audio = load_audio("output.wav")
online_asr.insert_audio_chunk(audio)
logging.debug("Processing audio...")
o = online_asr.process_iter()
o = online_asr.finish()
logging.debug("Finished processing audio")
# output_transcript(o, now=now)
logging.debug(o[2].strip())
# execute xclip
print(o[2].strip())
# os.system(f"echo '{o[2].strip()}' | xclip -selection clipboard")
