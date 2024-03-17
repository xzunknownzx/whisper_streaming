import logging

LOG_LEVEL = logging.DEBUG
logging = logging.getLogger(__name__)
logging.setLevel(LOG_LEVEL)


# import sounddevice as sd
from scipy.io.wavfile import write

fs = 16000
seconds = 1

myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
sd.wait()  # Wait until recording is finished
write("output.wav", fs, myrecording)  # Save as WAV file
