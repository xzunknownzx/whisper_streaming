import io
import logging
import socket

import librosa
import numpy as np
import soundfile

from whisper_online import SAMPLING_RATE, FasterWhisperASR, OnlineASRProcessor

HOST = "localhost"
PORT = 65432
LOG_LEVEL = logging.INFO

BUFFER_SIZE = 4096
MIN_SECONDS_TO_PROCESS = 3
MIN_SAMPLES_TO_PROCESS = SAMPLING_RATE * MIN_SECONDS_TO_PROCESS

logging.basicConfig(
    level=LOG_LEVEL, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
logger.setLevel(LOG_LEVEL)

asr = FasterWhisperASR(
    language="en",
    model_size="tiny.en",
)


def timestamped_segments_to_text(segments):
    return "".join([segment[2] for segment in segments])


def process_incoming_data(data: bytes) -> np.ndarray:
    sf = soundfile.SoundFile(
        io.BytesIO(data),
        channels=1,
        endian="LITTLE",
        samplerate=SAMPLING_RATE,
        subtype="PCM_16",
        format="RAW",
    )
    audio, _ = librosa.load(sf, sr=SAMPLING_RATE, dtype=np.float32)
    return audio


def handle_connection(client_socket: socket.socket):
    online_asr = OnlineASRProcessor(asr, buffer_trimming_sec=15)
    samples = np.array([], dtype=np.float32)
    while True:
        data = client_socket.recv(BUFFER_SIZE)
        if not data:
            online_asr.finish()
            # commited_text = timestamped_segments_to_text(online_asr.commited)
            # buffered_text = timestamped_segments_to_text(online_asr.transcript_buffer)
            # logger.info(f"online_asr.commited: {commited_text}")
            # logger.info(f"online_asr.transcript_buffer: {buffered_text}")
            # return "".join(commited_text + buffered_text)

        new_samples = process_incoming_data(data)
        samples = np.append(samples, new_samples)
        if len(samples) > MIN_SAMPLES_TO_PROCESS:
            logger.info(f"Transcribing {len(samples)} samples")
            online_asr.insert_audio_chunk(samples.copy())
            online_asr.process_iter()
            samples = np.array([], dtype=np.float32)
            logger.info(
                f"Transcription so far: {timestamped_segments_to_text(online_asr.commited)}"
            )
            client_socket.sendall(
                timestamped_segments_to_text(online_asr.commited).encode("utf-8")
            )


with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
    logger.info(f"Server listening on {HOST}:{PORT}")
    server_socket.bind((HOST, PORT))
    server_socket.listen()
    while True:
        client_socket, client_addr = server_socket.accept()
        logger.info(f"Received connection from {client_socket.getpeername()}")
        try:
            transcription = handle_connection(client_socket)
        except ConnectionResetError:
            logger.info(f"Connection reset by {client_socket.getpeername()}")

        # logger.info(f"Final transcription: {transcription}")
        # client_socket.sendall(transcription.encode("utf-8"))
