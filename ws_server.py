#!/usr/bin/env python

import asyncio
import io
import logging
from typing import AsyncGenerator

import numpy as np
import soundfile as sf
from numpy.typing import NDArray
from websockets.server import WebSocketServerProtocol, serve

from whisper_online import FasterWhisperASR, OnlineASRProcessor
from ws_shared import HOST, PORT, TranscriptionData

LOG_LEVEL = logging.INFO
SAMPLING_RATE = 16000
MIN_SECONDS_TO_PROCESS = 1.5
MIN_SAMPLES_TO_PROCESS = int(SAMPLING_RATE * MIN_SECONDS_TO_PROCESS)
MAX_SECONDS_WO_NEW_DATA = 1.5
TRANSCRIPTION_DELAY_SECONDS = 0.5

logging.basicConfig(
    level=LOG_LEVEL, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
logger.setLevel(LOG_LEVEL)


asr = FasterWhisperASR(
    language="en",
    model_size="tiny.en",
)


def ts_segments_to_text(segments) -> str:
    return "".join([segment[2] for segment in segments])


def parse_audio_bytes(data: bytes) -> NDArray:
    audio, _ = sf.read(
        io.BytesIO(data),
        format="RAW",
        channels=1,
        samplerate=SAMPLING_RATE,
        subtype="PCM_16",
        dtype="float32",
        endian="LITTLE",
    )
    return audio


class AudioBuffer:
    def __init__(self, min_chunk_size: int = MIN_SAMPLES_TO_PROCESS):
        self.min_chunk_size = min_chunk_size
        self.buffer = np.array([], dtype=np.float32)

    def append(self, audio_data):
        self.buffer = np.append(self.buffer, audio_data)

    def has_chunk(self):
        return len(self.buffer) >= self.min_chunk_size

    def get_chunk(self):
        if self.has_chunk():
            chunk = self.buffer[: self.min_chunk_size]
            self.buffer = self.buffer[self.min_chunk_size :]
            return chunk
        return None

    def get_remaining(self):
        if len(self.buffer) > 0:
            remaining_data = self.buffer
            self.buffer = np.array([], dtype=np.float32)
            return remaining_data
        return None


class Handler:
    def __init__(self, ws: WebSocketServerProtocol):
        self.ws = ws
        self.audio_buffer = AudioBuffer()
        self.stop = False

    async def consumer(self):
        logger.info("Starting consumer")
        async for message in self.ws:
            if message == "stop":
                logger.info("Received stop signal")
                self.stop = True
                return
            audio = parse_audio_bytes(message)
            self.audio_buffer.append(audio)

    async def transcribe_stream(
        self, online_asr: OnlineASRProcessor
    ) -> AsyncGenerator[tuple[str, bool], None]:
        while not self.stop:
            if self.audio_buffer.has_chunk():
                audio_chunk = self.audio_buffer.get_chunk()
                online_asr.insert_audio_chunk(audio_chunk)
                await asyncio.to_thread(online_asr.process_iter)
                yield ts_segments_to_text(online_asr.commited), False
            else:
                await asyncio.sleep(TRANSCRIPTION_DELAY_SECONDS)  # Wait for more data

        # Process any remaining audio data
        remaining_data = self.audio_buffer.get_remaining()
        if remaining_data is not None:
            online_asr.insert_audio_chunk(remaining_data)
            await asyncio.to_thread(online_asr.process_iter)
            await asyncio.to_thread(online_asr.finish)
            yield ts_segments_to_text(online_asr.commited), True

    # TODO: handle a scenario where the consumer keeps the socket open but isn't sending any data
    async def producer(self):
        logger.info("Starting producer")
        online_asr = OnlineASRProcessor(asr, buffer_trimming_sec=15)
        logger.info("OnlineASRProcessor initialized")
        async for transcript, is_complete in self.transcribe_stream(online_asr):
            await self.ws.send(
                TranscriptionData(
                    transcription=transcript, is_complete=is_complete
                ).model_dump_json()
            )

    async def handle(self):
        logger.info("Connection established")

        async with asyncio.TaskGroup() as tg:
            tg.create_task(self.consumer())
            tg.create_task(self.producer())


async def handler_wrapper(ws: WebSocketServerProtocol):
    await Handler(ws).handle()


async def main():
    async with serve(handler_wrapper, HOST, PORT):
        await asyncio.Future()  # run forever


asyncio.run(main())
