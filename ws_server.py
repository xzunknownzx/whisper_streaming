#!/usr/bin/env python

import asyncio
import io
import logging

import numpy as np
import soundfile as sf
from numpy.typing import NDArray
from websockets.server import WebSocketServerProtocol, serve

from whisper_online import FasterWhisperASR, OnlineASRProcessor
from ws_shared import HOST, PORT, TranscriptionData

LOG_LEVEL = logging.INFO
SAMPLING_RATE = 16000
MIN_SAMPLES_TO_PROCESS = SAMPLING_RATE * 1.5


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


def parse_message_to_audio(data: bytes) -> NDArray:
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


class Handler:
    def __init__(self, ws: WebSocketServerProtocol):
        self.ws = ws
        self.audio_buffer = np.array([], dtype=np.float32)
        self.beg, self.end = 0, 0
        self.stop = False

    async def consumer_handler(self):
        logger.info("Starting consumer")
        async for message in self.ws:
            if message == "stop":
                logger.info("Received stop signal")
                self.stop = True
                return
            audio = parse_message_to_audio(message)
            self.audio_buffer = np.append(self.audio_buffer, audio)

    async def transcribe_chunk(self, online_asr: OnlineASRProcessor):
        online_asr.insert_audio_chunk(self.audio_buffer[self.end :])
        self.end = len(self.audio_buffer)
        logger.info("Transcribing")
        await asyncio.to_thread(online_asr.process_iter)

    # TODO: handle a scenario where the consumer keeps the socket open but isn't sending any data
    async def producer_handler(self):
        logger.info("Starting producer")
        online_asr = OnlineASRProcessor(asr, buffer_trimming_sec=15)
        logger.info("OnlineASRProcessor initialized")
        while True:
            if self.stop:
                await self.transcribe_chunk(online_asr)
                logger.info("Stopping producer")
                await asyncio.to_thread(online_asr.finish)
                commited_text = ts_segments_to_text(online_asr.commited)
                logger.info(f"Transcript: {commited_text}")
                await self.ws.send(
                    TranscriptionData(
                        transcription=commited_text, is_complete=True
                    ).model_dump_json()
                )
                await self.ws.close()
                logger.info("Closed connection")
                return
            elif len(self.audio_buffer[self.end :]) > MIN_SAMPLES_TO_PROCESS:
                await self.transcribe_chunk(online_asr)
                commited_text = ts_segments_to_text(online_asr.commited)
                logger.info(f"Transcript: {commited_text}")
                await self.ws.send(
                    TranscriptionData(transcription=commited_text).model_dump_json()
                )
            else:
                logger.info(
                    f"Sleeping, buffer size: {len(self.audio_buffer[self.end :])}"
                )
            await asyncio.sleep(0.5)

    async def handle(self):
        logger.info("Connection established")

        async with asyncio.TaskGroup() as tg:
            tg.create_task(self.consumer_handler())
            tg.create_task(self.producer_handler())


async def handler_wrapper(ws: WebSocketServerProtocol):
    handler = Handler(ws)
    await handler.handle()


async def main():
    async with serve(handler_wrapper, HOST, PORT):
        await asyncio.Future()  # run forever


asyncio.run(main())
