#!/usr/bin/env python

import asyncio
import io
import logging

import librosa
import numpy as np
import soundfile
from numpy.typing import NDArray
from pydantic import BaseModel
from websockets import Data
from websockets.server import WebSocketServerProtocol, serve

from whisper_online import SAMPLING_RATE, FasterWhisperASR, OnlineASRProcessor

HOST = "localhost"
PORT = 5555
LOG_LEVEL = logging.INFO


logging.basicConfig(
    level=LOG_LEVEL, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
logger.setLevel(LOG_LEVEL)


class ServerResponse(BaseModel):
    full_transcription: str


asr = FasterWhisperASR(
    language="en",
    model_size="tiny.en",
)


def ts_segments_to_text(segments) -> str:
    return "".join([segment[2] for segment in segments])


def parse_message_to_audio(message: Data) -> NDArray:
    if isinstance(message, str):
        raise ValueError("Received string message, expected binary")
    sf = soundfile.SoundFile(
        io.BytesIO(message),
        channels=1,
        endian="LITTLE",
        samplerate=SAMPLING_RATE,
        subtype="PCM_16",
        format="RAW",
    )
    audio, _ = librosa.load(sf, sr=SAMPLING_RATE, dtype=np.float32)
    return audio


async def consumer_handler(ws: WebSocketServerProtocol, online_asr: OnlineASRProcessor):
    logger.info("Starting consumer")
    async for message in ws:
        if message == "stop":
            logger.info("Received stop signal")
            return
        try:
            audio = parse_message_to_audio(message)
        except Exception as e:
            logger.error(f"Failed to parse message: {e}")
            await ws.close()
            return
        online_asr.insert_audio_chunk(audio)


async def producer_handler(ws: WebSocketServerProtocol, online_asr: OnlineASRProcessor):
    logger.info("Starting producer")
    while True:
        # if stop:
        #     logger.info("Received stop signal, finishing")
        #     await asyncio.to_thread(online_asr.process_iter)
        #     await asyncio.to_thread(online_asr.finish)
        #     commited_text = ts_segments_to_text(online_asr.commited)
        #     logger.info(f"Full transcript: {commited_text}")
        #     await ws.send(
        #         ServerResponse(full_transcription=commited_text).model_dump_json()
        #     )
        #     logger.info("Acknowledging stop signal")
        #     await ws.send("stop")
        #     await ws.close()
        #     logger.info("Closed connection")
        if len(online_asr.audio_buffer) > SAMPLING_RATE * 3:
            await asyncio.to_thread(online_asr.process_iter)
            commited_text = ts_segments_to_text(online_asr.commited)
            if type(commited_text) != str:
                logger.error(f"commited_text is not a string but {type(commited_text)}")
                return
            logger.info(f"Transcript: {commited_text}")
            await ws.send(
                ServerResponse(full_transcription=commited_text).model_dump_json()
            )
        else:
            await asyncio.sleep(0.1)


async def handler(ws: WebSocketServerProtocol):
    logger.info("Connection established")
    online_asr = await asyncio.to_thread(
        OnlineASRProcessor, asr, buffer_trimming_sec=15
    )
    logger.info("OnlineASRProcessor initialized")
    async with asyncio.TaskGroup() as tg:
        consumer_handler_task = tg.create_task(consumer_handler(ws, online_asr))
        consumer_handler_task.add_done_callback(lambda _: online_asr.finish())
        producer_handler_task = tg.create_task(producer_handler(ws, online_asr))

    # await asyncio.gather(
    #     consumer_handler(ws, online_asr),
    #     producer_handler(ws, online_asr),
    # )


async def main():
    async with serve(handler, HOST, PORT):
        await asyncio.Future()  # run forever


asyncio.run(main())
