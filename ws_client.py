#!/usr/bin/env python

import asyncio
import json
import logging
import signal
import subprocess

import websockets
from websockets import WebSocketClientProtocol

from ws_shared import URL, TranscriptionData

LOG_LEVEL = logging.INFO


logging.basicConfig(
    level=LOG_LEVEL, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
logger.setLevel(LOG_LEVEL)


async def producer(ws: WebSocketClientProtocol):
    # Set up the arecord command
    arecord_command = [
        "arecord",
        "-f",
        "S16_LE",
        "-c1",
        "-r",
        "16000",
        "-t",
        "raw",
        "-D",
        "default",
    ]
    # Open a subprocess that runs the arecord command
    process = subprocess.Popen(arecord_command, stdout=subprocess.PIPE)
    if process.stdout is None:
        raise Exception("stdout is None")
    try:
        while True:
            data = process.stdout.read(8192)
            if not data:
                break
            # Send the audio data through the WebSocket
            await ws.send(data)
            await asyncio.sleep(0.1)
    except Exception as e:
        if not isinstance(e, asyncio.CancelledError):
            logger.error(
                "Received unexpected error", exc_info=True
            )  # NOTE: will `exc_info` print the stack trace?
    finally:
        logger.info("Stopping the producer")
        process.kill()
        process.wait()
        logger.info("Killed the arecord process")


def copy_to_clipboard(text: str) -> None:
    process = subprocess.Popen(
        ["xclip", "-selection", "clipboard"], stdin=subprocess.PIPE
    )
    process.communicate(input=text.encode("utf-8"))
    logger.info("Transcript copied to clipboard")
    # NOTE: should the process be explicitly killed here?
    # TODO: handle the case when the process fails


def send_desktop_notification(title: str, description: str | None = None) -> None:
    if description is None:
        cmd = ["notify-desktop", title]
    else:
        cmd = ["notify-desktop", title, description]
    _ = subprocess.Popen(cmd)
    # NOTE: should the process be explicitly killed here?
    logger.info("Sent desktop notification")


def handle_transcription(transcription_data: TranscriptionData) -> None:
    if len(transcription_data.transcription) == 0:
        return
    copy_to_clipboard(transcription_data.transcription)
    if transcription_data.is_complete:
        send_desktop_notification(transcription_data.transcription)


async def consumer(ws: WebSocketClientProtocol):
    async for message in ws:
        logger.info(message)
        transcription_data = TranscriptionData(**json.loads(message))
        handle_transcription(transcription_data)


async def main():
    async with websockets.connect(URL) as ws:
        loop = asyncio.get_running_loop()
        async with asyncio.TaskGroup() as tg:
            producer_task = tg.create_task(producer(ws))
            tg.create_task(consumer(ws))

            async def on_interrupt():
                logger.info("Received SIGINT")
                producer_task.cancel()
                await producer_task  # NOTE: doing this for some reason makes all the code below unreacheble
                # HACK: using sleep instead
                await asyncio.sleep(0.5)

                logger.info("Producer stopped")
                await ws.send("stop")
                logger.info("Sent stop command")

            # overrides the default behavior of the SIGINT signal
            loop.add_signal_handler(
                signal.SIGINT,
                lambda: asyncio.create_task(on_interrupt()),
            )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
