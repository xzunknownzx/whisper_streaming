#!/usr/bin/env python

import asyncio
import logging
import signal
import subprocess

import websockets
from websockets import WebSocketClientProtocol

HOST = "localhost"
PORT = 5555
URL = f"ws://{HOST}:{PORT}"
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
    finally:
        process.kill()


async def consumer_cleanup(ws: WebSocketClientProtocol):
    await ws.send("stop")
    logger.info("Sent stop command")
    async for message in ws:
        logger.info(message)
        if message == "stop":
            break


async def consumer(ws: WebSocketClientProtocol):
    try:
        async for message in ws:
            logger.info(message)
    except asyncio.CancelledError:
        logger.info("Received KeyboardInterrupt, waiting for a full transcript")
        try:
            async with asyncio.timeout(10):
                await consumer_cleanup(ws)
        except asyncio.TimeoutError:
            logger.error("Did not receive a final transcript in time")
        await ws.close()


async def main():
    loop = asyncio.get_running_loop()
    loop.add_signal_handler(signal.SIGINT, loop.stop)

    async with websockets.connect(URL) as ws:
        async with asyncio.TaskGroup() as tg:
            tg.create_task(producer(ws))
            tg.create_task(consumer(ws))


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
