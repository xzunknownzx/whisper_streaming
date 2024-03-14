import argparse
import logging
import sys
import time

from whisper_online import (SAMPLING_RATE, FasterWhisperASR,
                            OnlineASRProcessor, TimestampedSegment,
                            add_shared_args, load_audio, load_audio_chunk)

LOG_LEVEL = logging.ERROR
logging = logging.getLogger(__name__)
logging.setLevel(LOG_LEVEL)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "audio_path",
        type=str,
        help="Filename of 16kHz mono channel wav, on which live streaming is simulated.",
    )
    add_shared_args(parser)
    parser.add_argument(
        "--start_at",
        type=float,
        default=0.0,
        help="Start processing audio at this time.",
    )
    parser.add_argument(
        "--offline", action="store_true", default=False, help="Offline mode."
    )
    parser.add_argument(
        "--comp_unaware",
        action="store_true",
        default=False,
        help="Computationally unaware simulation.",
    )

    args = parser.parse_args()

    if args.offline and args.comp_unaware:
        logging.debug(
            "No or one option from --offline and --comp_unaware are available, not both. Exiting."
        )
        sys.exit(1)

    audio_path = args.audio_path

    duration = len(load_audio(audio_path)) / SAMPLING_RATE
    logging.debug("Audio duration is: %2.2f seconds" % duration)

    model_size = args.model_size

    t = time.time()
    logging.debug(f"Loading Whisper {model_size} model...")
    asr = FasterWhisperASR(
        model_size=model_size,
        cache_dir=args.model_cache_dir,
        model_dir=args.model_dir,
    )
    e = time.time()
    logging.debug(f"done. It took {round(e-t,2)} seconds.")

    min_chunk = args.min_chunk_size
    online_asr = OnlineASRProcessor(
        asr,
        buffer_trimming_sec=args.buffer_trimming_sec,
    )

    # load the audio into the LRU cache before we start the timer
    audio_chunk = load_audio_chunk(audio_path, 0, 1)

    # warm up the ASR, because the very first transcribe takes much more time than the other
    asr.transcribe(audio_chunk)

    beg: float = args.start_at
    start = time.time() - beg

    def output_transcript(o: TimestampedSegment, now=None) -> None:
        # output format in stdout is like:
        # 4186.3606 0 1720 Takhle to je
        # - the first three words are:
        #    - emission time from beginning of processing, in milliseconds
        #    - beg and end timestamp of the text segment, as estimated by Whisper model. The timestamps are not accurate, but they're useful anyway
        # - the next words: segment transcript
        if now is None:
            now = time.time() - start
        if o[0] is not None:
            logging.debug(
                "%1.4f %1.0f %1.0f %s" % (now * 1000, o[0] * 1000, o[1] * 1000, o[2])
            )
            logging.debug(
                "%1.4f %1.0f %1.0f %s" % (now * 1000, o[0] * 1000, o[1] * 1000, o[2])
            )
        else:
            logging.debug(o)

    if args.offline:  ## offline mode processing (for testing/debugging)
        audio_chunk = load_audio(audio_path)
        online_asr.insert_audio_chunk(audio_chunk)
        try:
            o = online_asr.process_iter()
        except AssertionError:
            logging.debug("assertion error")
            pass
        else:
            output_transcript(o)
        now = None
    elif args.comp_unaware:  # computational unaware mode
        end = beg + min_chunk
        while True:
            audio_chunk = load_audio_chunk(audio_path, beg, end)
            online_asr.insert_audio_chunk(audio_chunk)
            try:
                o = online_asr.process_iter()
            except AssertionError:
                logging.debug("assertion error")
                pass
            else:
                output_transcript(o, now=end)

            logging.debug(f"## last processed {end:.2f}s")

            if end >= duration:
                break

            beg = end

            if end + min_chunk > duration:
                end = duration
            else:
                end += min_chunk
        now = duration

    else:  # online = simultaneous mode
        end = 0
        while True:
            now = time.time() - start
            if now < end + min_chunk:
                time.sleep(min_chunk + end - now)
            end = time.time() - start
            audio_chunk = load_audio_chunk(audio_path, beg, end)
            beg = end
            online_asr.insert_audio_chunk(audio_chunk)

            try:
                o = online_asr.process_iter()
            except AssertionError:
                logging.debug("assertion error")
                pass
            else:
                output_transcript(o)
            now = time.time() - start
            logging.debug(
                f"## last processed {end:.2f} s, now is {now:.2f}, the latency is {now-end:.2f}"
            )

            if end >= duration:
                break
        now = None

    o = online_asr.finish()
    output_transcript(o, now=now)
