#!/usr/bin/env python3
import logging
from enum import StrEnum

# import librosa
import numpy as np
from faster_whisper import WhisperModel
from faster_whisper.transcribe import Segment

# from functools import lru_cache
# from pathlib import Path


SAMPLING_RATE = 16000
DEVICE = "cpu"
COMPUTE_TYPE = "int8"
LANGUAGE = "en"
TRIM_BUFFER_AFTER_SEC = 15


class ModelSize(StrEnum):
    TINY_EN = "tiny.en"
    TINY = "tiny"
    BASE_EN = "base.en"
    BASE = "base"
    SMALL_EN = "small.en"
    SMALL = "small"
    MEDIUM_EN = "medium.en"
    MEDIUM = "medium"
    LARGE_V1 = "large-v1"
    LARGE_V2 = "large-v2"
    LARGE_V3 = "large-v3"
    LARGE = "large"


TimestampedSegment = tuple[float, float, str]


# @lru_cache
# def load_audio(fname: str | Path) -> np.ndarray:
#     a, _ = librosa.load(fname, sr=SAMPLING_RATE, dtype=np.float32)
#     return a
#
#
# def load_audio_chunk(fname: str | Path, beg: float, end: float) -> np.ndarray:
#     audio = load_audio(fname)
#     beg_s = int(beg * SAMPLING_RATE)
#     end_s = int(end * SAMPLING_RATE)
#     return audio[beg_s:end_s]


class FasterWhisperASR:
    """Uses faster-whisper library as the backend. Works much faster, appx 4-times (in offline mode). For GPU, it requires installation with a specific CUDNN version."""

    sep = ""

    def __init__(
        self,
        language: str = LANGUAGE,
        model_size: str | None = None,
        cache_dir: str | None = None,
        model_dir: str | None = None,
    ):
        self.language = language

        self.model = self.load_model(model_size, cache_dir, model_dir)

    def load_model(
        self, model_size=None, cache_dir=None, model_dir=None
    ) -> WhisperModel:
        if model_dir is not None:
            logging.debug(
                f"Loading whisper model from model_dir {model_dir}. model_size and cache_dir parameters are not used."
            )
            model_size_or_path = model_dir
        elif model_size is not None:
            model_size_or_path = model_size
        else:
            raise ValueError("model_size or model_dir parameter must be set")

        model = WhisperModel(
            model_size_or_path,
            device=DEVICE,
            compute_type=COMPUTE_TYPE,
            download_root=cache_dir,
        )

        return model

    def transcribe(self, audio: np.ndarray, init_prompt="") -> list[Segment]:
        # tested: beam_size=5 is faster and better than 1 (on one 200 second document from En ESIC, min chunk 0.01)
        segments, info = self.model.transcribe(
            audio,
            language=self.language,
            initial_prompt=init_prompt,
            beam_size=5,
            word_timestamps=True,
            condition_on_previous_text=True,
        )
        # logging.debug(info)  # info contains language detection result

        return list(segments)

    def ts_words(self, segments: list[Segment]) -> list[TimestampedSegment]:
        o = []
        for segment in segments:
            for word in segment.words:
                # not stripping the spaces -- should not be merged with them!
                w = word.word
                t = (word.start, word.end, w)
                o.append(t)
        return o

    def segments_end_ts(self, res):
        return [s.end for s in res]


class HypothesisBuffer:
    def __init__(self):
        self.commited_in_buffer: list[TimestampedSegment] = []
        self.buffer: list[TimestampedSegment] = []
        self.new: list[TimestampedSegment] = []

        self.last_commited_time = 0
        self.last_commited_word: str | None = None

    def insert(self, new: list[TimestampedSegment], offset: float) -> None:
        # compare self.commited_in_buffer and new. It inserts only the words in new that extend the commited_in_buffer, it means they are roughly behind last_commited_time and new in content
        # the new tail is added to self.new

        new = [(start + offset, end + offset, contents) for start, end, contents in new]
        self.new = [
            (start, end, contents)
            for start, end, contents in new
            if start > self.last_commited_time - 0.1
        ]

        if len(self.new) >= 1:
            start, _, _ = self.new[0]
            if abs(start - self.last_commited_time) < 1:
                if self.commited_in_buffer:
                    # it's going to search for 1, 2, ..., 5 consecutive words (n-grams) that are identical in commited and new. If they are, they're dropped.
                    cn = len(self.commited_in_buffer)
                    nn = len(self.new)
                    for i in range(1, min(min(cn, nn), 5) + 1):  # 5 is the maximum
                        c = " ".join(
                            [self.commited_in_buffer[-j][2] for j in range(1, i + 1)][
                                ::-1
                            ]
                        )
                        tail = " ".join(self.new[j - 1][2] for j in range(1, i + 1))
                        if c == tail:
                            logging.debug("removing last", i, "words:")
                            for j in range(i):
                                logging.debug("\t", self.new.pop(0))
                            break

    def flush(self) -> list[TimestampedSegment]:
        # returns commited chunk = the longest common prefix of 2 last inserts.

        commit: list[TimestampedSegment] = []
        while self.new:
            start, end, contents = self.new[0]

            if len(self.buffer) == 0:
                break

            if contents == self.buffer[0][2]:
                commit.append((start, end, contents))
                self.last_commited_word = contents
                self.last_commited_time = end
                self.buffer.pop(0)
                self.new.pop(0)
            else:
                break
        self.buffer = self.new
        self.new = []
        self.commited_in_buffer.extend(commit)
        return commit

    def pop_commited(self, time):
        while self.commited_in_buffer and self.commited_in_buffer[0][1] <= time:
            self.commited_in_buffer.pop(0)

    def complete(self):
        return self.buffer


class OnlineASRProcessor:
    def __init__(
        self, asr: FasterWhisperASR, buffer_trimming_sec=TRIM_BUFFER_AFTER_SEC
    ):
        """asr: WhisperASR object
        buffer_trimming_sec: Buffer is trimmed if it is longer than "seconds" threshold. Default is the most recommended option.
        """
        self.asr = asr

        self.audio_buffer = np.array([], dtype=np.float32)
        self.buffer_time_offset = 0

        self.transcript_buffer = HypothesisBuffer()
        self.commited: list[TimestampedSegment] = []

        self.buffer_trimming_sec = buffer_trimming_sec

    def insert_audio_chunk(self, audio):
        self.audio_buffer = np.append(self.audio_buffer, audio)

    def prompt(self) -> tuple[str, str]:
        """Returns a tuple: (prompt, context), where "prompt" is a 200-character suffix of commited text that is inside of the scrolled away part of audio buffer.
        "context" is the commited text that is inside the audio buffer. It is transcribed again and skipped. It is returned only for debugging and logging reasons.
        """
        k = max(0, len(self.commited) - 1)
        while k > 0 and self.commited[k - 1][1] > self.buffer_time_offset:
            k -= 1

        p = self.commited[:k]
        p = [t for _, _, t in p]
        prompt = []
        l = 0
        while p and l < 200:  # 200 characters prompt size
            x = p.pop(-1)
            l += len(x) + 1
            prompt.append(x)
        non_prompt = self.commited[k:]
        return self.asr.sep.join(prompt[::-1]), self.asr.sep.join(
            t for _, _, t in non_prompt
        )

    def process_iter(self) -> TimestampedSegment:
        """Runs on the current audio buffer.
        Returns: a tuple (beg_timestamp, end_timestamp, "text"), or (None, None, "").
        The non-emty text is confirmed (committed) partial transcript.
        """

        prompt, non_prompt = self.prompt()
        logging.debug("PROMPT:", prompt)
        logging.debug("CONTEXT:", non_prompt)
        logging.debug(
            f"transcribing {len(self.audio_buffer)/SAMPLING_RATE:2.2f} seconds from {self.buffer_time_offset:2.2f}"
        )
        res = self.asr.transcribe(self.audio_buffer, init_prompt=prompt)

        # transform to [(beg,end,"word1"), ...]
        tsw = self.asr.ts_words(res)

        self.transcript_buffer.insert(tsw, self.buffer_time_offset)
        o = self.transcript_buffer.flush()
        self.commited.extend(o)
        logging.debug(">>>>COMPLETE NOW:", self.to_flush(o))
        logging.debug("INCOMPLETE:", self.to_flush(self.transcript_buffer.complete()))

        # there is a newly confirmed text

        # trim the completed segments longer than s,
        if len(self.audio_buffer) / SAMPLING_RATE > self.buffer_trimming_sec:
            self.chunk_completed_segment(res)

            # alternative: on any word
            # l = self.buffer_time_offset + len(self.audio_buffer)/self.SAMPLING_RATE - 10
            # let's find commited word that is less
            # k = len(self.commited)-1
            # while k>0 and self.commited[k][1] > l:
            #    k -= 1
            # t = self.commited[k][1]
            logging.debug(f"chunking segment")
            # self.chunk_at(t)

        logging.debug(f"len of buffer now: {len(self.audio_buffer)/SAMPLING_RATE:2.2f}")
        return self.to_flush(o)

    def chunk_completed_segment(self, res: list[Segment]):
        if self.commited == []:
            return

        ends = self.asr.segments_end_ts(res)

        t = self.commited[-1][1]

        if len(ends) > 1:
            e = ends[-2] + self.buffer_time_offset
            while len(ends) > 2 and e > t:
                ends.pop(-1)
                e = ends[-2] + self.buffer_time_offset
            if e <= t:
                logging.debug(f"--- segment chunked at {e:2.2f}")
                self.chunk_at(e)
            else:
                logging.debug(f"--- last segment not within commited area")
        else:
            logging.debug(f"--- not enough segments to chunk")

    def chunk_at(self, time):
        """trims the hypothesis and audio buffer at "time" """
        self.transcript_buffer.pop_commited(time)
        cut_seconds = time - self.buffer_time_offset
        self.audio_buffer = self.audio_buffer[int(cut_seconds * SAMPLING_RATE) :]
        self.buffer_time_offset = time

    def finish(self):
        """Flush the incomplete text when the whole processing ends.
        Returns: the same format as self.process_iter()
        """
        o = self.transcript_buffer.complete()
        f = self.to_flush(o)
        logging.debug("last, noncommited:", f)
        return f

    def to_flush(
        self,
        ts_segments: list[TimestampedSegment],
        sep=None,
        offset=0,
    ) -> TimestampedSegment | tuple[None, None, str]:
        # concatenates the timestamped words or sentences into one sequence that is flushed in one line
        # sents: [(beg1, end1, "sentence1"), ...] or [] if empty
        # return: (beg1,end-of-last-sentence,"concatenation of sentences") or (None, None, "") if empty
        if len(ts_segments) == 0:
            return (None, None, "")
        if sep is None:
            sep = self.asr.sep
        contents = sep.join(ts_segment[2] for ts_segment in ts_segments)
        start = offset + ts_segments[0][0]
        end = offset + ts_segments[-1][1]
        return (start, end, contents)


def add_shared_args(parser):
    """shared args for simulation (this entry point) and server
    parser: argparse.ArgumentParser object
    """
    parser.add_argument(
        "--min-chunk-size",
        type=float,
        default=1.0,
        help="Minimum audio chunk size in seconds. It waits up to this time to do processing. If the processing takes shorter time, it waits, otherwise it processes the whole segment that was received by this time.",
    )
    parser.add_argument(
        "--model-size",
        type=str,
        default=ModelSize.TINY_EN,
        choices=[e.value for e in ModelSize],
        help="Name size of the Whisper model to use (default: tiny.en). The model is automatically downloaded from the model hub if not present in model cache dir.",
    )
    parser.add_argument(
        "--model_cache_dir",
        type=str,
        default=None,
        help="Overriding the default model cache dir where models downloaded from the hub are saved",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default=None,
        help="Dir where Whisper model.bin and other files are saved. This option overrides --model and --model_cache_dir parameter.",
    )
    parser.add_argument(
        "--buffer_trimming_sec",
        type=float,
        default=15,
        help="Buffer trimming length threshold in seconds. If buffer length is longer, trimming sentence/segment is triggered.",
    )
