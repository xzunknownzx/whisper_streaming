from pydantic import BaseModel

HOST = "localhost"
PORT = 5555
URL = f"ws://{HOST}:{PORT}"


class TranscriptionData(BaseModel):
    transcription: str
    is_complete: bool = False
