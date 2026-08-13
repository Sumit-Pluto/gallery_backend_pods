"""Audio endpoints: transcription (Whisper, GPU) and denoise (ffmpeg, CPU)."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from crm_common.schemas import AudioOut, DenoiseIn, TranscribeIn, TranscribeOut

from .. import upstream
from ..deps import require_client

router = APIRouter(prefix="/v1/audio", tags=["audio"], dependencies=[Depends(require_client)])


@router.post("/transcribe", response_model=TranscribeOut, summary="Whisper transcription")
async def transcribe(req: TranscribeIn):
    return await upstream.vision.post("/transcribe", req.model_dump(exclude_none=True))


@router.post("/denoise", response_model=AudioOut, summary="RNNoise/afftdn denoise -> 48 kHz mono WAV")
async def denoise(req: DenoiseIn):
    return await upstream.cpu.post("/denoise", req.model_dump())
