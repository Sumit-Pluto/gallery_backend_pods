"""LLM, chat and translation.

Three different backends behind one base URL:

  /v1/llm/chat          -> Groq passthrough on the cpu pod (keys stay server-side)
  /v1/chat/completions  -> the existing chat pod (Qwen via Ollama)
  /v1/translate         -> the existing chat pod (NLLB-200)

The chat pod is not managed by this repo. It is fronted here so the client has
exactly one host and one key to configure, and so you can move or replace it
later without telling them.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Body, Depends
from pydantic import BaseModel, ConfigDict, Field

from crm_common.schemas import LlmIn

from .. import upstream
from ..deps import require_client

router = APIRouter(prefix="/v1", tags=["llm"], dependencies=[Depends(require_client)])


@router.post("/llm/chat", summary="OpenAI-compatible chat/vision via Groq")
async def llm_chat(req: LlmIn):
    """Passthrough. You own the prompt, the model and the response schema.

    Returns `{"response": <full Groq chat completion>, "key_index": n}` — read the
    answer at `response.choices[0].message.content`.
    """
    return await upstream.cpu.post("/llm", req.model_dump(exclude_none=True))


@router.post("/chat/completions", summary="Chat with the self-hosted Qwen pod")
async def chat_completions(body: dict[str, Any] = Body(...)):
    """OpenAI-shaped. `stream` is forced off — the gateway returns one JSON body."""
    payload = dict(body)
    payload["stream"] = False
    return await upstream.chat.post("/v1/chat/completions", payload)


class TranslateIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(..., max_length=20000)
    source: str = Field(..., description="FLORES-200 code, e.g. eng_Latn")
    target: str = Field(..., description="FLORES-200 code, e.g. hin_Deva")


@router.post("/translate", summary="NLLB-200 translation")
async def translate(req: TranslateIn):
    return await upstream.translate.post("/translate", req.model_dump())
