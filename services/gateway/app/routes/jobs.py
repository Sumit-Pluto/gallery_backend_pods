"""Job polling for the async (diffusion) endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from crm_common.schemas import JobStatus

from .. import jobs as job_store
from ..deps import require_client

router = APIRouter(prefix="/v1/jobs", tags=["jobs"], dependencies=[Depends(require_client)])


@router.get("/{job_id}", response_model=JobStatus, summary="Poll a render job")
async def get_job(job_id: str, client: str = Depends(require_client)):
    """`status` is queued | running | done | error.

    On `done` the payload is under `result`. On `error` it is under `error`.
    Jobs are kept for JOB_TTL_SECONDS after they finish, then 404.
    """
    job = job_store.store.get(job_id, owner=client)
    return job.public()
