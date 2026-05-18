from fastapi import APIRouter

from app.models.schemas import (
    LLMRequest,
    LLMResponse,
)

from app.services.llm_service import (
    llm_service,
)

router = APIRouter()


@router.post(
    "/generate",
    response_model=LLMResponse,
)
def generate(request: LLMRequest):

    answer = llm_service.generate_answer(
        query=request.query,
        context=request.context,
    )

    return LLMResponse(answer=answer)