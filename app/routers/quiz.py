from fastapi import APIRouter, HTTPException
from app.schemas.quiz import QuizGenerateRequest, QuizGenerateResponse
from app.services.quiz.generator import generate_quiz

router = APIRouter(prefix="/quiz", tags=["quiz"])


@router.post("/generate", response_model=QuizGenerateResponse)
async def generate(req: QuizGenerateRequest):
    try:
        result = await generate_quiz(
            user=req.user,
            quiz_type=req.quiz_type,
            topic=req.topic,
        )
        return QuizGenerateResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))