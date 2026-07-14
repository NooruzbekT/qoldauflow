from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.db.session import get_session
from app.ml.predictor import get_predictor
from app.models.ticket import Ticket, TicketFeedback
from app.schemas.ticket import (
    FeedbackCreate,
    FeedbackResponse,
    Label,
    TicketCreate,
    TicketDetailResponse,
    TicketListResponse,
    TicketResponse,
    TicketStatus,
)

router = APIRouter(prefix="/tickets", tags=["tickets"])


@router.post("", response_model=TicketResponse, status_code=status.HTTP_201_CREATED)
async def create_ticket(
    payload: TicketCreate, session: AsyncSession = Depends(get_session)
) -> Ticket:
    predictor = get_predictor()
    if not predictor.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="ML model is not available"
        )

    prediction = predictor.predict(payload.text)
    ticket = Ticket(
        text=payload.text,
        language=payload.language.value,
        predicted_label=prediction.predicted_label,
        confidence=prediction.confidence,
        top_predictions=prediction.top_predictions,
        needs_review=prediction.needs_review,
    )
    session.add(ticket)
    await session.commit()
    await session.refresh(ticket)
    return ticket


@router.get("", response_model=TicketListResponse)
async def list_tickets(
    session: AsyncSession = Depends(get_session),
    predicted_label: Label | None = None,
    needs_review: bool | None = None,
    status_filter: TicketStatus | None = Query(default=None, alias="status"),
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
) -> TicketListResponse:
    query = select(Ticket)
    if predicted_label is not None:
        query = query.where(Ticket.predicted_label == predicted_label.value)
    if needs_review is not None:
        query = query.where(Ticket.needs_review == needs_review)
    if status_filter is not None:
        query = query.where(Ticket.status == status_filter.value)

    total = await session.scalar(select(func.count()).select_from(query.subquery()))
    rows = await session.scalars(
        query.order_by(Ticket.id.desc()).limit(limit).offset(offset)
    )
    items = [TicketResponse.model_validate(row) for row in rows]
    return TicketListResponse(items=items, total=total, limit=limit, offset=offset)


@router.get("/{ticket_id}", response_model=TicketDetailResponse)
async def get_ticket(
    ticket_id: int, session: AsyncSession = Depends(get_session)
) -> Ticket:
    ticket = await session.scalar(
        select(Ticket).where(Ticket.id == ticket_id).options(selectinload(Ticket.feedbacks))
    )
    if ticket is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Ticket not found"
        )
    return ticket


@router.post(
    "/{ticket_id}/feedback",
    response_model=FeedbackResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_feedback(
    ticket_id: int, payload: FeedbackCreate, session: AsyncSession = Depends(get_session)
) -> TicketFeedback:
    ticket = await session.get(Ticket, ticket_id)
    if ticket is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Ticket not found"
        )

    feedback = TicketFeedback(
        ticket_id=ticket.id,
        correct_label=payload.correct_label.value,
        comment=payload.comment,
    )
    ticket.status = TicketStatus.reviewed.value
    session.add(feedback)
    await session.commit()
    await session.refresh(feedback)
    return feedback
