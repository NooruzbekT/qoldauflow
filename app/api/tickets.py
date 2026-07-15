import anyio.to_thread
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from sqlalchemy.sql.base import ExecutableOption

from app.db.session import get_session
from app.ml.embedder import get_embedder
from app.ml.predictor import get_predictor
from app.models.ticket import Ticket, TicketFeedback
from app.schemas.ticket import (
    FeedbackCreate,
    FeedbackResponse,
    Label,
    SimilarTicketResponse,
    TicketCreate,
    TicketDetailResponse,
    TicketListResponse,
    TicketResponse,
    TicketStatus,
)

router = APIRouter(prefix="/tickets", tags=["tickets"])


async def get_ticket_or_404(
    session: AsyncSession, ticket_id: int, *options: ExecutableOption
) -> Ticket:
    query = select(Ticket).where(Ticket.id == ticket_id)
    if options:
        query = query.options(*options)
    ticket = await session.scalar(query)
    if ticket is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Ticket not found"
        )
    return ticket


@router.post("", response_model=TicketResponse, status_code=status.HTTP_201_CREATED)
async def create_ticket(
    payload: TicketCreate, session: AsyncSession = Depends(get_session)
) -> Ticket:
    predictor = get_predictor()
    if not predictor.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="ML model is not available"
        )

    # CPU-инференс уводится в threadpool, чтобы не блокировать event loop
    prediction = await anyio.to_thread.run_sync(predictor.predict, payload.text)
    embedder = get_embedder()
    embedding = None
    if embedder.is_ready:
        embedding = await anyio.to_thread.run_sync(embedder.embed, payload.text)

    ticket = Ticket(
        text=payload.text,
        language=payload.language.value,
        predicted_label=prediction.predicted_label,
        confidence=prediction.confidence,
        top_predictions=prediction.top_predictions,
        needs_review=prediction.needs_review,
        embedding=embedding,
    )
    session.add(ticket)
    await session.commit()
    await session.refresh(ticket, attribute_names=["created_at"])
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
    return TicketListResponse(items=list(rows), total=total, limit=limit, offset=offset)


@router.get("/{ticket_id}", response_model=TicketDetailResponse)
async def get_ticket(
    ticket_id: int, session: AsyncSession = Depends(get_session)
) -> Ticket:
    return await get_ticket_or_404(session, ticket_id, selectinload(Ticket.feedbacks))


@router.get("/{ticket_id}/similar", response_model=list[SimilarTicketResponse])
async def similar_tickets(
    ticket_id: int, session: AsyncSession = Depends(get_session)
) -> list[SimilarTicketResponse]:
    ticket = await get_ticket_or_404(session, ticket_id)
    if ticket.embedding is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Embedding is not available for this ticket",
        )

    distance = Ticket.embedding.cosine_distance(ticket.embedding).label("distance")
    rows = await session.execute(
        select(Ticket, distance)
        .where(Ticket.id != ticket_id, Ticket.embedding.is_not(None))
        .order_by(distance)
        .limit(5)
    )
    return [
        SimilarTicketResponse(
            id=row.Ticket.id,
            text=row.Ticket.text,
            language=row.Ticket.language,
            predicted_label=row.Ticket.predicted_label,
            # cosine_distance лежит в [0..2] — без клампа similarity ушла бы в минус
            similarity=round(max(0.0, 1 - row.distance), 4),
        )
        for row in rows
    ]


@router.post(
    "/{ticket_id}/feedback",
    response_model=FeedbackResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_feedback(
    ticket_id: int, payload: FeedbackCreate, session: AsyncSession = Depends(get_session)
) -> TicketFeedback:
    ticket = await get_ticket_or_404(session, ticket_id)

    feedback = TicketFeedback(
        ticket_id=ticket.id,
        correct_label=payload.correct_label.value,
        comment=payload.comment,
    )
    ticket.status = TicketStatus.reviewed.value
    session.add(feedback)
    await session.commit()
    await session.refresh(feedback, attribute_names=["created_at"])
    return feedback
