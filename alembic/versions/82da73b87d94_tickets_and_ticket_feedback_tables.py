"""tickets and ticket_feedback tables

Revision ID: 82da73b87d94
Revises: 
Create Date: 2026-07-14 20:51:59.234745

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = '82da73b87d94'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table('tickets',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('text', sa.Text(), nullable=False),
    sa.Column('language', sa.String(length=2), nullable=False),
    sa.Column('predicted_label', sa.String(length=20), nullable=False),
    sa.Column('confidence', sa.Float(), nullable=False),
    sa.Column('top_predictions', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
    sa.Column('needs_review', sa.Boolean(), nullable=False),
    sa.Column('status', sa.String(length=20), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_table('ticket_feedback',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('ticket_id', sa.Integer(), nullable=False),
    sa.Column('correct_label', sa.String(length=20), nullable=False),
    sa.Column('comment', sa.Text(), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.ForeignKeyConstraint(['ticket_id'], ['tickets.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_ticket_feedback_ticket_id'), 'ticket_feedback', ['ticket_id'], unique=False)


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(op.f('ix_ticket_feedback_ticket_id'), table_name='ticket_feedback')
    op.drop_table('ticket_feedback')
    op.drop_table('tickets')
