"""embedding column for similar search

Revision ID: 9c982cb4f754
Revises: 82da73b87d94
Create Date: 2026-07-15 00:17:43.638975

"""
from typing import Sequence, Union

import pgvector.sqlalchemy
import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = '9c982cb4f754'
down_revision: Union[str, Sequence[str], None] = '82da73b87d94'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.add_column('tickets', sa.Column('embedding', pgvector.sqlalchemy.vector.VECTOR(dim=384), nullable=True))
    # HNSW по косинусному расстоянию; на малых объёмах поиск точный и без индекса,
    # индекс закладывается на рост данных
    op.create_index(
        'ix_tickets_embedding_hnsw',
        'tickets',
        ['embedding'],
        postgresql_using='hnsw',
        postgresql_ops={'embedding': 'vector_cosine_ops'},
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index('ix_tickets_embedding_hnsw', table_name='tickets')
    op.drop_column('tickets', 'embedding')
