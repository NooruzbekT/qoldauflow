"""labse embedding 768

Revision ID: b7d31c05e2aa
Revises: 9c982cb4f754
Create Date: 2026-07-15 10:20:00.000000

"""
from typing import Sequence, Union

from alembic import op

revision: str = 'b7d31c05e2aa'
down_revision: Union[str, Sequence[str], None] = '9c982cb4f754'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.drop_index('ix_tickets_embedding_hnsw', table_name='tickets')
    # эмбеддинги старой модели (MiniLM, 384) несовместимы с LaBSE — обнуляем
    op.execute("UPDATE tickets SET embedding = NULL")
    op.execute("ALTER TABLE tickets ALTER COLUMN embedding TYPE vector(768)")
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
    op.execute("UPDATE tickets SET embedding = NULL")
    op.execute("ALTER TABLE tickets ALTER COLUMN embedding TYPE vector(384)")
    op.create_index(
        'ix_tickets_embedding_hnsw',
        'tickets',
        ['embedding'],
        postgresql_using='hnsw',
        postgresql_ops={'embedding': 'vector_cosine_ops'},
    )
