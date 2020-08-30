"""create tweets table

Revision ID: d23ea47887bb
Revises:
Create Date: 2020-08-27 10:13:33.165394

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'd23ea47887bb'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    op.create_table('tweets',
                    sa.Column('tweet_id', sa.String(), nullable=False),
                    sa.Column('tweet', sa.String(), nullable=False),
                    sa.Column('hashtags', sa.ARRAY(sa.String()), nullable=False),
                    sa.Column('label', sa.String(), nullable=True),
                    sa.Column('tweet_created_at', sa.DateTime(), nullable=False),
                    sa.Column('updated_at', sa.DateTime(), nullable=False),
                    sa.Column('annotated_at', sa.DateTime(), nullable=False)
                    )


def downgrade():
    op.drop_table('tweets')
