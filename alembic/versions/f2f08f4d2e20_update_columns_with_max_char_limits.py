"""Update columns with max char limits

Revision ID: f2f08f4d2e20
Revises: fe2265adac4a
Create Date: 2020-08-31 17:14:54.012510

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'f2f08f4d2e20'
down_revision = 'fe2265adac4a'
branch_labels = None
depends_on = None


def upgrade():
    op.alter_column('tweets', 'tweet_id', type_=sa.String(20))
    op.alter_column('tweets', 'tweet', type_=sa.String(2000))
    op.alter_column('tweets', 'hashtags', type_=sa.ARRAY(sa.String(500)))
    op.alter_column('tweets', 'label', type_=sa.String(8))


def downgrade():
    op.alter_column('tweets', 'tweet_id', type_=sa.String())
    op.alter_column('tweets', 'tweet', type_=sa.String())
    op.alter_column('tweets', 'hashtags', type_=sa.ARRAY(sa.String()))
    op.alter_column('tweets', 'label', type_=sa.String())
