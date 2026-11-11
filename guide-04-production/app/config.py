"""Settings for every feature. Read from the environment, never hard-coded."""
import os
from decimal import Decimal
from zoneinfo import ZoneInfo

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://app:app@localhost/app")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Model IDs go stale. Set each one to a current ID from the provider's docs.
PRIMARY_MODEL = os.getenv("LLM_MODEL", "claude-sonnet-5-5")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "claude-opus-5-5")  # not the SUT
# A model from a second provider, e.g. "<your-model-id>". Unset = no fallback.
FALLBACK_MODEL = os.getenv("FALLBACK_MODEL")
EMBED_MODEL = os.getenv("EMBED_MODEL", "<your-embedding-model-id>")
EMBED_DIM = int(os.getenv("EMBED_DIM", "1024"))  # must match vector(N)

# One business timezone decides what "today" means everywhere.
BUSINESS_TZ = ZoneInfo(os.getenv("BUSINESS_TZ", "America/Los_Angeles"))
DAILY_BUDGET_USD = Decimal(os.getenv("DAILY_BUDGET_USD", "50"))
TRACK_USAGE = os.getenv("TRACK_USAGE", "1") == "1"  # CI sets this to 0
