import logging
from decimal import Decimal

log = logging.getLogger(__name__)

# ILLUSTRATIVE VALUES, in USD per million tokens. Prices change, and cache
# multipliers differ by model: copy every number from the provider's
# pricing page, and add a row for each model you call, fallback included.
PRICES: dict[str, dict[str, Decimal]] = {
    "claude-sonnet-5-5": {
        "input": Decimal("2.00"),
        "output": Decimal("10.00"),
        "cache_read": Decimal("0.10"),
        "cache_write": Decimal("2.50"),  # 5-minute TTL; 1-hour costs more
    },
}
KINDS = ("input", "output", "cache_read", "cache_write")


def cost_usd(row: dict) -> Decimal | None:
    """Exact cost of one call. None (stored as NULL) for unknown models,
    so a missing price never throws away a call you already paid for."""
    price = PRICES.get(row["model"])
    if price is None:
        log.warning("no price configured for %s", row["model"])
        return None
    total = sum(row[f"{k}_tokens"] * price[k] for k in KINDS)
    return total / Decimal(1_000_000)
