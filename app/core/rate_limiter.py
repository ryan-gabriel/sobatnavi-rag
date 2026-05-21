# app/core/rate_limiter.py
# ============================================================
# Centralised rate-limiter configuration for SobatNavi API.
#
# Uses slowapi (wraps the `limits` library) with an in-memory
# sliding-window store. No Redis required for single-process
# deployments. To scale across multiple workers/replicas, swap
# the storage backend to a Redis URI:
#
#   from slowapi import Limiter
#   from slowapi.util import get_remote_address
#   limiter = Limiter(
#       key_func=_identify_caller,
#       storage_uri="redis://localhost:6379",
#   )
#
# Rate-limit tiers (per sliding window):
#   chat             → 10  requests / 60 s  (expensive AI calls)
#   itinerary_write  → 30  requests / 60 s  (POST/PUT/PATCH/DELETE)
#   itinerary_read   → 60  requests / 60 s  (GET list/detail)
#   search           → 30  requests / 60 s  (place search)
#   health           → 120 requests / 60 s  (monitoring probes)
# ============================================================

import logging
from fastapi import Request
from slowapi import Limiter
from slowapi.util import get_remote_address

logger = logging.getLogger(__name__)


def _identify_caller(request: Request) -> str:
    """
    Key function for the rate limiter.

    Priority:
    1. Authenticated user ID (extracted from a processed request state if set
       by the auth dependency) — fairer and harder to spoof than IP.
    2. Forwarded IP from X-Forwarded-For (when behind a reverse proxy).
    3. Direct remote IP.

    The auth middleware sets `request.state.user_id` after validating the JWT.
    """
    # Prefer user_id set by auth dependency (set in api.py after token check)
    user_id: str | None = getattr(request.state, "user_id", None)
    if user_id:
        return f"user:{user_id}"

    # Respect reverse-proxy forwarded header
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        # Take only the first (leftmost) address — the real client IP
        client_ip = forwarded_for.split(",")[0].strip()
        if client_ip:
            return f"ip:{client_ip}"

    # Fallback to direct remote address
    return f"ip:{get_remote_address(request)}"


# ── Global limiter instance ──────────────────────────────────
# All endpoints import this single instance.
limiter = Limiter(
    key_func=_identify_caller,
    # default_limits apply to every route that doesn't have an explicit
    # @limiter.limit() decorator — acts as a global safety net.
    default_limits=["200/minute"],
)

# ── Rate limit strings (import these in api.py) ─────────────
RATE_CHAT            = "10/minute"      # POST /api/chat
RATE_ITINERARY_WRITE = "30/minute"      # POST/PUT/PATCH/DELETE itinerary
RATE_ITINERARY_READ  = "60/minute"      # GET itinerary list / detail
RATE_SEARCH          = "30/minute"      # GET place/search, place/recommendations
RATE_HEALTH          = "120/minute"     # GET /health
