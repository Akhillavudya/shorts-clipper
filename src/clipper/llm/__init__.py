"""Provider-agnostic LLM layer.

FULL_PLAN.md section 2 assumed Anthropic. We run on free tiers instead, so the
plan's model choices moved into ``config/settings.yaml`` and became config
rather than code. Pipeline code asks for a logical *tier* - ``recall`` or
``rerank`` - and never names a model.

    from clipper.llm import Router

    router = Router(job_dir=store.job_dir(video_id))
    data = router.complete("recall", prompt, json_schema=SCHEMA)

Everything in this package exists to survive a free tier: a token-bucket
limiter that self-throttles below the published limits, exponential backoff
with jitter that honours ``Retry-After``, automatic failover to a second
provider, and an on-disk response cache so re-running a stage during
development does not respend quota.
"""

from clipper.llm.base import (
    AuthError,
    LLMError,
    LLMResponse,
    Provider,
    RateLimitError,
    ResponseFormatError,
    TransientError,
)
from clipper.llm.cache import LLMCache
from clipper.llm.limiter import DailyQuota, RateLimiter, TokenBucket
from clipper.llm.router import Router, TierConfig

__all__ = [
    "AuthError",
    "DailyQuota",
    "LLMCache",
    "LLMError",
    "LLMResponse",
    "Provider",
    "RateLimitError",
    "RateLimiter",
    "ResponseFormatError",
    "Router",
    "TierConfig",
    "TokenBucket",
    "TransientError",
]
