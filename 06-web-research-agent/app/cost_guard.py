from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from fastapi import Depends, HTTPException

from app.auth import AuthenticatedUser, verify_api_key
from app.config import settings
from app.redis_client import get_redis


@dataclass(frozen=True)
class BudgetStatus:
    spent_usd: float
    remaining_usd: float
    budget_usd: float
    month_key: str


def _month_key() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m")


def _redis_key_for_user(user_id: str, month_key: str | None = None) -> str:
    return f"budget:{user_id}:{month_key or _month_key()}"


def get_budget_status(user_id: str) -> BudgetStatus:
    redis_client = get_redis()
    month_key = _month_key()
    key = _redis_key_for_user(user_id, month_key)
    spent_usd = float(redis_client.get(key) or 0.0)
    return BudgetStatus(
        spent_usd=spent_usd,
        remaining_usd=max(settings.monthly_budget_usd - spent_usd, 0.0),
        budget_usd=settings.monthly_budget_usd,
        month_key=month_key,
    )


def ensure_budget_available(user_id: str, projected_cost_usd: float = 0.0) -> BudgetStatus:
    status = get_budget_status(user_id)
    if status.spent_usd + projected_cost_usd > settings.monthly_budget_usd:
        raise HTTPException(
            status_code=402,
            detail={
                "error": "Monthly budget exceeded",
                "budget_usd": settings.monthly_budget_usd,
                "spent_usd": round(status.spent_usd, 6),
                "projected_cost_usd": round(projected_cost_usd, 6),
            },
        )
    return status


def check_budget(
    user: AuthenticatedUser = Depends(verify_api_key),
) -> BudgetStatus:
    return ensure_budget_available(user.user_id)


def estimate_llm_cost(input_tokens: int, output_tokens: int) -> float:
    input_cost = (input_tokens / 1000) * settings.input_cost_per_1k_tokens_usd
    output_cost = (output_tokens / 1000) * settings.output_cost_per_1k_tokens_usd
    return input_cost + output_cost


def record_cost(user_id: str, amount_usd: float) -> BudgetStatus:
    redis_client = get_redis()
    month_key = _month_key()
    key = _redis_key_for_user(user_id, month_key)
    redis_client.incrbyfloat(key, amount_usd)
    redis_client.expire(key, 60 * 60 * 24 * 40)
    return get_budget_status(user_id)
