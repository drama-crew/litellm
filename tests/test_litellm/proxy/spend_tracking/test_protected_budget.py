from decimal import Decimal


def test_cutover_requires_structured_writer_and_drain_evidence():
    import pytest
    from pydantic import ValidationError
    from litellm.proxy.spend_tracking.protected_budget import CutoverReceipt

    with pytest.raises(ValidationError):
        CutoverReceipt(receipt_id="unchecked", baselines={"spend:team:team": Decimal(0)})


async def test_direct_writer_propagates_durable_settlement_failure(monkeypatch):
    from unittest.mock import AsyncMock
    import pytest
    from litellm.proxy import proxy_server
    from litellm.proxy.db.db_spend_update_writer import DBSpendUpdateWriter
    from litellm.proxy.spend_tracking import spend_tracking_utils
    from litellm.proxy.video_endpoints import moderation_metering_runtime as runtime

    monkeypatch.setattr(proxy_server, "disable_spend_logs", True)
    monkeypatch.setattr(
        spend_tracking_utils,
        "get_logging_payload",
        lambda **kwargs: {
            "startTime": "synthetic",
            "endTime": "synthetic",
            "request_id": "durable-failure",
        },
    )
    monkeypatch.setattr(runtime, "settle_legacy", AsyncMock(side_effect=RuntimeError("durable pending")))
    with pytest.raises(RuntimeError, match="durable pending"):
        await DBSpendUpdateWriter().update_database(
            token=None,
            user_id=None,
            end_user_id=None,
            team_id="team",
            org_id=None,
            kwargs={},
            completion_response=None,
            start_time=None,
            end_time=None,
            response_cost=3,
        )


async def test_reservation_only_reconcile_cannot_issue_actual_phase_receipt():
    from decimal import Decimal
    from unittest.mock import AsyncMock
    import pytest
    from litellm.proxy.spend_tracking.protected_budget import ProtectedBudgetStore

    db = AsyncMock()
    authority = ProtectedBudgetStore(db, AsyncMock(), AsyncMock())
    with pytest.raises(ValueError, match="phase receipt requires actual debit"):
        await authority.mutate(
            "reconcile-only",
            ("spend:team:team",),
            kind="reconcile",
            amount=Decimal(3),
            reservation_id="reservation",
            phase_request_id="phase",
            phase_hash="hash",
        )
    db.query_raw.assert_not_awaited()
    db.execute_raw.assert_not_awaited()
