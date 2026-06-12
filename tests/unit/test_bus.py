"""Behavioural tests for :class:`~optimus.bus.nats.EventBus`.

The JetStream context and ``Msg`` are faked but faithful: the fake context
records publish headers (so we can assert ``Nats-Msg-Id`` dedup wiring) and hands
out a bounded queue of fake messages whose ``ack``/``nak``/``term`` are recorded.
This lets us prove, without a live NATS, that:

* a publish carrying ``msg_id`` sends the ``Nats-Msg-Id`` header (server-side
  dedup), and one without it does not;
* a handler failure naks (redeliver) and an undecodable payload terminates
  (poison-drop), while a success acks;
* concurrent in-flight processing stays bounded by ``max_inflight`` even when a
  burst of messages is available and handlers are slow.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from pydantic import BaseModel

from optimus.bus.nats import NATS_MSG_ID_HEADER, EventBus


class _Evt(BaseModel):
    correlation_id: str = "c"
    n: int = 0


class _FakeMsg:
    """A faithful stand-in for ``nats.aio.msg.Msg`` recording ack/nak/term."""

    def __init__(self, data: bytes) -> None:
        self.data = data
        self.acked = False
        self.naked = False
        self.termed = False

    async def ack(self) -> None:
        self.acked = True

    async def nak(self) -> None:
        self.naked = True

    async def term(self) -> None:
        self.termed = True


class _FakeSub:
    """Pull-subscription that drains a fixed list of messages then times out."""

    def __init__(self, msgs: list[_FakeMsg]) -> None:
        self._pending = list(msgs)
        self.fetch_sizes: list[int] = []

    async def fetch(self, batch: int, timeout: float = 5.0) -> list[_FakeMsg]:
        self.fetch_sizes.append(batch)
        if not self._pending:
            # Real NATS waits out the network timeout here, yielding to the loop;
            # emulate that so a busy poll cannot starve the stop signal.
            await asyncio.sleep(0)
            raise TimeoutError
        take = self._pending[:batch]
        self._pending = self._pending[batch:]
        return take


class _FakeJetStream:
    """Records publishes and serves a single pre-seeded pull subscription."""

    def __init__(self, sub: _FakeSub | None = None) -> None:
        self.published: list[tuple[str, bytes, dict[str, str] | None]] = []
        self._sub = sub
        self.consumer_configs: list[object] = []

    async def publish(
        self, subject: str, payload: bytes, headers: dict[str, str] | None = None, **_: object
    ) -> object:
        self.published.append((subject, payload, headers))
        return object()

    async def pull_subscribe(self, subject: str, durable: str, config: object) -> _FakeSub:
        self.consumer_configs.append(config)
        assert self._sub is not None
        return self._sub


def _bus(js: _FakeJetStream) -> EventBus:
    return EventBus(js)  # type: ignore[arg-type]


# --- publish dedup wiring ---------------------------------------------------


async def test_publish_sets_nats_msg_id_header_when_given() -> None:
    js = _FakeJetStream()
    await _bus(js).publish("events.verdict.v1", _Evt(), msg_id="idem-key")
    _subject, _payload, headers = js.published[0]
    assert headers == {NATS_MSG_ID_HEADER: "events.verdict.v1:idem-key"}


async def test_publish_without_msg_id_sends_no_header() -> None:
    js = _FakeJetStream()
    await _bus(js).publish("events.verdict.v1", _Evt())
    assert js.published[0][2] is None


async def test_publish_namespaces_msg_id_by_subject() -> None:
    js = _FakeJetStream()
    bus = _bus(js)
    await bus.publish("events.verdict.v1", _Evt(), msg_id="k")
    await bus.publish("events.action_result.v1", _Evt(), msg_id="k")
    ids = [h[NATS_MSG_ID_HEADER] for *_, h in js.published if h]
    # Same business key on two subjects must not collide into one dedup id.
    assert ids == ["events.verdict.v1:k", "events.action_result.v1:k"]


# --- dispatch ack / nak / term ---------------------------------------------


async def _run_once(js: _FakeJetStream, handler: Callable[[_Evt], Awaitable[None]]) -> None:
    stop = asyncio.Event()

    async def runner() -> None:
        await _bus(js).consume(
            "events.verdict.v1",
            durable="d",
            model=_Evt,
            handler=handler,
            fetch_timeout=0.01,
            stop_event=stop,
        )

    task = asyncio.create_task(runner())
    # Let the loop drain the seeded messages, then stop.
    await asyncio.sleep(0.05)
    stop.set()
    await asyncio.wait_for(task, timeout=2.0)


async def test_dispatch_acks_on_success() -> None:
    msg = _FakeMsg(_Evt(n=1).model_dump_json().encode())
    js = _FakeJetStream(_FakeSub([msg]))

    async def ok(_e: _Evt) -> None:
        return None

    await _run_once(js, ok)
    assert msg.acked and not msg.naked and not msg.termed


async def test_dispatch_naks_on_handler_failure() -> None:
    msg = _FakeMsg(_Evt(n=1).model_dump_json().encode())
    js = _FakeJetStream(_FakeSub([msg]))

    async def boom(_e: _Evt) -> None:
        raise RuntimeError("handler exploded")

    await _run_once(js, boom)
    assert msg.naked and not msg.acked


async def test_dispatch_terms_undecodable_payload() -> None:
    msg = _FakeMsg(b"not-json-at-all")
    js = _FakeJetStream(_FakeSub([msg]))

    async def never(_e: _Evt) -> None:  # pragma: no cover - must not be called
        raise AssertionError("handler should not run for poison message")

    await _run_once(js, never)
    assert msg.termed and not msg.acked and not msg.naked


# --- back-pressure: bounded in-flight ---------------------------------------


async def test_consume_bounds_inflight_under_burst() -> None:
    burst = 50
    max_inflight = 4
    msgs = [_FakeMsg(_Evt(n=i).model_dump_json().encode()) for i in range(burst)]
    sub = _FakeSub(msgs)
    js = _FakeJetStream(sub)

    inflight = 0
    peak = 0
    gate = asyncio.Event()

    async def slow(_e: _Evt) -> None:
        nonlocal inflight, peak
        inflight += 1
        peak = max(peak, inflight)
        # Hold until released so many messages pile up concurrently if unbounded.
        await gate.wait()
        inflight -= 1

    stop = asyncio.Event()

    async def runner() -> None:
        await _bus(js).consume(
            "events.image_fetched.v1",
            durable="d",
            model=_Evt,
            handler=slow,
            batch=32,
            fetch_timeout=0.01,
            max_inflight=max_inflight,
            stop_event=stop,
        )

    task = asyncio.create_task(runner())
    await asyncio.sleep(0.1)  # let the loop fill up to the in-flight ceiling
    assert peak <= max_inflight, f"in-flight {peak} exceeded ceiling {max_inflight}"
    assert peak == max_inflight, "should saturate the in-flight budget"
    # Never pull more than the in-flight budget in a single fetch.
    assert all(size <= max_inflight for size in sub.fetch_sizes)
    gate.set()
    stop.set()
    await asyncio.wait_for(task, timeout=2.0)


async def test_consume_sets_max_ack_pending_to_inflight() -> None:
    js = _FakeJetStream(_FakeSub([]))
    stop = asyncio.Event()

    async def runner() -> None:
        await _bus(js).consume(
            "events.image_fetched.v1",
            durable="d",
            model=_Evt,
            handler=lambda _e: asyncio.sleep(0),
            fetch_timeout=0.01,
            max_inflight=7,
            ack_wait=42.0,
            stop_event=stop,
        )

    task = asyncio.create_task(runner())
    await asyncio.sleep(0.02)
    stop.set()
    await asyncio.wait_for(task, timeout=2.0)
    cfg = js.consumer_configs[0]
    assert cfg.max_ack_pending == 7  # type: ignore[attr-defined]
    assert cfg.ack_wait == 42.0  # type: ignore[attr-defined]
