"""
Schedule Provider Module.

This module calculates charging schedules based on electricity prices and time.
It does NOT control charging directly - it provides schedule information via callback.

Architecture:
- Fetches prices from price_fetcher
- Calculates optimal charging windows
- Notifies listeners when schedule changes (e.g., new hour starts)
- Does NOT know about FSM - just calls the callback
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from .config import get_config
from .price_fetcher import PriceFetcher, PriceForecast, HourlyPrice

if TYPE_CHECKING:
    from .charging_fsm import PhaseDetectionResult, ScheduleInfo


# =============================================================================
# SCHEDULE CONSUMER INTERFACE (Protocol)
# =============================================================================


@runtime_checkable
class ScheduleConsumer(Protocol):
    """
    Interface for schedule consumers.

    ScheduleProvider doesn't care who implements this - just calls it.
    """

    async def on_schedule_update(self, schedule: ScheduleInfo) -> None:
        """Called when schedule changes."""
        ...

logger = logging.getLogger(__name__)


@dataclass
class ChargingWindow:
    """A scheduled charging window."""

    start_time: datetime
    end_time: datetime
    price_sek: float
    is_night_rate: bool = False


@dataclass
class ChargingPlan:
    """A charging plan with multiple windows."""

    created_at: datetime
    windows: list[ChargingWindow] = field(default_factory=list)
    target_kwh: float = 0.0
    estimated_cost_sek: float = 0.0

    @property
    def is_active(self) -> bool:
        """Check if plan has any windows."""
        return len(self.windows) > 0

    def is_in_window(self, dt: datetime) -> bool:
        """Check if given time is in a charging window."""
        for window in self.windows:
            if window.start_time <= dt < window.end_time:
                return True
        return False

    def get_current_window(self, dt: datetime) -> ChargingWindow | None:
        """Get the window containing the given time."""
        for window in self.windows:
            if window.start_time <= dt < window.end_time:
                return window
        return None

    def get_next_window(self, dt: datetime) -> ChargingWindow | None:
        """Get the next upcoming window after given time."""
        future_windows = [w for w in self.windows if w.start_time > dt]
        if future_windows:
            return min(future_windows, key=lambda w: w.start_time)
        return None


class ScheduleProvider:
    """
    Provides charging schedule information.

    This class:
    - Calculates optimal charging windows based on prices
    - Tracks current schedule state
    - Notifies consumers when schedule changes via callback
    - Adjusts schedule based on detected charging phases

    It does NOT know about FSM or charger control.
    Implements PhaseDetectionConsumer to receive phase detection results.
    """

    def __init__(self, price_fetcher: PriceFetcher) -> None:
        self._price_fetcher = price_fetcher
        self._consumers: list[ScheduleConsumer] = []
        self._scheduler = AsyncIOScheduler()
        self._current_plan: ChargingPlan | None = None
        self._running = False

        # Configuration
        config = get_config()
        self._target_kwh = config.charging.target_kwh
        self._charging_power_kw = config.charging.charging_power_kw
        self._night_start = config.grid_tariff.night_start_hour
        self._night_end = config.grid_tariff.night_end_hour

        # Phase detection results
        self._detected_phases: int | None = None
        self._detected_power_kw: float | None = None

        # Calculate hours needed (initial estimate, updated after phase detection)
        if self._charging_power_kw > 0:
            self._hours_needed = self._target_kwh / self._charging_power_kw
        else:
            self._hours_needed = 3

    @property
    def target_kwh(self) -> float:
        """Get current target kWh."""
        return self._target_kwh

    async def set_target_kwh(self, kwh: float) -> None:
        """Set target kWh and recalculate schedule."""
        if kwh < 1 or kwh > 100:
            raise ValueError("Target kWh must be between 1 and 100")

        self._target_kwh = kwh

        # Recalculate hours needed
        power = self._detected_power_kw or self._charging_power_kw
        if power > 0:
            self._hours_needed = self._target_kwh / power

        logger.info(f"Target kWh set to {kwh}, hours needed: {self._hours_needed:.1f}")

        # Recreate plan with new target
        await self._create_daily_plan()
        await self._evaluate_schedule()

    def add_consumer(self, consumer: ScheduleConsumer) -> None:
        """Add a schedule consumer (e.g., FSM)."""
        self._consumers.append(consumer)

    # Keep old method for backwards compatibility
    def set_fsm(self, fsm: ScheduleConsumer) -> None:
        """Add FSM as a schedule consumer."""
        self.add_consumer(fsm)

    # =========================================================================
    # PhaseDetectionConsumer interface
    # =========================================================================

    async def on_phases_detected(self, result: PhaseDetectionResult) -> None:
        """
        Called when phase detection completes.

        Updates charging power estimate and recalculates schedule.
        Sets default target kWh based on detected phases:
        - 1-phase: 10 kWh
        - 3-phase: 20 kWh
        """

        self._detected_phases = result.detected_phases
        self._detected_power_kw = result.detected_power_kw

        # Set default target based on detected phases (only on first detection)
        if self._detected_phases == 1:
            default_target = 10.0
        else:
            default_target = 20.0

        # Update target to phase-appropriate default
        self._target_kwh = default_target

        # Recalculate hours needed based on detected power
        if result.detected_power_kw > 0:
            old_hours = self._hours_needed
            self._hours_needed = self._target_kwh / result.detected_power_kw

            logger.info(
                f"Phase detection: {result.detected_phases} phase(s), "
                f"{result.detected_power_kw:.1f}kW. "
                f"Default target: {default_target}kWh. "
                f"Hours needed: {old_hours:.1f} -> {self._hours_needed:.1f}"
            )

            # Recreate plan with updated power estimate
            await self._create_daily_plan()
            await self._evaluate_schedule()

    async def start(self) -> None:
        """Start the schedule provider."""
        if self._running:
            return

        self._running = True

        # Schedule periodic tasks
        # Evaluate at the top of each hour when price slot changes
        self._scheduler.add_job(
            self._evaluate_schedule,
            CronTrigger(minute="0"),  # Only at hour boundaries
            id="evaluate_schedule",
            replace_existing=True,
        )

        # Refresh prices at 5 past each hour (gives APIs time to publish)
        self._scheduler.add_job(
            self._refresh_prices,
            CronTrigger(minute="5"),
            id="refresh_prices",
            replace_existing=True,
        )

        # Create daily plan at midnight and noon
        self._scheduler.add_job(
            self._create_daily_plan,
            CronTrigger(hour="0,12", minute="1"),
            id="create_daily_plan",
            replace_existing=True,
        )

        self._scheduler.start()

        # Create initial plan (also evaluates schedule)
        await self._create_daily_plan()

        logger.info("Schedule provider started")

    async def stop(self) -> None:
        """Stop the schedule provider."""
        self._running = False
        try:
            if self._scheduler.running:
                self._scheduler.shutdown(wait=False)
        except Exception:
            pass
        logger.info("Schedule provider stopped")

    async def _refresh_prices(self) -> None:
        """Refresh price data and re-evaluate schedule."""
        try:
            await self._price_fetcher.get_prices(force_refresh=True)
            logger.debug("Prices refreshed")
            # Re-create plan with new prices
            await self._create_daily_plan()
        except Exception as e:
            logger.warning(f"Failed to refresh prices: {e}")

    async def _create_daily_plan(self) -> None:
        """Create a charging plan for the day."""
        try:
            forecast = await self._price_fetcher.get_prices()

            # Get cheapest hours for charging (round up to ensure we have enough time)
            hours_needed = math.ceil(self._hours_needed)
            cheapest_hours = self._select_cheapest_hours(forecast, hours_needed)

            if not cheapest_hours:
                logger.warning("No hours available for charging plan")
                return

            # Create windows from selected hours
            windows = []

            for hour_price in cheapest_hours:
                window = ChargingWindow(
                    start_time=hour_price.start_time,
                    end_time=hour_price.end_time,
                    price_sek=hour_price.price,
                    is_night_rate=hour_price.is_cheap,
                )
                windows.append(window)

            # Calculate cost based on actual target kWh, not full hours
            # Use average price from selected windows
            if windows:
                avg_price = sum(w.price_sek for w in windows) / len(windows)
                total_cost = self._target_kwh * avg_price
            else:
                total_cost = 0.0

            self._current_plan = ChargingPlan(
                created_at=datetime.now(timezone.utc),
                windows=sorted(windows, key=lambda w: w.start_time),
                target_kwh=self._target_kwh,
                estimated_cost_sek=total_cost,
            )

            logger.info(
                f"Created charging plan: {len(windows)} windows, "
                f"est. cost: {total_cost:.2f} SEK"
            )

            # Evaluate and notify consumers of new plan
            await self._evaluate_schedule()

        except Exception as e:
            logger.error(f"Failed to create daily plan: {e}")

    def _select_cheapest_hours(
        self,
        forecast: PriceForecast,
        hours_needed: int,
    ) -> list[HourlyPrice]:
        """Select the cheapest hours for charging."""
        now = datetime.now(timezone.utc)

        # Filter to future hours only
        future_hours = [h for h in forecast.prices if h.end_time > now]

        if not future_hours:
            return []

        # Sort by total price (includes grid tariff)
        sorted_hours = sorted(future_hours, key=lambda h: h.price)

        # Select cheapest hours
        return sorted_hours[:hours_needed]

    async def _evaluate_schedule(self) -> None:
        """Evaluate current schedule and notify all consumers."""
        if not self._consumers:
            return

        from .charging_fsm import ScheduleInfo

        now = datetime.now(timezone.utc)

        # Check if in a charging window
        in_window = False
        current_window_end = None
        next_window_start = None
        is_cheap = False
        current_price = None

        if self._current_plan:
            current_window = self._current_plan.get_current_window(now)
            if current_window:
                in_window = True
                current_window_end = current_window.end_time
                is_cheap = current_window.is_night_rate
                current_price = current_window.price_sek

            next_window = self._current_plan.get_next_window(now)
            if next_window:
                next_window_start = next_window.start_time

        # Get current price from forecast
        try:
            forecast = await self._price_fetcher.get_prices()
            if forecast.current_price:
                current_price = forecast.current_price.price
                is_cheap = forecast.current_price.is_cheap
        except Exception:
            pass

        schedule_info = ScheduleInfo(
            is_in_window=in_window,
            current_window_end=current_window_end,
            next_window_start=next_window_start,
            is_cheap_hour=is_cheap,
            current_price_sek=current_price,
        )

        # Notify all consumers
        for consumer in self._consumers:
            try:
                await consumer.on_schedule_update(schedule_info)
            except Exception as e:
                logger.warning(f"Error notifying schedule consumer: {e}")

    def get_status(self) -> dict:
        """Get schedule status for API."""
        now = datetime.now(timezone.utc)

        status = {
            "has_plan": self._current_plan is not None,
            "plan_created_at": (
                self._current_plan.created_at.isoformat() if self._current_plan else None
            ),
            "target_kwh": self._target_kwh,
            "hours_needed": self._hours_needed,
        }

        if self._current_plan:
            status["windows"] = [
                {
                    "start": w.start_time.isoformat(),
                    "end": w.end_time.isoformat(),
                    "price_sek": w.price_sek,
                    "is_night_rate": w.is_night_rate,
                    "is_current": w.start_time <= now < w.end_time,
                    "is_past": w.end_time <= now,
                }
                for w in self._current_plan.windows
            ]
            status["estimated_cost_sek"] = self._current_plan.estimated_cost_sek
            status["is_in_window"] = self._current_plan.is_in_window(now)

        return status

    @property
    def current_plan(self) -> ChargingPlan | None:
        """Get current charging plan."""
        return self._current_plan


# Singleton instance
_provider_instance: ScheduleProvider | None = None


def get_schedule_provider() -> ScheduleProvider | None:
    """Get the global ScheduleProvider instance."""
    return _provider_instance


def set_schedule_provider(provider: ScheduleProvider) -> None:
    """Set the global ScheduleProvider instance."""
    global _provider_instance
    _provider_instance = provider
