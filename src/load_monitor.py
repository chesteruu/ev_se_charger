"""
Load Monitor Module.

This module monitors home electrical load and calculates available capacity for EV charging.
It does NOT control charging directly - it provides load information via callback.

Architecture:
- Receives power readings from SaveEye monitor
- Calculates home load and available EV capacity
- Notifies consumers of load changes
- Does NOT know about FSM or charger control
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from .config import get_config

if TYPE_CHECKING:
    from .charging_fsm import LoadInfo
    from .easee_controller import ChargerStatus
    from .saveeye_monitor import PowerReading


# =============================================================================
# LOAD CONSUMER INTERFACE (Protocol)
# =============================================================================


@runtime_checkable
class LoadConsumer(Protocol):
    """
    Interface for load consumers.

    LoadMonitor doesn't care who implements this - just calls it.
    """

    async def on_load_update(self, load: LoadInfo) -> None:
        """Called when load changes significantly."""
        ...

logger = logging.getLogger(__name__)


@dataclass
class LoadStatus:
    """Current load status."""

    timestamp: datetime
    home_current_amps: float
    ev_current_amps: float
    total_current_amps: float
    current_l1_amps: float  # Total (including EV)
    current_l2_amps: float
    current_l3_amps: float
    home_l1_amps: float = 0.0  # Home only (EV excluded)
    home_l2_amps: float = 0.0
    home_l3_amps: float = 0.0
    limiting_phase: str = ""
    available_for_ev_amps: float = 0.0
    utilization_percent: float = 0.0
    should_pause: bool = False
    recommended_current: float = 0.0


class LoadMonitor:
    """
    Monitors home load and calculates EV charging capacity.

    This class:
    - Receives power readings from SaveEye
    - Calculates home load (excluding EV)
    - Calculates available capacity for EV
    - Notifies consumers of load changes via callback

    It does NOT know about FSM or charger control.
    """

    MIN_CHARGING_CURRENT = 6.0  # Minimum charging current (A)

    def __init__(self) -> None:
        config = get_config()

        self._fuse_limit = config.home.main_fuse_amps
        self._safety_margin = config.home.safety_margin_amps
        self._effective_limit = self._fuse_limit - self._safety_margin
        self._max_ev_current = config.easee.max_current

        self._consumers: list[LoadConsumer] = []
        self._last_status: LoadStatus | None = None
        self._current_ev_limit: float = self._max_ev_current
        self._running = False

        # Smoothing
        self._reading_history: list[float] = []
        self._history_size = 5

    def add_consumer(self, consumer: LoadConsumer) -> None:
        """Add a load consumer (e.g., FSM)."""
        self._consumers.append(consumer)

    # Keep old method for backwards compatibility
    def set_fsm(self, fsm: LoadConsumer) -> None:
        """Add FSM as a load consumer."""
        self.add_consumer(fsm)

    def set_ev_limit(self, current: float) -> None:
        """Update the current EV charging limit (for calculations)."""
        self._current_ev_limit = current

    async def on_power_reading(
        self,
        reading: PowerReading,
        charger_status: ChargerStatus | None = None,
    ) -> None:
        """
        Process a power reading and update FSM.

        Args:
            reading: Power reading from SaveEye
            charger_status: Current charger status (for EV current subtraction)
        """
        if not self._running:
            return

        # Get phase currents
        current_l1 = reading.current_l1
        current_l2 = reading.current_l2
        current_l3 = reading.current_l3

        # Get max phase current
        max_phase_current = max(current_l1, current_l2, current_l3)

        # Determine limiting phase
        if max_phase_current == current_l1:
            limiting_phase = "L1"
        elif max_phase_current == current_l2:
            limiting_phase = "L2"
        else:
            limiting_phase = "L3"

        # Get EV per-phase currents from charger status
        # This tells us EXACTLY how much current EV is drawing on each phase
        if charger_status and charger_status.is_charging:
            ev_l1 = charger_status.current_l1_amps
            ev_l2 = charger_status.current_l2_amps
            ev_l3 = charger_status.current_l3_amps
            ev_current = ev_l1 + ev_l2 + ev_l3  # Total for display
            is_charging = True
        else:
            ev_l1 = ev_l2 = ev_l3 = 0.0
            ev_current = 0.0
            is_charging = False

        # Calculate home load by subtracting EV from EACH phase
        # SaveEye measures total home including EV, so we need to subtract EV per-phase
        home_l1 = max(0, current_l1 - ev_l1)
        home_l2 = max(0, current_l2 - ev_l2)
        home_l3 = max(0, current_l3 - ev_l3)
        
        # Home current is the max phase (for display/general info)
        home_current = max(home_l1, home_l2, home_l3)
        
        # Debug log for verification
        if is_charging:
            logger.debug(
                f"Load calc: SaveEye[L1={current_l1:.1f}A, L2={current_l2:.1f}A, L3={current_l3:.1f}A] "
                f"- EV[L1={ev_l1:.1f}A, L2={ev_l2:.1f}A, L3={ev_l3:.1f}A] "
                f"= Home[L1={home_l1:.1f}A, L2={home_l2:.1f}A, L3={home_l3:.1f}A]"
            )

        # Calculate utilization
        utilization = max_phase_current / self._fuse_limit

        # Calculate per-phase available currents (always needed for optimal 3-phase)
        avail_l1 = max(0, self._effective_limit - home_l1)
        avail_l2 = max(0, self._effective_limit - home_l2)
        avail_l3 = max(0, self._effective_limit - home_l3)
        
        if is_charging:
            logger.debug(
                f"Available: limit={self._effective_limit}A - Home => "
                f"Avail[L1={avail_l1:.1f}A, L2={avail_l2:.1f}A, L3={avail_l3:.1f}A]"
            )

        # Determine which phase(s) EV is using and calculate available accordingly
        # For 1-phase: only the EV's phase matters
        # For 3-phase: use per-phase limits (charger can set different current per phase)
        ev_phases_active = sum([1 for x in [ev_l1, ev_l2, ev_l3] if x > 0.5])
        
        if ev_phases_active == 0:
            # Not charging or unknown - use max home load (conservative)
            available_for_ev = max(0, self._effective_limit - home_current)
        elif ev_phases_active == 1:
            # 1-phase charging: calculate available on EV's specific phase
            if ev_l1 > 0.5:
                available_for_ev = avail_l1
            elif ev_l2 > 0.5:
                available_for_ev = avail_l2
            else:  # ev_l3 > 0.5
                available_for_ev = avail_l3
        else:
            # 3-phase charging: for recommended_current, use minimum (safe default)
            # But we also pass per-phase limits for optimal control
            available_for_ev = min(avail_l1, avail_l2, avail_l3)

        # Should pause? Only if ALL phases below minimum (for 3-phase)
        # or EV's phase below minimum (for 1-phase)
        should_pause = available_for_ev < self.MIN_CHARGING_CURRENT

        # Recommended current (minimum of phases for safety)
        recommended = min(available_for_ev, self._max_ev_current)

        # Create status
        status = LoadStatus(
            timestamp=datetime.now(timezone.utc),
            home_current_amps=home_current,
            ev_current_amps=ev_current,
            total_current_amps=max_phase_current,
            current_l1_amps=current_l1,
            current_l2_amps=current_l2,
            current_l3_amps=current_l3,
            home_l1_amps=home_l1,
            home_l2_amps=home_l2,
            home_l3_amps=home_l3,
            limiting_phase=limiting_phase,
            available_for_ev_amps=available_for_ev,
            utilization_percent=utilization * 100,
            should_pause=should_pause,
            recommended_current=recommended,
        )

        # Check for significant change before notifying
        old_should_pause = self._last_status.should_pause if self._last_status else False
        old_recommended = self._last_status.recommended_current if self._last_status else 0

        self._last_status = status

        # Notify consumers if significant change
        if self._consumers:
            if should_pause != old_should_pause or abs(recommended - old_recommended) >= 1.0:
                from .charging_fsm import LoadInfo

                load_info = LoadInfo(
                    home_current_amps=home_current,
                    available_for_ev_amps=available_for_ev,
                    should_pause=should_pause,
                    recommended_current=recommended,
                    available_l1_amps=min(avail_l1, self._max_ev_current),
                    available_l2_amps=min(avail_l2, self._max_ev_current),
                    available_l3_amps=min(avail_l3, self._max_ev_current),
                )
                for consumer in self._consumers:
                    try:
                        await consumer.on_load_update(load_info)
                    except Exception as e:
                        logger.warning(f"Error notifying load consumer: {e}")

    def start(self) -> None:
        """Start the load monitor."""
        self._running = True
        logger.info("Load monitor started")

    def stop(self) -> None:
        """Stop the load monitor."""
        self._running = False
        logger.info("Load monitor stopped")

    @property
    def last_status(self) -> LoadStatus | None:
        """Get last calculated load status."""
        return self._last_status

    def get_status(self) -> dict:
        """Get load status for API."""
        if not self._last_status:
            return {
                "available": False,
                "message": "No load data available",
            }

        s = self._last_status
        return {
            "available": True,
            "timestamp": s.timestamp.isoformat(),
            "home_current_amps": s.home_current_amps,
            "ev_current_amps": s.ev_current_amps,
            "total_current_amps": s.total_current_amps,
            # Home-only per-phase (EV excluded) - use this for display
            "phases": {
                "l1": s.home_l1_amps,
                "l2": s.home_l2_amps,
                "l3": s.home_l3_amps,
            },
            # Total per-phase (including EV) - for reference
            "phases_total": {
                "l1": s.current_l1_amps,
                "l2": s.current_l2_amps,
                "l3": s.current_l3_amps,
            },
            "limiting_phase": s.limiting_phase,
            "available_for_ev_amps": s.available_for_ev_amps,
            "utilization_percent": s.utilization_percent,
            "should_pause": s.should_pause,
            "recommended_current": s.recommended_current,
            "fuse_limit": self._fuse_limit,
            "safety_margin": self._safety_margin,
            "effective_limit": self._effective_limit,
        }


# Singleton instance
_monitor_instance: LoadMonitor | None = None


def get_load_monitor() -> LoadMonitor | None:
    """Get the global LoadMonitor instance."""
    return _monitor_instance


def set_load_monitor(monitor: LoadMonitor) -> None:
    """Set the global LoadMonitor instance."""
    global _monitor_instance
    _monitor_instance = monitor
