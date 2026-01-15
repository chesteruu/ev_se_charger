"""
Load Monitor Module.

This module monitors home electrical load and controls EV charging current.

Architecture:
- Receives power readings from SaveEye monitor
- Calculates home load and available EV capacity
- DIRECTLY sets charging current via CurrentController
- Notifies FSM only about should_pause signal (not current values)

Separation of concerns:
- LoadMonitor: Owns current calculation and adjustment
- FSM: Owns charging lifecycle (start/stop/pause)
- ChargerController: Executes commands
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from .config import get_config

if TYPE_CHECKING:
    from .charging_fsm import LoadInfo
    from .charging_fsm_base import ChargerStatus, CurrentController
    from .saveeye_monitor import PowerReading
    from .schedule_provider import CurrentChangeConsumer


# =============================================================================
# LOAD CONSUMER INTERFACE (Protocol)
# =============================================================================


@runtime_checkable
class LoadConsumer(Protocol):
    """
    Interface for load consumers (FSM).

    LoadMonitor notifies FSM only about should_pause signal.
    Current adjustment is handled internally by LoadMonitor.
    """

    async def on_load_update(self, load: LoadInfo) -> None:
        """Called when should_pause status changes."""
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
    Monitors home load and controls EV charging current.

    Responsibilities:
    - Receives power readings from SaveEye
    - Calculates home load (excluding EV)
    - Calculates available capacity for EV per phase
    - DIRECTLY sets charging current via CurrentController
    - Notifies FSM about should_pause signal
    - Notifies ScheduleProvider about current changes (affects charging plan)
    """

    MIN_CHARGING_CURRENT = 6.0  # Minimum charging current (A)
    PHASE_SWITCH_GRACE_SECONDS = 15.0  # Grace period after phase switch
    CURRENT_CHANGE_THRESHOLD = 2.0  # Only notify scheduler if current changed by this much (A)
    CURRENT_SET_COOLDOWN_SECONDS = 15.0  # Minimum time between current adjustments
    CURRENT_DEAD_BAND = 4.0  # Dead band around current setting - ignore changes smaller than this

    def __init__(self) -> None:
        config = get_config()

        self._fuse_limit = config.home.main_fuse_amps
        self._safety_margin = config.home.safety_margin_amps
        self._effective_limit = self._fuse_limit - self._safety_margin
        self._max_ev_current = config.easee.max_current

        self._consumers: list[LoadConsumer] = []
        self._current_controller: CurrentController | None = None
        self._current_change_consumer: CurrentChangeConsumer | None = None
        self._last_status: LoadStatus | None = None
        self._last_set_current: float = 0.0  # Track what we last set
        self._last_set_time: datetime | None = None  # Track when we last set current
        self._last_notified_current: float = 0.0  # Track what we last notified to scheduler
        self._running = False

        # Smoothing
        self._reading_history: list[float] = []
        self._history_size = 5
        
        # Phase switch tracking
        self._last_ev_phase: str | None = None
        self._phase_switch_time: datetime | None = None
        
        # Detected phases (updated by FSM after phase detection)
        self._detected_phases: int = 1

    def set_current_controller(self, controller: CurrentController) -> None:
        """Set the current controller for direct current adjustment."""
        self._current_controller = controller

    def set_current_change_consumer(self, consumer: CurrentChangeConsumer) -> None:
        """Set the consumer for current change notifications (e.g., ScheduleProvider)."""
        self._current_change_consumer = consumer

    def set_detected_phases(self, phases: int) -> None:
        """Update the detected number of phases (called by FSM after detection)."""
        self._detected_phases = phases
        logger.info(f"LoadMonitor: Detected phases updated to {phases}")

    def add_consumer(self, consumer: LoadConsumer) -> None:
        """Add a load consumer (e.g., FSM) for should_pause notifications."""
        self._consumers.append(consumer)

    # Keep old method for backwards compatibility
    def set_fsm(self, fsm: LoadConsumer) -> None:
        """Add FSM as a load consumer."""
        self.add_consumer(fsm)

    def signal_charger_resumed(self) -> None:
        """
        Signal that the charger just resumed from pause/awaiting state.
        
        This triggers a grace period to avoid false load protection
        due to stale SaveEye readings during phase switches.
        Also resets current tracking so we don't carry over stale values.
        """
        logger.info("Charger resumed - starting phase switch grace period")
        self._phase_switch_time = datetime.now(timezone.utc)
        # Reset current tracking so we can set fresh values after grace period
        self._last_set_current = 0.0
        self._last_set_time = None

    def is_in_grace_period(self) -> bool:
        """
        Check if we're in the phase switch grace period.
        
        During this period, charger state may be unstable and FSM
        should not enforce its intent aggressively.
        """
        if self._phase_switch_time is None:
            return False
        elapsed = (datetime.now(timezone.utc) - self._phase_switch_time).total_seconds()
        return elapsed < self.PHASE_SWITCH_GRACE_SECONDS

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

        # Get EV current from charger status
        # Use outputCurrent as the actual EV current (not circuitTotalPhaseConductorCurrentLX
        # which includes all loads on the circuit, not just EV)
        if charger_status and charger_status.is_charging and charger_status.power_watts > 100:
            # Use charger's reported output current as the actual EV current
            ev_current = charger_status.current_amps or 0.0
            
            # Use per-phase circuit currents only to DETECT which phase is active
            # (these values may include other circuit loads, so don't use them as EV current)
            circuit_l1 = charger_status.current_l1_amps or 0.0
            circuit_l2 = charger_status.current_l2_amps or 0.0
            circuit_l3 = charger_status.current_l3_amps or 0.0
            
            # Determine which phase(s) are active based on relative values
            # The phase(s) with significant current are the active ones
            max_circuit = max(circuit_l1, circuit_l2, circuit_l3)
            
            # Count active phases (within 50% of max = active for 3-phase)
            threshold = max_circuit * 0.5 if max_circuit > 1 else 1.0
            active_phases = sum([1 for x in [circuit_l1, circuit_l2, circuit_l3] if x > threshold])
            
            # Distribute EV current to the active phase(s)
            if active_phases >= 3:
                # 3-phase charging: distribute evenly
                ev_per_phase = ev_current / 3
                ev_l1 = ev_per_phase
                ev_l2 = ev_per_phase
                ev_l3 = ev_per_phase
                current_ev_phase = None
                self._last_ev_phase = None
            else:
                # 1-phase charging: put all current on the active phase
                ev_l1 = ev_l2 = ev_l3 = 0.0
                current_ev_phase = None
                
                # Find which phase is active (highest circuit current)
                if circuit_l1 >= circuit_l2 and circuit_l1 >= circuit_l3 and circuit_l1 > 1:
                    ev_l1 = ev_current
                    current_ev_phase = "L1"
                elif circuit_l2 >= circuit_l1 and circuit_l2 >= circuit_l3 and circuit_l2 > 1:
                    ev_l2 = ev_current
                    current_ev_phase = "L2"
                elif circuit_l3 > 1:
                    ev_l3 = ev_current
                    current_ev_phase = "L3"
                else:
                    # Fallback: use highest
                    if circuit_l1 >= circuit_l2 and circuit_l1 >= circuit_l3:
                        ev_l1 = ev_current
                        current_ev_phase = "L1"
                    elif circuit_l2 >= circuit_l3:
                        ev_l2 = ev_current
                        current_ev_phase = "L2"
                    else:
                        ev_l3 = ev_current
                        current_ev_phase = "L3"
                
                # Detect phase switch (only possible in 1-phase mode)
                if self._last_ev_phase and current_ev_phase and self._last_ev_phase != current_ev_phase:
                    logger.info(f"1-phase switch detected: {self._last_ev_phase} -> {current_ev_phase}")
                    self._phase_switch_time = datetime.now(timezone.utc)
                
                self._last_ev_phase = current_ev_phase
            
            logger.debug(
                f"EV current: {ev_current:.1f}A on phase {current_ev_phase or 'ALL'}, "
                f"circuit readings: L1={circuit_l1:.1f}A, L2={circuit_l2:.1f}A, L3={circuit_l3:.1f}A"
            )
            
            is_charging = True
        else:
            ev_l1 = ev_l2 = ev_l3 = 0.0
            ev_current = 0.0
            is_charging = False
            # Reset phase tracking when not charging
            self._last_ev_phase = None

        # Calculate home load by subtracting EV from detected phase(s)
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
        # For 1-phase: only the EV's active phase matters
        # For 3-phase: minimum available across all phases
        ev_phases_active = sum([1 for x in [ev_l1, ev_l2, ev_l3] if x > 0.5])
        
        if ev_phases_active == 0:
            # Not charging - use minimum available (conservative, we don't know which phase will be used)
            available_for_ev = min(avail_l1, avail_l2, avail_l3)
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
        
        # Grace period after phase switch - don't trigger load protection
        # SaveEye readings may be stale, causing false positives
        if should_pause and self._phase_switch_time:
            elapsed = (datetime.now(timezone.utc) - self._phase_switch_time).total_seconds()
            if elapsed < self.PHASE_SWITCH_GRACE_SECONDS:
                logger.info(
                    f"Phase switch grace period: {elapsed:.1f}s < {self.PHASE_SWITCH_GRACE_SECONDS}s, "
                    f"ignoring low available ({available_for_ev:.1f}A)"
                )
                should_pause = False
                # Use minimum charging current during grace period
                available_for_ev = max(available_for_ev, self.MIN_CHARGING_CURRENT)

        # Calculate per-phase currents to set (capped at max)
        set_l1 = min(avail_l1, self._max_ev_current)
        set_l2 = min(avail_l2, self._max_ev_current)
        set_l3 = min(avail_l3, self._max_ev_current)
        
        # Recommended current (minimum of available phases for display)
        recommended = min(set_l1, set_l2, set_l3)

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

        # Check for significant change
        old_should_pause = self._last_status.should_pause if self._last_status else False
        old_recommended = self._last_status.recommended_current if self._last_status else 0

        self._last_status = status

        # =====================================================================
        # DIRECTLY SET CURRENT if charging and current changed significantly
        # =====================================================================
        if is_charging and self._current_controller and not should_pause:
            # Check cooldown - don't set current too frequently
            now = datetime.now(timezone.utc)
            in_cooldown = (
                self._last_set_time is not None and
                (now - self._last_set_time).total_seconds() < self.CURRENT_SET_COOLDOWN_SECONDS
            )
            
            # Dead band: only change if recommended is outside [last_set - deadband, last_set + deadband]
            # This prevents oscillation from small fluctuations (e.g., 13A <-> 16A)
            change = abs(recommended - self._last_set_current)
            is_first_set = self._last_set_current == 0.0
            outside_dead_band = change >= self.CURRENT_DEAD_BAND
            
            # Only set if:
            # 1. First time OR outside dead band
            # 2. Either cooldown has passed OR it's an emergency decrease >= 6A
            is_emergency = recommended < self._last_set_current - 6.0
            should_set_current = (is_first_set or outside_dead_band) and (
                not in_cooldown or is_emergency
            )
            
            if should_set_current:
                logger.debug(
                    f"Current adjustment: {self._last_set_current:.0f}A -> {recommended:.0f}A "
                    f"(change={change:.0f}A, deadband={self.CURRENT_DEAD_BAND}A, first={is_first_set})"
                )
                try:
                    if self._detected_phases >= 3:
                        # 3-phase: set each phase to its available current
                        logger.info(
                            f"LoadMonitor: Setting 3-phase current: "
                            f"L1={set_l1:.1f}A, L2={set_l2:.1f}A, L3={set_l3:.1f}A"
                        )
                        await self._current_controller.do_set_current(
                            current=recommended,
                            phases=3,
                            current_l1=set_l1,
                            current_l2=set_l2,
                            current_l3=set_l3,
                        )
                    else:
                        # 1-phase: set single current (charger uses active phase)
                        logger.info(f"LoadMonitor: Setting 1-phase current: {recommended:.1f}A")
                        await self._current_controller.do_set_current(
                            current=recommended,
                            phases=1,
                        )
                    self._last_set_current = recommended
                    self._last_set_time = now
                except Exception as e:
                    logger.error(f"LoadMonitor: Failed to set current: {e}")
            elif change >= 2.0:
                # Log when we skip a noticeable change (inside dead band)
                logger.debug(
                    f"Skipping current change {self._last_set_current:.0f}A -> {recommended:.0f}A "
                    f"(change={change:.0f}A < deadband={self.CURRENT_DEAD_BAND}A)"
                )

        # =====================================================================
        # NOTIFY SCHEDULER if current changed significantly (affects charging plan)
        # =====================================================================
        if self._current_change_consumer and is_charging:
            if abs(recommended - self._last_notified_current) >= self.CURRENT_CHANGE_THRESHOLD:
                try:
                    await self._current_change_consumer.on_current_change(
                        available_current=recommended,
                        phases=self._detected_phases,
                    )
                    self._last_notified_current = recommended
                except Exception as e:
                    logger.warning(f"Error notifying current change consumer: {e}")

        # =====================================================================
        # NOTIFY FSM only about should_pause changes
        # =====================================================================
        if self._consumers and should_pause != old_should_pause:
            from .charging_fsm import LoadInfo

            load_info = LoadInfo(
                should_pause=should_pause,
                home_current_amps=home_current,
                available_for_ev_amps=available_for_ev,
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
