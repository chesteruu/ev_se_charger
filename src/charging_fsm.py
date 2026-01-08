"""
Charging Finite State Machine (FSM).

This is the CENTRAL controller for all charging decisions.

Event Sources (FSM doesn't care WHO triggers events):
1. Charger state changes (from EaseeController monitoring)
2. Schedule updates (from ScheduleProvider)
3. Load updates (from LoadMonitor)
4. External requests (start/stop/resume/mode change)

The FSM:
- Receives events/requests
- Decides based on STATE + MODE whether to act
- Orchestrates actions through a ChargerController interface
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from .easee_controller import ChargerState, ChargerStatus

logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class PhaseDetectionResult:
    """Result of phase detection."""

    detected_phases: int  # 1 or 3
    detected_power_kw: float  # Estimated charging power
    max_current_per_phase: float  # Max current per phase


# =============================================================================
# INTERFACES (Protocols)
# =============================================================================


@runtime_checkable
class ChargerController(Protocol):
    """
    Interface for charger control.

    FSM doesn't care about implementation - just needs these methods.
    """

    async def do_charge(self, current: float | None = None) -> bool:
        """Start charging at given current."""
        ...

    async def do_pause(self) -> bool:
        """Pause charging."""
        ...

    async def do_resume(self) -> bool:
        """Resume charging."""
        ...

    async def do_stop(self) -> bool:
        """Stop charging completely."""
        ...

    async def do_set_current(self, current: float) -> bool:
        """Set charging current."""
        ...

    async def do_detect_phases(self) -> PhaseDetectionResult | None:
        """
        Perform phase detection by brief charging.

        Returns detected phases and power, or None if detection failed.
        """
        ...

    async def get_status(self) -> ChargerStatus:
        """Get current charger status."""
        ...


@runtime_checkable
class PhaseDetectionConsumer(Protocol):
    """
    Interface for components that need phase detection results.

    ChargerController notifies consumers when phases are detected.
    """

    async def on_phases_detected(self, result: PhaseDetectionResult) -> None:
        """Called when phase detection completes."""
        ...


class FSMState(Enum):
    """System FSM states."""

    IDLE = "idle"  # No car connected
    CAR_CONNECTED = "car_connected"  # Car just connected, evaluating
    WAITING_FOR_SCHEDULE = "waiting_for_schedule"  # Waiting for charging window
    CHARGING = "charging"  # Actively charging
    PAUSED_LOAD = "paused_load"  # Paused due to load protection
    PAUSED_SCHEDULE = "paused_schedule"  # Paused, outside schedule window
    PAUSED_USER = "paused_user"  # Paused by user request
    CHARGE_COMPLETE = "charge_complete"  # Charging finished
    ERROR = "error"  # Error state


class ChargingMode(Enum):
    """Charging modes that affect FSM behavior."""

    SMART = "smart"  # Price-aware, follows schedule
    IMMEDIATE = "immediate"  # Charge immediately when car connected
    MANUAL = "manual"  # Manual control only
    SOLAR = "solar"  # Solar-first charging
    SCHEDULED = "scheduled"  # Time-based scheduling


@dataclass
class ScheduleInfo:
    """Information about the current charging schedule."""

    is_in_window: bool = False
    current_window_end: datetime | None = None
    next_window_start: datetime | None = None
    is_cheap_hour: bool = False
    current_price_sek: float | None = None


@dataclass
class LoadInfo:
    """Information about the current load status."""

    home_current_amps: float = 0.0
    available_for_ev_amps: float = 0.0
    should_pause: bool = False
    recommended_current: float = 0.0
    # Per-phase available currents for optimal 3-phase charging
    available_l1_amps: float = 0.0
    available_l2_amps: float = 0.0
    available_l3_amps: float = 0.0


@dataclass
class FSMContext:
    """Context information for FSM decisions."""

    state: FSMState = FSMState.IDLE
    previous_state: FSMState = FSMState.IDLE
    mode: ChargingMode = ChargingMode.SMART

    # Charger info (from EaseeController)
    charger_state: ChargerState | None = None
    charger_power_watts: float = 0.0
    charger_current_amps: float = 0.0
    session_energy_kwh: float = 0.0

    # Schedule info (from ScheduleProvider)
    schedule: ScheduleInfo = field(default_factory=ScheduleInfo)

    # Load info (from LoadMonitor)
    load: LoadInfo = field(default_factory=LoadInfo)

    # Current limit set by FSM
    current_limit_amps: float = 6.0

    # Timestamps
    state_entered_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    car_connected_at: datetime | None = None
    charging_started_at: datetime | None = None


class ChargingFSM:
    """
    Finite State Machine for charging control.

    This is the ONLY component that controls the charger.
    External callers use request_* methods - FSM decides based on state + mode.
    """

    MIN_CHARGING_CURRENT = 6.0

    def __init__(self, charger: ChargerController) -> None:
        """
        Initialize FSM with a charger controller.

        Args:
            charger: Any object implementing ChargerController protocol
        """
        self._charger = charger
        self._context = FSMContext()
        self._state_callbacks: list[Callable] = []
        self._phase_detection_consumers: list[PhaseDetectionConsumer] = []
        self._lock = asyncio.Lock()
        self._running = False
        self._last_detection_result: PhaseDetectionResult | None = None

    def add_phase_detection_consumer(self, consumer: PhaseDetectionConsumer) -> None:
        """Add a consumer for phase detection results (e.g., ScheduleProvider)."""
        self._phase_detection_consumers.append(consumer)

    @property
    def state(self) -> FSMState:
        """Current FSM state."""
        return self._context.state

    @property
    def context(self) -> FSMContext:
        """Current FSM context."""
        return self._context

    @property
    def mode(self) -> ChargingMode:
        """Current charging mode."""
        return self._context.mode

    def set_mode(self, mode: ChargingMode) -> None:
        """Set charging mode."""
        old_mode = self._context.mode
        self._context.mode = mode
        logger.info(f"Mode changed: {old_mode.value} -> {mode.value}")

    async def start(self) -> None:
        """Start the FSM."""
        self._running = True

        # Get initial charger state
        try:
            status = await self._charger.get_status()
            await self._on_charger_state_change(None, status)
        except Exception as e:
            logger.error(f"Failed to get initial charger state: {e}")

        logger.info("Charging FSM started")

    async def stop(self) -> None:
        """Stop the FSM."""
        self._running = False
        logger.info("Charging FSM stopped")

    def on_state_change(self, callback: Callable) -> None:
        """Register callback for FSM state changes."""
        self._state_callbacks.append(callback)

    # =========================================================================
    # EXTERNAL TRIGGERS
    # =========================================================================

    async def on_charger_state_change(
        self,
        old_status: ChargerStatus | None,
        new_status: ChargerStatus,
    ) -> None:
        """Called by EaseeController when charger state changes."""
        logger.debug("FSM: on_charger_state_change - waiting for lock...")
        async with self._lock:
            logger.debug("FSM: on_charger_state_change - lock acquired")
            await self._on_charger_state_change(old_status, new_status)
            logger.debug("FSM: on_charger_state_change - done, releasing lock")

    async def on_schedule_update(self, schedule: ScheduleInfo) -> None:
        """Called by ScheduleProvider when schedule changes."""
        async with self._lock:
            old_in_window = self._context.schedule.is_in_window
            self._context.schedule = schedule

            if old_in_window != schedule.is_in_window:
                logger.info(f"Schedule: in_window={schedule.is_in_window}")
                await self._evaluate_and_act()

    async def on_load_update(self, load: LoadInfo) -> None:
        """Called by LoadMonitor when load changes significantly."""
        async with self._lock:
            old_should_pause = self._context.load.should_pause
            self._context.load = load

            # Update current if charging
            if self._context.state == FSMState.CHARGING:
                if load.should_pause and not old_should_pause:
                    logger.warning(f"Load protection: pausing (available={load.available_for_ev_amps:.1f}A)")
                    await self._do_pause("Load protection")
                    await self._transition_to(FSMState.PAUSED_LOAD, "Load protection")
                elif load.recommended_current != self._context.current_limit_amps:
                    # Pass per-phase currents for optimal 3-phase charging
                    await self._do_set_current(
                        load.recommended_current,
                        current_l1=load.available_l1_amps,
                        current_l2=load.available_l2_amps,
                        current_l3=load.available_l3_amps,
                    )

            # Resume from load pause if load is OK
            elif self._context.state == FSMState.PAUSED_LOAD and not load.should_pause:
                if self._should_charge_now():
                    logger.info(f"Load protection cleared: resuming (available={load.available_for_ev_amps:.1f}A)")
                    # Trigger grace period in case charger does a phase switch on resume
                    detected_phases = self._last_detection_result.detected_phases if self._last_detection_result else 1
                    if detected_phases == 1:
                        from .load_monitor import get_load_monitor
                        lm = get_load_monitor()
                        if lm:
                            lm.signal_charger_resumed()
                    await self._do_resume()
                    await self._transition_to(FSMState.CHARGING, "Load protection cleared")

    # =========================================================================
    # EXTERNAL REQUESTS - Generic methods, FSM decides based on state + mode
    # =========================================================================

    async def request_start(self) -> dict:
        """
        Request to start charging.

        Allows manual start in any mode (user override).
        """
        logger.info("FSM: request_start - waiting for lock...")
        try:
            async with asyncio.timeout(15.0):
                async with self._lock:
                    logger.info("FSM: request_start - lock acquired")
                    mode = self._context.mode
                    state = self._context.state

                    # Check preconditions
                    if state == FSMState.IDLE:
                        return {"success": False, "reason": "No car connected"}

                    if state == FSMState.CHARGING:
                        return {"success": False, "reason": "Already charging"}

                    if state == FSMState.CHARGE_COMPLETE:
                        return {"success": False, "reason": "Charging complete"}

                    # Allow manual start in any mode (user override)
                    await self._do_charge()
                    self._context.charging_started_at = datetime.now(timezone.utc)
                    await self._transition_to(FSMState.CHARGING, f"Manual start ({mode.value})")
                    return {"success": True, "reason": f"Started (manual override)"}
        except asyncio.TimeoutError:
            logger.error("FSM: request_start - timeout waiting for lock (another operation may be stuck)")
            return {"success": False, "reason": "System busy, try again"}

    async def request_stop(self) -> dict:
        """
        Request to stop/pause charging.

        Works from CHARGING or WAITING_FOR_SCHEDULE states.
        Transitions to PAUSED_USER to prevent schedule from auto-starting.
        """
        logger.info("FSM: request_stop - waiting for lock...")
        try:
            async with asyncio.timeout(15.0):
                async with self._lock:
                    logger.info("FSM: request_stop - lock acquired")
                    state = self._context.state

                    # Can stop from charging or waiting states
                    if state not in (FSMState.CHARGING, FSMState.WAITING_FOR_SCHEDULE):
                        return {"success": False, "reason": "Not in a stoppable state"}

                    # If charging, pause the charger
                    if state == FSMState.CHARGING:
                        await self._do_pause("User request")

                    # Transition to user-paused (prevents schedule from auto-starting)
                    await self._transition_to(FSMState.PAUSED_USER, "User request")
                    return {"success": True, "reason": "Paused by user"}
        except asyncio.TimeoutError:
            logger.error("FSM: request_stop - timeout waiting for lock")
            return {"success": False, "reason": "System busy, try again"}

    async def request_resume(self) -> dict:
        """
        Request to resume from user-paused state.

        In SMART mode: Returns to WAITING_FOR_SCHEDULE (schedule takes over)
        In MANUAL mode: Starts charging immediately
        """
        logger.info("FSM: request_resume - waiting for lock...")
        try:
            async with asyncio.timeout(15.0):
                async with self._lock:
                    logger.info("FSM: request_resume - lock acquired")
                    state = self._context.state
                    mode = self._context.mode

                    if state != FSMState.PAUSED_USER:
                        return {"success": False, "reason": "Not in user-paused state"}

                    if mode == ChargingMode.MANUAL:
                        # Manual mode: start charging immediately
                        await self._do_charge()
                        self._context.charging_started_at = datetime.now(timezone.utc)
                        await self._transition_to(FSMState.CHARGING, "Manual resume")
                        return {"success": True, "reason": "Resumed charging"}

                    # SMART mode: return to waiting for schedule
                    await self._transition_to(FSMState.WAITING_FOR_SCHEDULE, "User resumed - schedule takes over")
                    return {"success": True, "reason": "Resumed - schedule will control charging"}
        except asyncio.TimeoutError:
            logger.error("FSM: request_resume - timeout waiting for lock")
            return {"success": False, "reason": "System busy, try again"}

    # =========================================================================
    # INTERNAL FSM LOGIC
    # =========================================================================

    async def _on_charger_state_change(
        self,
        old_status: ChargerStatus | None,
        new_status: ChargerStatus,
    ) -> None:
        """Handle charger state change."""
        from .easee_controller import ChargerState

        old_charger_state = self._context.charger_state
        new_charger_state = new_status.state

        # Update context
        self._context.charger_state = new_charger_state
        self._context.charger_power_watts = new_status.power_watts
        self._context.charger_current_amps = new_status.current_amps
        self._context.session_energy_kwh = new_status.energy_session_kwh

        # Handle state transitions
        if new_charger_state == ChargerState.DISCONNECTED:
            await self._transition_to(FSMState.IDLE, "Car disconnected")
            self._context.car_connected_at = None
            self._context.charging_started_at = None

        elif new_charger_state == ChargerState.AWAITING_START:
            # Transition to CAR_CONNECTED if:
            # - Coming from DISCONNECTED state, OR
            # - Initial startup (old_charger_state is None) and FSM is IDLE
            if old_charger_state == ChargerState.DISCONNECTED or (
                old_charger_state is None and self._context.state == FSMState.IDLE
            ):
                self._context.car_connected_at = datetime.now(timezone.utc)
                await self._transition_to(FSMState.CAR_CONNECTED, "Car connected")
                # Entry action: detect phases
                await self._do_detect_phases()
                await self._evaluate_and_act()
            elif self._context.state == FSMState.CHARGING:
                # Charger briefly stopped but FSM is still in CHARGING state
                # This means we WANT to be charging (user started or schedule started)
                # Stay in CHARGING state - the charger will resume and we'll continue
                # FSM state represents INTENT, not charger's momentary state
                logger.info("FSM: Charger paused but staying in CHARGING state (intent unchanged)")

        elif new_charger_state == ChargerState.CHARGING:
            # Signal load monitor if coming from paused states (potential phase switch)
            # Charger can go: AWAITING_START -> CHARGING or READY_TO_CHARGE -> CHARGING
            # Only relevant for 1-phase charging - 3-phase doesn't switch phases
            if old_charger_state in (ChargerState.AWAITING_START, ChargerState.READY_TO_CHARGE):
                detected_phases = self._last_detection_result.detected_phases if self._last_detection_result else 1
                if detected_phases == 1:
                    from .load_monitor import get_load_monitor
                    load_monitor = get_load_monitor()
                    if load_monitor:
                        load_monitor.signal_charger_resumed()
                        logger.info("FSM: Signaled phase switch grace period (1-phase resumed)")
            
            if self._context.state != FSMState.CHARGING:
                # If we're already in a paused state, don't keep trying to pause
                # The charger just hasn't stopped yet - wait for it
                if self._context.state in (
                    FSMState.PAUSED_USER,
                    FSMState.PAUSED_SCHEDULE,
                    FSMState.PAUSED_LOAD,
                ):
                    logger.debug("FSM: Charger still charging but pause already issued, waiting...")
                    return

                # Check if we should actually be charging (schedule/mode)
                should_charge = self._should_charge_now()

                if should_charge:
                    self._context.charging_started_at = datetime.now(timezone.utc)
                    await self._transition_to(FSMState.CHARGING, "Charging started")
                else:
                    # Charger started externally but we're not in a charging window
                    # Stop it and keep the current paused state
                    logger.warning(
                        "FSM: Charger started externally but not in charging window - stopping"
                    )
                    await self._do_pause("Not in charging window")
                    # Preserve current paused state - don't change if already paused
                    if self._context.state not in (
                        FSMState.WAITING_FOR_SCHEDULE,
                        FSMState.PAUSED_SCHEDULE,
                        FSMState.PAUSED_USER,  # Keep user pause state
                    ):
                        await self._transition_to(
                            FSMState.PAUSED_SCHEDULE, "Outside schedule window"
                        )

        elif new_charger_state == ChargerState.COMPLETED:
            await self._transition_to(FSMState.CHARGE_COMPLETE, "Charging complete")

        elif new_charger_state == ChargerState.READY_TO_CHARGE:
            if self._context.state == FSMState.IDLE:
                self._context.car_connected_at = datetime.now(timezone.utc)
                await self._transition_to(FSMState.CAR_CONNECTED, "Car connected")
                # Entry action: detect phases
                await self._do_detect_phases()
                await self._evaluate_and_act()
            elif self._context.state == FSMState.CHARGING:
                # READY_TO_CHARGE can be transient during charger negotiation
                # Stay in CHARGING - our intent hasn't changed
                logger.info("FSM: Charger ready_to_charge but staying in CHARGING state (intent unchanged)")

        elif new_charger_state == ChargerState.ERROR:
            await self._transition_to(FSMState.ERROR, "Charger error")

    async def _evaluate_and_act(self) -> None:
        """Evaluate current state and take action if needed."""
        state = self._context.state

        # Terminal states - no action
        if state in (FSMState.IDLE, FSMState.CHARGE_COMPLETE, FSMState.ERROR):
            return

        should_charge = self._should_charge_now()
        should_pause_load = self._context.load.should_pause

        logger.debug(
            f"Evaluate: state={state.value}, should_charge={should_charge}, "
            f"load_pause={should_pause_load}, mode={self._context.mode.value}"
        )

        if state == FSMState.CAR_CONNECTED:
            if should_charge and not should_pause_load:
                await self._do_charge()
            else:
                await self._transition_to(FSMState.WAITING_FOR_SCHEDULE, "Waiting for schedule")

        elif state == FSMState.WAITING_FOR_SCHEDULE:
            if should_charge and not should_pause_load:
                await self._do_charge()

        elif state == FSMState.CHARGING:
            if should_pause_load:
                await self._do_pause("Load protection")
                await self._transition_to(FSMState.PAUSED_LOAD, "Load protection")
            elif not should_charge:
                await self._do_pause("Outside schedule")
                await self._transition_to(FSMState.PAUSED_SCHEDULE, "Schedule pause")

        elif state == FSMState.PAUSED_LOAD:
            if not should_pause_load and should_charge:
                logger.info("Load protection cleared - resuming charging")
                # Trigger grace period in case charger does a phase switch on resume
                detected_phases = self._last_detection_result.detected_phases if self._last_detection_result else 1
                if detected_phases == 1:
                    from .load_monitor import get_load_monitor
                    lm = get_load_monitor()
                    if lm:
                        lm.signal_charger_resumed()
                await self._do_resume()
                await self._transition_to(FSMState.CHARGING, "Load protection cleared")

        elif state == FSMState.PAUSED_SCHEDULE:
            if should_charge and not should_pause_load:
                logger.info("Schedule window opened - resuming charging")
                await self._do_resume()
                await self._transition_to(FSMState.CHARGING, "Schedule resumed")

        elif state == FSMState.PAUSED_USER:
            # User pause - only manual mode can auto-resume
            if self._context.mode == ChargingMode.MANUAL:
                pass  # Stay paused until GUI resumes

    def _should_charge_now(self) -> bool:
        """Determine if charging should be active."""
        mode = self._context.mode
        state = self._context.state

        if mode == ChargingMode.IMMEDIATE:
            return True

        if mode == ChargingMode.MANUAL:
            # In manual mode, respect current FSM state
            # Allow charging if state is CHARGING or transitioning to it
            # Don't auto-start from other states
            return state == FSMState.CHARGING

        if mode in (ChargingMode.SMART, ChargingMode.SCHEDULED):
            # In SMART mode, if user paused, don't auto-resume
            if state == FSMState.PAUSED_USER:
                return False
            return self._context.schedule.is_in_window

        if mode == ChargingMode.SOLAR:
            return self._context.schedule.is_in_window

        return False

    # =========================================================================
    # ACTIONS - All charger control goes through these
    # =========================================================================

    async def _do_charge(self) -> None:
        """Start charging at minimum current with detected phase information."""
        try:
            current = self.MIN_CHARGING_CURRENT
            # Get detected phases (default to 1 if unknown)
            phases = 1
            if self._last_detection_result:
                phases = self._last_detection_result.detected_phases
            
            success = await self._charger.do_charge(current, phases=phases)
            if success:
                self._context.current_limit_amps = current
                logger.info(f"FSM: Started charging at {current}A ({phases}-phase)")
        except Exception as e:
            logger.error(f"FSM: Failed to start charging: {e}")

    async def _do_pause(self, reason: str) -> None:
        """Pause charging."""
        try:
            await self._charger.do_pause()
            logger.info(f"FSM: Paused - {reason}")
        except Exception as e:
            logger.error(f"FSM: Failed to pause: {e}")

    async def _do_resume(self) -> None:
        """Resume charging."""
        try:
            await self._charger.do_resume()
            logger.info("FSM: Resumed charging")
        except Exception as e:
            logger.error(f"FSM: Failed to resume: {e}")

    async def _do_stop(self) -> None:
        """Stop charging completely."""
        try:
            await self._charger.do_stop()
            logger.info("FSM: Stopped charging")
        except Exception as e:
            logger.error(f"FSM: Failed to stop: {e}")

    async def _do_set_current(
        self, 
        current: float,
        current_l1: float | None = None,
        current_l2: float | None = None,
        current_l3: float | None = None,
    ) -> None:
        """Set charging current with detected phase information and per-phase limits."""
        try:
            current = max(self.MIN_CHARGING_CURRENT, current)
            if abs(current - self._context.current_limit_amps) >= 1.0:
                # Get detected phases and which phase EV is on
                phases = 1
                ev_phase = None  # Which phase(s) EV is actually using
                
                if self._last_detection_result:
                    phases = self._last_detection_result.detected_phases
                
                # Determine which phase EV is on from charger status
                charger_status = self._charger.last_status
                if charger_status and charger_status.is_charging:
                    l1 = charger_status.current_l1_amps or 0
                    l2 = charger_status.current_l2_amps or 0
                    l3 = charger_status.current_l3_amps or 0
                    # EV is on the phase with significant current (>1A)
                    if l1 > 1: ev_phase = "L1"
                    elif l2 > 1: ev_phase = "L2"
                    elif l3 > 1: ev_phase = "L3"
                
                # Set per-phase currents based on charging type
                max_current = self._charger.max_current
                
                if phases >= 3:
                    # 3-phase: set each phase to its available current
                    p1 = current_l1 if current_l1 is not None else current
                    p2 = current_l2 if current_l2 is not None else current
                    p3 = current_l3 if current_l3 is not None else current
                    logger.info(f"FSM: Setting 3-phase current: L1={p1:.0f}A, L2={p2:.0f}A, L3={p3:.0f}A")
                else:
                    # 1-phase: only limit the EV's phase, set others to max
                    if ev_phase == "L1" and current_l1 is not None:
                        p1 = current_l1
                        p2 = max_current
                        p3 = max_current
                    elif ev_phase == "L2" and current_l2 is not None:
                        p1 = max_current
                        p2 = current_l2
                        p3 = max_current
                    elif ev_phase == "L3" and current_l3 is not None:
                        p1 = max_current
                        p2 = max_current
                        p3 = current_l3
                    else:
                        # Unknown phase - set all to same (conservative)
                        p1 = p2 = p3 = current
                    logger.info(f"FSM: Setting 1-phase ({ev_phase}) current: L1={p1:.0f}A, L2={p2:.0f}A, L3={p3:.0f}A")
                
                success = await self._charger.do_set_current(
                    current, 
                    phases=phases,
                    current_l1=p1,
                    current_l2=p2,
                    current_l3=p3,
                )
                
                if success:
                    self._context.current_limit_amps = current
        except Exception as e:
            logger.error(f"FSM: Failed to set current: {e}")

    async def _do_detect_phases(self) -> None:
        """
        Perform phase detection and notify consumers.

        Entry action for CAR_CONNECTED state.
        """
        try:
            logger.info("FSM: Starting phase detection...")
            result = await self._charger.do_detect_phases()

            if result:
                self._last_detection_result = result
                logger.info(
                    f"FSM: Phase detection complete - "
                    f"{result.detected_phases} phase(s), "
                    f"{result.detected_power_kw:.1f}kW, "
                    f"{result.max_current_per_phase}A/phase"
                )

                # Notify all consumers (e.g., ScheduleProvider)
                for consumer in self._phase_detection_consumers:
                    try:
                        await consumer.on_phases_detected(result)
                    except Exception as e:
                        logger.warning(f"Error notifying phase detection consumer: {e}")
            else:
                logger.warning("FSM: Phase detection failed or not supported")

        except Exception as e:
            logger.error(f"FSM: Phase detection error: {e}")

    async def _transition_to(self, new_state: FSMState, reason: str) -> None:
        """Transition to a new FSM state."""
        old_state = self._context.state
        if old_state == new_state:
            return

        self._context.previous_state = old_state
        self._context.state = new_state
        self._context.state_entered_at = datetime.now(timezone.utc)

        logger.info(f"FSM: {old_state.value} -> {new_state.value} ({reason})")

        # Notify callbacks
        for callback in self._state_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(old_state, new_state, reason)
                else:
                    callback(old_state, new_state, reason)
            except Exception as e:
                logger.warning(f"FSM callback error: {e}")

    # =========================================================================
    # STATUS
    # =========================================================================

    def get_status(self) -> dict:
        """Get FSM status for API."""
        ctx = self._context
        return {
            "state": ctx.state.value,
            "previous_state": ctx.previous_state.value,
            "mode": ctx.mode.value,
            "charger_state": ctx.charger_state.name if ctx.charger_state else None,
            "charger_power_watts": ctx.charger_power_watts,
            "charger_current_amps": ctx.charger_current_amps,
            "current_limit_amps": ctx.current_limit_amps,
            "session_energy_kwh": ctx.session_energy_kwh,
            "schedule": {
                "is_in_window": ctx.schedule.is_in_window,
                "current_window_end": (
                    ctx.schedule.current_window_end.isoformat()
                    if ctx.schedule.current_window_end
                    else None
                ),
                "next_window_start": (
                    ctx.schedule.next_window_start.isoformat()
                    if ctx.schedule.next_window_start
                    else None
                ),
                "is_cheap_hour": ctx.schedule.is_cheap_hour,
                "current_price_sek": ctx.schedule.current_price_sek,
            },
            "load": {
                "home_current_amps": ctx.load.home_current_amps,
                "available_for_ev_amps": ctx.load.available_for_ev_amps,
                "should_pause": ctx.load.should_pause,
                "recommended_current": ctx.load.recommended_current,
            },
            "timestamps": {
                "state_entered_at": ctx.state_entered_at.isoformat(),
                "car_connected_at": (
                    ctx.car_connected_at.isoformat() if ctx.car_connected_at else None
                ),
                "charging_started_at": (
                    ctx.charging_started_at.isoformat() if ctx.charging_started_at else None
                ),
            },
            "phase_detection": {
                "detected_phases": (
                    self._last_detection_result.detected_phases
                    if self._last_detection_result else None
                ),
                "detected_power_kw": (
                    self._last_detection_result.detected_power_kw
                    if self._last_detection_result else None
                ),
                "max_current_per_phase": (
                    self._last_detection_result.max_current_per_phase
                    if self._last_detection_result else None
                ),
            },
        }


# Singleton
_fsm_instance: ChargingFSM | None = None


def get_charging_fsm() -> ChargingFSM | None:
    """Get the global ChargingFSM instance."""
    return _fsm_instance


def set_charging_fsm(fsm: ChargingFSM) -> None:
    """Set the global ChargingFSM instance."""
    global _fsm_instance
    _fsm_instance = fsm
