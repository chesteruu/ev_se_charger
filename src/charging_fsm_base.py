"""
Base classes and shared types for Charging FSMs.

This module defines the INTERFACE CONTRACT between FSM and ChargerController:
- ChargerState: Abstract charger states (vendor-agnostic)
- ChargerStatus: Charger status data (vendor-agnostic)
- ChargerController: Protocol for charger control
- Common data classes (ScheduleInfo, LoadInfo, FSMContext)
- Common enums (FSMState, ChargingMode)
- BaseChargingFSM: Base class for FSM implementations
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Protocol, runtime_checkable

import logging

logger = logging.getLogger(__name__)


# =============================================================================
# CHARGER INTERFACE CONTRACT
# =============================================================================


class ChargerState(Enum):
    """
    Abstract charger operational states.
    
    This is the INTERFACE CONTRACT - vendor-agnostic charger states.
    Concrete controllers (EaseeController, etc.) map their vendor-specific
    states to these abstract states.
    
    FSM should use helper properties (is_ready, is_charging, etc.)
    rather than checking specific enum values.
    """

    DISCONNECTED = 1  # No car connected
    READY = 2         # Car connected, ready to charge
    CHARGING = 3      # Actively charging
    COMPLETED = 4     # Charging completed
    ERROR = 5         # Error state

    @property
    def is_car_connected(self) -> bool:
        """Check if a car is connected."""
        return self not in (ChargerState.DISCONNECTED, ChargerState.ERROR)

    @property
    def is_ready(self) -> bool:
        """Check if ready to charge (car connected, not charging)."""
        return self == ChargerState.READY

    @property
    def is_charging(self) -> bool:
        """Check if actively charging."""
        return self == ChargerState.CHARGING

    @property
    def is_completed(self) -> bool:
        """Check if charging is complete."""
        return self == ChargerState.COMPLETED

    @property
    def is_error(self) -> bool:
        """Check if in error state."""
        return self == ChargerState.ERROR


@dataclass
class ChargerStatus:
    """
    Charger status data - INTERFACE CONTRACT.
    
    Vendor-agnostic charger status. Concrete controllers populate this
    from their vendor-specific API responses.
    """

    state: ChargerState
    is_online: bool
    power_watts: float
    current_amps: float
    voltage_volts: float
    energy_session_kwh: float
    energy_total_kwh: float
    temperature_celsius: float
    max_current_amps: float
    dynamic_current_amps: float
    cable_locked: bool
    timestamp: datetime
    # Per-phase output currents from charger (for accurate load calculation)
    current_l1_amps: float = 0.0
    current_l2_amps: float = 0.0
    current_l3_amps: float = 0.0
    # Optional vendor-specific data (for debugging)
    raw_data: dict[str, Any] | None = None

    @property
    def is_charging(self) -> bool:
        """Check if actively charging."""
        return self.state.is_charging

    @property
    def is_car_connected(self) -> bool:
        """Check if car is connected."""
        return self.state.is_car_connected

    @property
    def power_kw(self) -> float:
        """Get power in kW."""
        return self.power_watts / 1000


@dataclass
class PhaseDetectionResult:
    """Result of phase detection."""

    detected_phases: int  # 1 or 3
    detected_power_kw: float  # Estimated charging power
    max_current_per_phase: float  # Max current per phase


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
    """
    Information about the current load status.
    
    This is a simplified signal from LoadMonitor to FSM.
    LoadMonitor handles current calculations and sets current directly.
    FSM only needs to know if it should pause.
    """

    should_pause: bool = False
    # For informational purposes only (API display)
    home_current_amps: float = 0.0
    available_for_ev_amps: float = 0.0


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

    # Load info (from LoadMonitor) - FSM only cares about pause signal
    load: LoadInfo = field(default_factory=LoadInfo)

    # Timestamps
    state_entered_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    car_connected_at: datetime | None = None
    charging_started_at: datetime | None = None


# =============================================================================
# PROTOCOLS - INTERFACE CONTRACTS
# =============================================================================


@runtime_checkable
class CurrentController(Protocol):
    """
    Interface for current control - used by LoadMonitor.
    
    This is a focused interface for setting charging current.
    LoadMonitor owns current calculations and sets current directly.
    """

    @property
    def last_status(self) -> ChargerStatus | None:
        """Get last known charger status."""
        ...

    @property
    def max_current(self) -> float:
        """Get configured max current."""
        ...

    async def do_set_current(
        self,
        current: float,
        phases: int = 1,
        current_l1: float | None = None,
        current_l2: float | None = None,
        current_l3: float | None = None,
    ) -> bool:
        """
        Set charging current per phase.
        
        Args:
            current: Default/minimum current
            phases: Number of phases (1 or 3)
            current_l1: Current limit for L1 (optional, per-phase control)
            current_l2: Current limit for L2 (optional, per-phase control)
            current_l3: Current limit for L3 (optional, per-phase control)
        """
        ...


@runtime_checkable
class ChargerController(Protocol):
    """
    Interface for charger control - used by FSM.

    FSM controls charging lifecycle (start/stop/pause/resume).
    Current adjustment is handled by LoadMonitor via CurrentController.
    """

    @property
    def last_status(self) -> ChargerStatus | None:
        """Get last known charger status."""
        ...

    @property
    def max_current(self) -> float:
        """Get configured max current."""
        ...

    async def do_charge(self, current: float | None = None, phases: int = 1) -> bool:
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

    async def do_set_current(
        self,
        current: float,
        phases: int = 1,
        current_l1: float | None = None,
        current_l2: float | None = None,
        current_l3: float | None = None,
    ) -> bool:
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


# =============================================================================
# BASE FSM CLASS
# =============================================================================


class BaseChargingFSM(ABC):
    """
    Base class for charging FSMs.

    Provides common functionality for state management, callbacks, and locking.
    Subclasses implement specific behavior for different modes.
    """

    MIN_CHARGING_CURRENT = 6.0
    COMMAND_GRACE_PERIOD_SECONDS = 15  # Grace period after issuing a command

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
        
        # Track when we last issued a command to avoid spamming
        self._last_command_time: datetime | None = None
        self._last_command_type: str | None = None  # "start", "pause", "resume", "stop"

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

    @property
    def last_detection_result(self) -> PhaseDetectionResult | None:
        """Get current session's phase detection result (None if not detected yet)."""
        return self._last_detection_result

    def _reset_session(self) -> None:
        """
        Reset session-specific state when car disconnects.
        
        Each session starts fresh - no carrying over stale detection results.
        """
        logger.info("FSM: Resetting session state")
        self._last_detection_result = None
        self._context.car_connected_at = None
        self._context.charging_started_at = None
        self._context.session_energy_kwh = 0.0
        self._last_command_time = None
        self._last_command_type = None
        
        # Notify LoadMonitor to reset detected phases
        from .load_monitor import get_load_monitor
        lm = get_load_monitor()
        if lm:
            lm.set_detected_phases(1)  # Reset to default (1-phase)

    def set_mode(self, mode: ChargingMode) -> None:
        """Set charging mode."""
        old_mode = self._context.mode
        self._context.mode = mode
        logger.info(f"Mode changed: {old_mode.value} -> {mode.value}")

    def on_state_change(self, callback: Callable) -> None:
        """Register callback for FSM state changes."""
        self._state_callbacks.append(callback)

    def restore_context(self, context: FSMContext, detection_result: PhaseDetectionResult | None) -> None:
        """
        Restore FSM context from another FSM instance.
        
        Used when switching between FSMs to preserve state WITHIN the same session.
        Detection result is only restored if car is still connected.
        """
        self._context = context
        # Only restore detection result if we're in an active session (not IDLE)
        if context.state != FSMState.IDLE and detection_result is not None:
            self._last_detection_result = detection_result
        else:
            # Don't carry over stale detection results
            self._last_detection_result = None

    def _is_in_command_grace_period(self) -> bool:
        """
        Check if we're within the grace period after issuing a command.
        
        Returns True if we should skip enforcement to let the charger respond.
        """
        if self._last_command_time is None:
            return False
        
        elapsed = (datetime.now(timezone.utc) - self._last_command_time).total_seconds()
        in_grace = elapsed < self.COMMAND_GRACE_PERIOD_SECONDS
        
        if in_grace:
            logger.debug(
                f"FSM: In command grace period ({elapsed:.1f}s since {self._last_command_type})"
            )
        
        return in_grace

    def _record_command(self, command_type: str) -> None:
        """Record that we just issued a command."""
        self._last_command_time = datetime.now(timezone.utc)
        self._last_command_type = command_type
        logger.debug(f"FSM: Command recorded: {command_type}")

    # =========================================================================
    # ABSTRACT METHODS - Subclasses must implement
    # =========================================================================

    @abstractmethod
    async def start(self) -> None:
        """Start the FSM."""
        ...

    @abstractmethod
    async def stop(self) -> None:
        """Stop the FSM."""
        ...

    @abstractmethod
    async def on_charger_state_change(
        self,
        old_status: ChargerStatus | None,
        new_status: ChargerStatus,
    ) -> None:
        """Called by EaseeController when charger state changes."""
        ...

    @abstractmethod
    async def on_schedule_update(self, schedule: ScheduleInfo) -> None:
        """Called by ScheduleProvider when schedule changes."""
        ...

    @abstractmethod
    async def on_load_update(self, load: LoadInfo) -> None:
        """Called by LoadMonitor when load changes significantly."""
        ...

    @abstractmethod
    async def request_start(self) -> dict:
        """Request to start charging."""
        ...

    @abstractmethod
    async def request_stop(self) -> dict:
        """Request to stop/pause charging."""
        ...

    @abstractmethod
    async def request_resume(self) -> dict:
        """Request to resume from paused state."""
        ...

    @abstractmethod
    def get_status(self) -> dict:
        """Get FSM status for API."""
        ...

    # =========================================================================
    # COMMON ACTIONS - Shared by all FSMs
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
                self._record_command("start")
                logger.info(f"FSM: Started charging at {current}A ({phases}-phase)")
                # Note: LoadMonitor will adjust current based on available capacity
        except Exception as e:
            logger.error(f"FSM: Failed to start charging: {e}")

    async def _do_pause(self, reason: str) -> None:
        """Pause charging."""
        try:
            await self._charger.do_pause()
            self._record_command("pause")
            logger.info(f"FSM: Paused - {reason}")
        except Exception as e:
            logger.error(f"FSM: Failed to pause: {e}")

    async def _do_resume(self) -> None:
        """Resume charging."""
        try:
            await self._charger.do_resume()
            self._record_command("resume")
            logger.info("FSM: Resumed charging")
        except Exception as e:
            logger.error(f"FSM: Failed to resume: {e}")

    async def _do_stop(self) -> None:
        """Stop charging completely."""
        try:
            await self._charger.do_stop()
            self._record_command("stop")
            logger.info("FSM: Stopped charging")
        except Exception as e:
            logger.error(f"FSM: Failed to stop: {e}")

    # Note: Current adjustment is now handled by LoadMonitor directly.
    # FSM only manages charging lifecycle (start/stop/pause/resume).

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

                # Notify LoadMonitor about detected phases (for current calculation)
                from .load_monitor import get_load_monitor
                lm = get_load_monitor()
                if lm:
                    lm.set_detected_phases(result.detected_phases)

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

        # Reset session state when car disconnects (entering IDLE)
        if new_state == FSMState.IDLE:
            self._reset_session()

        # Notify callbacks
        for callback in self._state_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(old_state, new_state, reason)
                else:
                    callback(old_state, new_state, reason)
            except Exception as e:
                logger.warning(f"FSM callback error: {e}")

    def _get_base_status(self) -> dict:
        """Get common FSM status for API."""
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
                    if self._last_detection_result
                    else None
                ),
                "detected_power_kw": (
                    self._last_detection_result.detected_power_kw
                    if self._last_detection_result
                    else None
                ),
                "max_current_per_phase": (
                    self._last_detection_result.max_current_per_phase
                    if self._last_detection_result
                    else None
                ),
            },
        }
