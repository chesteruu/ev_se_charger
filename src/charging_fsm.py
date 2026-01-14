"""
Smart Charging FSM.

Handles schedule-aware charging modes (SMART, SCHEDULED, SOLAR):
- Price-aware scheduling
- Follows charging windows from ScheduleProvider
- Respects load protection

States:
- IDLE: No car connected
- CAR_CONNECTED: Car connected, evaluating
- WAITING_FOR_SCHEDULE: Waiting for charging window
- CHARGING: Actively charging
- PAUSED_LOAD: Paused due to load protection
- PAUSED_SCHEDULE: Paused, outside schedule window
- PAUSED_USER: Paused by user request
- CHARGE_COMPLETE: Charging finished
- ERROR: Error state

Re-exports base types for backward compatibility.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

# Re-export base types for backward compatibility
from .charging_fsm_base import (
    BaseChargingFSM,
    ChargerController,
    ChargerState,
    ChargerStatus,
    ChargingMode,
    CurrentController,
    FSMContext,
    FSMState,
    LoadInfo,
    PhaseDetectionConsumer,
    PhaseDetectionResult,
    ScheduleInfo,
)

logger = logging.getLogger(__name__)

# Re-export for backward compatibility
__all__ = [
    "SmartChargingFSM",
    "ChargingFSM",  # Alias for SmartChargingFSM
    "BaseChargingFSM",
    "ChargerController",
    "ChargerState",
    "ChargerStatus",
    "ChargingMode",
    "CurrentController",
    "FSMContext",
    "FSMState",
    "LoadInfo",
    "PhaseDetectionConsumer",
    "PhaseDetectionResult",
    "ScheduleInfo",
    "get_charging_fsm",
    "set_charging_fsm",
]


class SmartChargingFSM(BaseChargingFSM):
    """
    FSM for smart/scheduled charging.

    Schedule-aware logic:
    - SMART mode: Price-aware, follows schedule windows
    - SCHEDULED mode: Time-based scheduling
    - SOLAR mode: Solar-first charging

    Respects load protection and user overrides.
    """

    def __init__(self, charger: ChargerController) -> None:
        """Initialize SmartChargingFSM."""
        super().__init__(charger)
        logger.info("SmartChargingFSM initialized")

    async def start(self) -> None:
        """Start the FSM."""
        self._running = True

        # Get initial charger state
        try:
            status = await self._charger.get_status()
            await self._on_charger_state_change(None, status)
        except Exception as e:
            logger.error(f"Failed to get initial charger state: {e}")

        logger.info("SmartChargingFSM started")

    async def stop(self) -> None:
        """Stop the FSM."""
        self._running = False
        logger.info("SmartChargingFSM stopped")

    # =========================================================================
    # EXTERNAL TRIGGERS
    # =========================================================================

    async def on_charger_state_change(
        self,
        old_status: ChargerStatus | None,
        new_status: ChargerStatus,
    ) -> None:
        """Called by EaseeController when charger state changes."""
        logger.debug("SmartFSM: on_charger_state_change - waiting for lock...")
        try:
            async with asyncio.timeout(60.0):  # 60s timeout for phase detection
                async with self._lock:
                    logger.debug("SmartFSM: on_charger_state_change - lock acquired")
                    await self._on_charger_state_change(old_status, new_status)
                    logger.debug("SmartFSM: on_charger_state_change - done, releasing lock")
        except asyncio.TimeoutError:
            logger.error("SmartFSM: on_charger_state_change - timeout waiting for lock (possible deadlock)")

    async def on_schedule_update(self, schedule: ScheduleInfo) -> None:
        """Called by ScheduleProvider when schedule changes."""
        try:
            async with asyncio.timeout(30.0):
                async with self._lock:
                    old_in_window = self._context.schedule.is_in_window
                    self._context.schedule = schedule

                    if old_in_window != schedule.is_in_window:
                        logger.info(f"Schedule: in_window={schedule.is_in_window}")
                        await self._evaluate_and_act()
        except asyncio.TimeoutError:
            logger.error("SmartFSM: on_schedule_update - timeout waiting for lock")

    async def on_load_update(self, load: LoadInfo) -> None:
        """Called by LoadMonitor when load changes significantly."""
        try:
            async with asyncio.timeout(30.0):
                async with self._lock:
                    old_should_pause = self._context.load.should_pause
                    self._context.load = load

                    # LoadMonitor handles current adjustment directly.
                    # FSM only cares about should_pause signal.
                    
                    if self._context.state == FSMState.CHARGING:
                        if load.should_pause and not old_should_pause:
                            logger.warning(f"Load protection: pausing (available={load.available_for_ev_amps:.1f}A)")
                            await self._do_pause("Load protection")
                            await self._transition_to(FSMState.PAUSED_LOAD, "Load protection")

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
        except asyncio.TimeoutError:
            logger.error("SmartFSM: on_load_update - timeout waiting for lock")

    # =========================================================================
    # EXTERNAL REQUESTS
    # =========================================================================

    async def request_start(self) -> dict:
        """
        Request to start charging.

        Allows manual start in any mode (user override).
        """
        logger.info("SmartFSM: request_start - waiting for lock...")
        try:
            async with asyncio.timeout(15.0):
                async with self._lock:
                    logger.info("SmartFSM: request_start - lock acquired")
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
                    return {"success": True, "reason": "Started (manual override)"}
        except asyncio.TimeoutError:
            logger.error("SmartFSM: request_start - timeout waiting for lock (another operation may be stuck)")
            return {"success": False, "reason": "System busy, try again"}

    async def request_stop(self) -> dict:
        """
        Request to stop/pause charging.

        Works from CHARGING or WAITING_FOR_SCHEDULE states.
        Transitions to PAUSED_USER to prevent schedule from auto-starting.
        """
        logger.info("SmartFSM: request_stop - waiting for lock...")
        try:
            async with asyncio.timeout(15.0):
                async with self._lock:
                    logger.info("SmartFSM: request_stop - lock acquired")
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
            logger.error("SmartFSM: request_stop - timeout waiting for lock")
            return {"success": False, "reason": "System busy, try again"}

    async def request_resume(self) -> dict:
        """
        Request to resume from user-paused state.

        Returns to WAITING_FOR_SCHEDULE (schedule takes over).
        """
        logger.info("SmartFSM: request_resume - waiting for lock...")
        try:
            async with asyncio.timeout(15.0):
                async with self._lock:
                    logger.info("SmartFSM: request_resume - lock acquired")
                    state = self._context.state

                    if state != FSMState.PAUSED_USER:
                        return {"success": False, "reason": "Not in user-paused state"}

                    # Return to waiting for schedule - schedule will take over
                    await self._transition_to(FSMState.WAITING_FOR_SCHEDULE, "User resumed - schedule takes over")
                    
                    # If we're already in a charging window, start charging
                    if self._should_charge_now() and not self._context.load.should_pause:
                        await self._do_charge()
                        self._context.charging_started_at = datetime.now(timezone.utc)
                        await self._transition_to(FSMState.CHARGING, "Schedule resumed")
                    
                    return {"success": True, "reason": "Resumed - schedule will control charging"}
        except asyncio.TimeoutError:
            logger.error("SmartFSM: request_resume - timeout waiting for lock")
            return {"success": False, "reason": "System busy, try again"}

    # =========================================================================
    # INTERNAL FSM LOGIC
    # =========================================================================

    async def _on_charger_state_change(
        self,
        old_status: ChargerStatus | None,
        new_status: ChargerStatus,
    ) -> None:
        """
        Handle charger state change.
        
        FSM is the AUTHORITY - charger state is just reality.
        If reality doesn't match intent, FSM takes corrective action.
        
        FSM only transitions on DEFINITIVE events:
        - Car connected/disconnected
        - Charge complete
        - Error
        
        For other states, FSM enforces its intent.
        
        Note: FSM uses abstract charger states (READY, CHARGING, etc.)
        The controller handles mapping from EASEE-specific states.
        """
        old_charger_state = self._context.charger_state
        new_charger_state = new_status.state

        # Update context with reality (for reporting)
        self._context.charger_state = new_charger_state
        self._context.charger_power_watts = new_status.power_watts
        self._context.charger_current_amps = new_status.current_amps
        self._context.session_energy_kwh = new_status.energy_session_kwh

        # =====================================================================
        # DEFINITIVE EVENTS - These cause FSM state transitions
        # =====================================================================

        # Car disconnected - always transition to IDLE
        if not new_charger_state.is_car_connected:
            await self._transition_to(FSMState.IDLE, "Car disconnected")
            self._context.car_connected_at = None
            self._context.charging_started_at = None
            return

        # Charge complete - always transition to CHARGE_COMPLETE
        if new_charger_state.is_completed:
            await self._transition_to(FSMState.CHARGE_COMPLETE, "Charging complete")
            return

        # Charger error - always transition to ERROR
        if new_charger_state.is_error:
            await self._transition_to(FSMState.ERROR, "Charger error")
            return

        # Car just connected - transition to CAR_CONNECTED and evaluate
        if new_charger_state.is_ready:
            was_disconnected = old_charger_state is None or not old_charger_state.is_car_connected
            if was_disconnected and self._context.state == FSMState.IDLE:
                self._context.car_connected_at = datetime.now(timezone.utc)
                await self._transition_to(FSMState.CAR_CONNECTED, "Car connected")
                # Entry action: detect phases
                await self._do_detect_phases()
                await self._evaluate_and_act()
                return

        # =====================================================================
        # ENFORCE FSM INTENT - Charger state must match FSM intent
        # =====================================================================

        # Signal load monitor for potential phase switch (1-phase only)
        if new_charger_state.is_charging:
            was_ready = old_charger_state is not None and old_charger_state.is_ready
            if was_ready:
                detected_phases = self._last_detection_result.detected_phases if self._last_detection_result else 1
                if detected_phases == 1:
                    from .load_monitor import get_load_monitor
                    load_monitor = get_load_monitor()
                    if load_monitor:
                        load_monitor.signal_charger_resumed()
                        logger.debug("SmartFSM: Signaled phase switch grace period")

        # Enforce FSM intent based on current FSM state
        await self._enforce_intent(new_charger_state)

    async def _enforce_intent(self, charger_state: ChargerState) -> None:
        """
        Enforce FSM intent - make charger match what FSM wants.
        
        FSM state = INTENT (what we want)
        Charger state = REALITY (what's happening)
        
        If mismatch, take corrective action (with grace period).
        """
        # Check grace period - don't spam commands
        if self._is_in_command_grace_period():
            return

        fsm_state = self._context.state
        charger_is_charging = charger_state.is_charging
        charger_is_ready = charger_state.is_ready

        # FSM wants to be CHARGING
        if fsm_state == FSMState.CHARGING:
            if charger_is_ready:
                # Charger is ready but not charging - start it
                logger.info("SmartFSM: Enforcing CHARGING intent - issuing start command")
                await self._do_charge()
            # If charger is already charging, good - nothing to do

        # FSM wants to be PAUSED (any pause state)
        elif fsm_state in (FSMState.PAUSED_USER, FSMState.PAUSED_SCHEDULE, FSMState.PAUSED_LOAD):
            if charger_is_charging:
                # Charger is charging but we want it paused - pause it
                logger.info(f"SmartFSM: Enforcing {fsm_state.value} intent - issuing pause command")
                await self._do_pause(f"Enforcing {fsm_state.value}")
            # If charger is already paused/ready, good - nothing to do

        # FSM is WAITING_FOR_SCHEDULE - check if we should start
        elif fsm_state == FSMState.WAITING_FOR_SCHEDULE:
            should_charge = self._should_charge_now()
            should_pause_load = self._context.load.should_pause

            if should_charge and not should_pause_load:
                if charger_is_ready:
                    # Schedule says charge now - start charging
                    logger.info("SmartFSM: Schedule window active - starting charging")
                    await self._do_charge()
                    self._context.charging_started_at = datetime.now(timezone.utc)
                    await self._transition_to(FSMState.CHARGING, "Schedule window opened")
            elif charger_is_charging:
                # Charger is charging but schedule says no - pause it
                logger.info("SmartFSM: Outside schedule window - pausing")
                await self._do_pause("Outside schedule window")
                await self._transition_to(FSMState.PAUSED_SCHEDULE, "Outside schedule window")

        # FSM is CAR_CONNECTED - should have been handled by evaluate_and_act
        elif fsm_state == FSMState.CAR_CONNECTED:
            # This state is transient - evaluate_and_act should handle it
            pass

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
                self._context.charging_started_at = datetime.now(timezone.utc)
                await self._transition_to(FSMState.CHARGING, "Schedule started")
            else:
                await self._transition_to(FSMState.WAITING_FOR_SCHEDULE, "Waiting for schedule")

        elif state == FSMState.WAITING_FOR_SCHEDULE:
            if should_charge and not should_pause_load:
                await self._do_charge()
                self._context.charging_started_at = datetime.now(timezone.utc)
                await self._transition_to(FSMState.CHARGING, "Schedule window opened")

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
            # User pause - stay paused until user resumes
            pass

    def _should_charge_now(self) -> bool:
        """Determine if charging should be active based on schedule."""
        mode = self._context.mode
        state = self._context.state

        # In SMART mode, if user paused, don't auto-resume
        if state == FSMState.PAUSED_USER:
            return False

        if mode in (ChargingMode.SMART, ChargingMode.SCHEDULED, ChargingMode.SOLAR):
            return self._context.schedule.is_in_window

        return False

    # =========================================================================
    # STATUS
    # =========================================================================

    def get_status(self) -> dict:
        """Get FSM status for API."""
        status = self._get_base_status()
        status["fsm_type"] = "smart"
        return status


# Alias for backward compatibility
ChargingFSM = SmartChargingFSM


# =============================================================================
# SINGLETON (for backward compatibility - prefer FSMManager)
# =============================================================================

_fsm_instance: BaseChargingFSM | None = None


def get_charging_fsm() -> BaseChargingFSM | None:
    """Get the global ChargingFSM instance."""
    return _fsm_instance


def set_charging_fsm(fsm: BaseChargingFSM) -> None:
    """Set the global ChargingFSM instance."""
    global _fsm_instance
    _fsm_instance = fsm
