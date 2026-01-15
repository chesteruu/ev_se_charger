"""
Manual Charging FSM.

Handles manual and immediate charging modes with simplified lifecycle:
- User has full control over start/stop
- No schedule-based decisions
- Still respects load protection for safety

States:
- IDLE: No car connected
- CAR_CONNECTED: Car connected, waiting for user action
- CHARGING: Actively charging
- PAUSED_LOAD: Paused due to load protection (auto-resumes when safe)
- PAUSED_USER: Paused by user request
- CHARGE_COMPLETE: Charging finished
- ERROR: Error state
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

from .charging_fsm_base import (
    BaseChargingFSM,
    ChargerController,
    ChargerState,
    ChargerStatus,
    ChargingMode,
    FSMState,
    LoadInfo,
    ScheduleInfo,
)

logger = logging.getLogger(__name__)


class ManualChargingFSM(BaseChargingFSM):
    """
    FSM for manual charging control.

    Simplified logic:
    - MANUAL mode: User must explicitly start/stop
    - IMMEDIATE mode: Auto-starts when car connected
    - Load protection still applies (safety)
    - No schedule-based decisions
    """

    def __init__(self, charger: ChargerController) -> None:
        """Initialize ManualChargingFSM."""
        super().__init__(charger)
        logger.info("ManualChargingFSM initialized")

    async def start(self) -> None:
        """Start the FSM."""
        self._running = True

        # Get initial charger state
        try:
            status = await self._charger.get_status()
            await self._on_charger_state_change(None, status)
        except Exception as e:
            logger.error(f"Failed to get initial charger state: {e}")

        logger.info("ManualChargingFSM started")

    async def stop(self) -> None:
        """Stop the FSM."""
        self._running = False
        logger.info("ManualChargingFSM stopped")

    # =========================================================================
    # EXTERNAL TRIGGERS
    # =========================================================================

    async def on_charger_state_change(
        self,
        old_status: ChargerStatus | None,
        new_status: ChargerStatus,
    ) -> None:
        """Called by EaseeController when charger state changes."""
        logger.debug("ManualFSM: on_charger_state_change - waiting for lock...")
        try:
            async with asyncio.timeout(60.0):
                async with self._lock:
                    logger.debug("ManualFSM: on_charger_state_change - lock acquired")
                    await self._on_charger_state_change(old_status, new_status)
        except asyncio.TimeoutError:
            logger.error("ManualFSM: on_charger_state_change - timeout waiting for lock")

    async def on_schedule_update(self, schedule: ScheduleInfo) -> None:
        """
        Called by ScheduleProvider when schedule changes.
        
        In manual mode, we ignore schedule updates (no schedule-based decisions).
        """
        # Just update context but don't act on it
        self._context.schedule = schedule

    async def on_load_update(self, load: LoadInfo) -> None:
        """Called by LoadMonitor when load changes significantly."""
        try:
            async with asyncio.timeout(30.0):
                async with self._lock:
                    old_should_pause = self._context.load.should_pause
                    self._context.load = load

                    # LoadMonitor handles current adjustment directly.
                    # FSM only cares about should_pause signal for load protection.
                    
                    if self._context.state == FSMState.CHARGING:
                        if load.should_pause and not old_should_pause:
                            logger.warning(
                                f"ManualFSM: Load protection - pausing (available={load.available_for_ev_amps:.1f}A)"
                            )
                            await self._do_pause("Load protection")
                            await self._transition_to(FSMState.PAUSED_LOAD, "Load protection")

                    # Auto-resume from load pause when safe
                    elif self._context.state == FSMState.PAUSED_LOAD and not load.should_pause:
                        logger.info(
                            f"ManualFSM: Load protection cleared - resuming (available={load.available_for_ev_amps:.1f}A)"
                        )
                        # Trigger grace period for 1-phase charging
                        detected_phases = self._last_detection_result.detected_phases if self._last_detection_result else 1
                        if detected_phases == 1:
                            from .load_monitor import get_load_monitor
                            lm = get_load_monitor()
                            if lm:
                                lm.signal_charger_resumed()
                        await self._do_resume()
                        await self._transition_to(FSMState.CHARGING, "Load protection cleared")
        except asyncio.TimeoutError:
            logger.error("ManualFSM: on_load_update - timeout waiting for lock")

    # =========================================================================
    # EXTERNAL REQUESTS
    # =========================================================================

    async def request_start(self) -> dict:
        """
        Request to start charging.
        
        In manual mode, this immediately starts charging.
        """
        logger.info("ManualFSM: request_start - waiting for lock...")
        try:
            async with asyncio.timeout(15.0):
                async with self._lock:
                    logger.info("ManualFSM: request_start - lock acquired")
                    state = self._context.state

                    # Check preconditions
                    if state == FSMState.IDLE:
                        return {"success": False, "reason": "No car connected"}

                    if state == FSMState.CHARGING:
                        return {"success": False, "reason": "Already charging"}

                    if state == FSMState.CHARGE_COMPLETE:
                        return {"success": False, "reason": "Charging complete"}

                    if state == FSMState.PAUSED_LOAD:
                        return {"success": False, "reason": "Load protection active - wait for home load to decrease"}

                    # Start charging immediately
                    await self._do_charge()
                    self._context.charging_started_at = datetime.now(timezone.utc)
                    await self._transition_to(FSMState.CHARGING, "Manual start")
                    return {"success": True, "reason": "Started charging"}
        except asyncio.TimeoutError:
            logger.error("ManualFSM: request_start - timeout waiting for lock")
            return {"success": False, "reason": "System busy, try again"}

    async def request_stop(self) -> dict:
        """
        Request to stop/pause charging.
        
        In manual mode, this immediately stops charging.
        """
        logger.info("ManualFSM: request_stop - waiting for lock...")
        try:
            async with asyncio.timeout(15.0):
                async with self._lock:
                    logger.info("ManualFSM: request_stop - lock acquired")
                    state = self._context.state

                    # Can stop from charging or car_connected states
                    if state not in (FSMState.CHARGING, FSMState.CAR_CONNECTED, FSMState.PAUSED_LOAD):
                        return {"success": False, "reason": "Not in a stoppable state"}

                    # If charging, pause the charger
                    if state == FSMState.CHARGING:
                        await self._do_pause("User request")

                    await self._transition_to(FSMState.PAUSED_USER, "User stopped")
                    return {"success": True, "reason": "Stopped by user"}
        except asyncio.TimeoutError:
            logger.error("ManualFSM: request_stop - timeout waiting for lock")
            return {"success": False, "reason": "System busy, try again"}

    async def request_resume(self) -> dict:
        """
        Request to resume from paused state.
        
        In manual mode, this immediately resumes charging.
        """
        logger.info("ManualFSM: request_resume - waiting for lock...")
        try:
            async with asyncio.timeout(15.0):
                async with self._lock:
                    logger.info("ManualFSM: request_resume - lock acquired")
                    state = self._context.state

                    if state != FSMState.PAUSED_USER:
                        return {"success": False, "reason": "Not in user-paused state"}

                    # Check load protection before resuming
                    if self._context.load.should_pause:
                        await self._transition_to(FSMState.PAUSED_LOAD, "Load protection")
                        return {"success": False, "reason": "Cannot resume - load protection active"}

                    # Resume charging
                    await self._do_resume()
                    self._context.charging_started_at = datetime.now(timezone.utc)
                    await self._transition_to(FSMState.CHARGING, "User resumed")
                    return {"success": True, "reason": "Resumed charging"}
        except asyncio.TimeoutError:
            logger.error("ManualFSM: request_resume - timeout waiting for lock")
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

        # Car just connected - transition to CAR_CONNECTED and start phase detection
        if new_charger_state.is_ready:
            was_disconnected = old_charger_state is None or not old_charger_state.is_car_connected
            if was_disconnected and self._context.state == FSMState.IDLE:
                self._context.car_connected_at = datetime.now(timezone.utc)
                await self._transition_to(FSMState.CAR_CONNECTED, "Car connected")
                
                # Phase detection (runs in background, doesn't block)
                # _post_detection_evaluate will be called when detection completes
                await self._do_detect_phases()
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
                        logger.debug("ManualFSM: Signaled phase switch grace period")

        # Enforce FSM intent based on current FSM state
        await self._enforce_intent(new_charger_state)

    async def _enforce_intent(self, charger_state: ChargerState) -> None:
        """
        Enforce FSM intent - make charger match what FSM wants.
        
        FSM state = INTENT (what we want)
        Charger state = REALITY (what's happening)
        
        If mismatch, take corrective action (with grace period).
        """
        # Check command grace period - don't spam commands
        if self._is_in_command_grace_period():
            return
        
        # Check LoadMonitor's phase switch grace period - charger may be unstable
        from .load_monitor import get_load_monitor
        load_monitor = get_load_monitor()
        if load_monitor and load_monitor.is_in_grace_period():
            logger.debug("ManualFSM: Skipping enforcement - LoadMonitor in grace period")
            return

        fsm_state = self._context.state
        charger_is_charging = charger_state.is_charging
        charger_is_ready = charger_state.is_ready

        # FSM wants to be CHARGING
        if fsm_state == FSMState.CHARGING:
            if charger_is_ready:
                # Charger is ready but not charging - resume it
                # Use resume instead of charge to avoid resetting current to 6A
                logger.info("ManualFSM: Enforcing CHARGING intent - issuing resume command")
                await self._do_resume()
            # If charger is already charging, good - nothing to do

        # FSM wants to be PAUSED_USER
        elif fsm_state == FSMState.PAUSED_USER:
            if charger_is_charging:
                # Charger is charging but user wants it paused - pause it
                logger.info("ManualFSM: Enforcing PAUSED_USER intent - issuing pause command")
                await self._do_pause("User requested pause")
            # If charger is already paused/ready, good - nothing to do

        # FSM is in PAUSED_LOAD - load protection active
        elif fsm_state == FSMState.PAUSED_LOAD:
            if charger_is_charging:
                # Charger is charging but load protection is active - pause it
                logger.info("ManualFSM: Enforcing PAUSED_LOAD intent - issuing pause command")
                await self._do_pause("Load protection")
            # If charger is already paused/ready, good - nothing to do

        # FSM is CAR_CONNECTED (MANUAL mode) - waiting for user to start
        elif fsm_state == FSMState.CAR_CONNECTED:
            if charger_is_charging:
                # In MANUAL mode, user hasn't started yet, but charger is charging
                # This could be external start - in manual mode we allow it
                if self._context.mode == ChargingMode.MANUAL:
                    # Allow external start in manual mode
                    logger.info("ManualFSM: Charger started externally - transitioning to CHARGING")
                    self._context.charging_started_at = datetime.now(timezone.utc)
                    await self._transition_to(FSMState.CHARGING, "External start")
    
    async def _post_detection_evaluate(self) -> None:
        """
        Evaluate and act after phase detection completes.
        
        Called from background task with lock held.
        """
        logger.info("ManualFSM: Post-detection evaluation")
        
        # In IMMEDIATE mode, auto-start after phase detection
        if self._context.mode == ChargingMode.IMMEDIATE:
            if self._context.state == FSMState.CAR_CONNECTED:
                if not self._context.load.should_pause:
                    await self._do_charge()
                    self._context.charging_started_at = datetime.now(timezone.utc)
                    await self._transition_to(FSMState.CHARGING, "Immediate mode - auto start")
                else:
                    logger.info("ManualFSM: Immediate mode - waiting for load protection to clear")
                    await self._transition_to(FSMState.PAUSED_LOAD, "Load protection")

    # =========================================================================
    # STATUS
    # =========================================================================

    def get_status(self) -> dict:
        """Get FSM status for API."""
        status = self._get_base_status()
        status["fsm_type"] = "manual"
        return status
