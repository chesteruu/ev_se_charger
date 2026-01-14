"""
FSM Manager - Manages FSM instances and handles mode switching.

The FSMManager:
- Creates and manages SmartChargingFSM and ManualChargingFSM instances
- Routes events to the active FSM based on current mode
- Handles mode switching while preserving state
- Provides unified API for web app and other components
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .charging_fsm import SmartChargingFSM, set_charging_fsm
from .charging_fsm_base import (
    BaseChargingFSM,
    ChargerController,
    ChargingMode,
    LoadInfo,
    PhaseDetectionConsumer,
    ScheduleInfo,
)
from .manual_fsm import ManualChargingFSM

if TYPE_CHECKING:
    from .charging_fsm_base import ChargerStatus

logger = logging.getLogger(__name__)


class FSMManager:
    """
    Manages FSM instances and handles mode switching.

    Routes all events to the active FSM.
    When mode changes, swaps to the appropriate FSM while preserving state.
    """

    def __init__(self, charger: ChargerController) -> None:
        """
        Initialize FSM Manager.

        Args:
            charger: The charger controller to use
        """
        self._charger = charger

        # Create both FSM instances
        self._smart_fsm = SmartChargingFSM(charger)
        self._manual_fsm = ManualChargingFSM(charger)

        # Active FSM (starts with Smart)
        self._active_fsm: BaseChargingFSM = self._smart_fsm

        # Track mode for switching
        self._current_mode = ChargingMode.SMART

        logger.info("FSMManager initialized with SmartChargingFSM active")

    @property
    def active_fsm(self) -> BaseChargingFSM:
        """Get the currently active FSM."""
        return self._active_fsm

    @property
    def smart_fsm(self) -> SmartChargingFSM:
        """Get the Smart FSM instance."""
        return self._smart_fsm

    @property
    def manual_fsm(self) -> ManualChargingFSM:
        """Get the Manual FSM instance."""
        return self._manual_fsm

    @property
    def mode(self) -> ChargingMode:
        """Get current charging mode."""
        return self._current_mode

    def add_phase_detection_consumer(self, consumer: PhaseDetectionConsumer) -> None:
        """Add a consumer for phase detection results to both FSMs."""
        self._smart_fsm.add_phase_detection_consumer(consumer)
        self._manual_fsm.add_phase_detection_consumer(consumer)

    async def start(self) -> None:
        """Start the active FSM."""
        await self._active_fsm.start()
        logger.info(f"FSMManager started with {self._active_fsm.__class__.__name__}")

    async def stop(self) -> None:
        """Stop the active FSM."""
        await self._active_fsm.stop()
        logger.info("FSMManager stopped")

    def set_mode(self, mode: ChargingMode) -> None:
        """
        Set charging mode and switch FSM if needed.

        SMART, SCHEDULED, SOLAR -> SmartChargingFSM
        MANUAL, IMMEDIATE -> ManualChargingFSM
        """
        old_mode = self._current_mode
        self._current_mode = mode

        # Determine which FSM should handle this mode
        if mode in (ChargingMode.SMART, ChargingMode.SCHEDULED, ChargingMode.SOLAR):
            new_fsm = self._smart_fsm
        else:  # MANUAL, IMMEDIATE
            new_fsm = self._manual_fsm

        # Switch FSM if needed
        if new_fsm != self._active_fsm:
            old_fsm = self._active_fsm
            logger.info(
                f"FSMManager: Switching FSM {old_fsm.__class__.__name__} -> "
                f"{new_fsm.__class__.__name__} (mode: {old_mode.value} -> {mode.value})"
            )

            # Transfer state from old FSM to new FSM
            new_fsm.restore_context(
                old_fsm.context,
                old_fsm.last_detection_result,
            )

            # Update active FSM
            self._active_fsm = new_fsm

            # Update global singleton for backward compatibility
            set_charging_fsm(new_fsm)

        # Set mode on the active FSM
        self._active_fsm.set_mode(mode)

        logger.info(f"FSMManager: Mode set to {mode.value}")

    # =========================================================================
    # EVENT ROUTING - All events go to active FSM
    # =========================================================================

    async def on_charger_state_change(
        self,
        old_status: ChargerStatus | None,
        new_status: ChargerStatus,
    ) -> None:
        """Route charger state change to active FSM."""
        await self._active_fsm.on_charger_state_change(old_status, new_status)

    async def on_schedule_update(self, schedule: ScheduleInfo) -> None:
        """Route schedule update to active FSM."""
        await self._active_fsm.on_schedule_update(schedule)

    async def on_load_update(self, load: LoadInfo) -> None:
        """Route load update to active FSM."""
        await self._active_fsm.on_load_update(load)

    # =========================================================================
    # REQUEST ROUTING - All requests go to active FSM
    # =========================================================================

    async def request_start(self) -> dict:
        """Route start request to active FSM."""
        return await self._active_fsm.request_start()

    async def request_stop(self) -> dict:
        """Route stop request to active FSM."""
        return await self._active_fsm.request_stop()

    async def request_resume(self) -> dict:
        """Route resume request to active FSM."""
        return await self._active_fsm.request_resume()

    # =========================================================================
    # STATUS
    # =========================================================================

    def get_status(self) -> dict:
        """Get FSM status from active FSM."""
        status = self._active_fsm.get_status()
        status["manager"] = {
            "active_fsm": self._active_fsm.__class__.__name__,
            "mode": self._current_mode.value,
        }
        return status

    @property
    def state(self):
        """Get current FSM state."""
        return self._active_fsm.state

    @property
    def context(self):
        """Get current FSM context."""
        return self._active_fsm.context


# =============================================================================
# SINGLETON
# =============================================================================

_fsm_manager_instance: FSMManager | None = None


def get_fsm_manager() -> FSMManager | None:
    """Get the global FSMManager instance."""
    return _fsm_manager_instance


def set_fsm_manager(manager: FSMManager) -> None:
    """Set the global FSMManager instance."""
    global _fsm_manager_instance
    _fsm_manager_instance = manager
