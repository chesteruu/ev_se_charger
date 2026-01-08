"""
EASEE Charger Controller Module.

This module provides:
1. State monitoring - polls EASEE API and emits state change events
2. Basic actions - do_charge(), do_pause(), do_stop(), do_set_current()

It does NOT contain business logic about when to charge.
All decisions are made by the ChargingFSM.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import TYPE_CHECKING, Any

from pyeasee import Charger, Easee, Site
from pyeasee.site import Circuit

from .config import get_config

if TYPE_CHECKING:
    from .charging_fsm import PhaseDetectionResult

logger = logging.getLogger(__name__)


class ChargerState(Enum):
    """EASEE charger operational states."""

    DISCONNECTED = 1  # No car connected
    AWAITING_START = 2  # Car connected, waiting to start
    CHARGING = 3  # Actively charging
    COMPLETED = 4  # Charging completed
    ERROR = 5  # Error state
    READY_TO_CHARGE = 6  # Ready to charge

    @classmethod
    def from_api_state(cls, state) -> ChargerState:
        """Convert API state (string or int) to enum."""
        # Handle string states (newer API format)
        if isinstance(state, str):
            state_str_map = {
                "DISCONNECTED": cls.DISCONNECTED,
                "AWAITING_START": cls.AWAITING_START,
                "AwaitingStart": cls.AWAITING_START,
                "CHARGING": cls.CHARGING,
                "Charging": cls.CHARGING,
                "COMPLETED": cls.COMPLETED,
                "Completed": cls.COMPLETED,
                "ERROR": cls.ERROR,
                "Error": cls.ERROR,
                "READY_TO_CHARGE": cls.READY_TO_CHARGE,
                "ReadyToCharge": cls.READY_TO_CHARGE,
                "PAUSED": cls.AWAITING_START,
                "Paused": cls.AWAITING_START,
                "OFFLINE": cls.DISCONNECTED,
                "Offline": cls.DISCONNECTED,
            }
            return state_str_map.get(state, cls.ERROR)

        # Handle integer states (older API format)
        state_int_map = {
            1: cls.DISCONNECTED,
            2: cls.AWAITING_START,
            3: cls.CHARGING,
            4: cls.COMPLETED,
            5: cls.ERROR,
            6: cls.READY_TO_CHARGE,
        }
        return state_int_map.get(state, cls.ERROR)

    @property
    def is_car_connected(self) -> bool:
        """Check if a car is connected."""
        return self not in (ChargerState.DISCONNECTED, ChargerState.ERROR)

    @property
    def is_charging(self) -> bool:
        """Check if actively charging."""
        return self == ChargerState.CHARGING


@dataclass
class ChargerStatus:
    """Current charger status data."""

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
    raw_data: dict[str, Any]
    output_phase: int = 0  # Number of phases in use (0=none, 1=single, 3=three)
    session_start: datetime | None = None  # Session start time from EASEE API
    # Per-phase output currents from charger (for accurate load calculation)
    current_l1_amps: float = 0.0
    current_l2_amps: float = 0.0
    current_l3_amps: float = 0.0

    @property
    def is_charging(self) -> bool:
        """Check if actively charging."""
        return self.state.is_charging

    @property
    def is_car_connected(self) -> bool:
        """Check if car is connected."""
        return self.state.is_car_connected

    @property
    def detected_phases(self) -> int:
        """Get detected number of phases.

        EASEE API uses encoded values:
        - outputPhase = 10 → 1-phase
        - outputPhase = 20 → 2-phase
        - outputPhase = 30 → 3-phase
        """
        if self.output_phase >= 10:
            # Decode EASEE API value
            return self.output_phase // 10
        if self.output_phase > 0:
            return self.output_phase
        # Fallback: estimate from power if available
        if self.power_watts > 0 and self.current_amps > 0:
            voltage = self.voltage_volts or 230
            estimated_phases = self.power_watts / (self.current_amps * voltage)
            if estimated_phases > 2:
                return 3
            return 1
        return 0

    @property
    def power_kw(self) -> float:
        """Get power in kW."""
        return self.power_watts / 1000


class EaseeController:
    """
    Controller for EASEE EV charger.

    Provides:
    - State monitoring with change callbacks
    - Basic actions: do_charge(), do_pause(), do_stop(), do_set_current()

    Does NOT contain business logic - that's the FSM's job.
    """

    MIN_CURRENT = 6  # Minimum charging current (A)
    MAX_CURRENT = 32  # Maximum charging current (A)

    def __init__(
        self,
        username: str | None = None,
        password: str | None = None,
        charger_id: str | None = None,
    ) -> None:
        config = get_config()

        self._username = username or config.easee.username
        self._password = password or config.easee.password
        self._charger_id = charger_id or config.easee.charger_id
        self._max_current = config.easee.max_current

        self._easee: Easee | None = None
        self._charger: Charger | None = None
        self._site: Site | None = None
        self._circuit: Circuit | None = None
        self._connected = False
        self._last_status: ChargerStatus | None = None

        # State monitoring
        self._state_change_callbacks: list[Callable] = []
        self._monitoring = False
        self._monitor_task: asyncio.Task | None = None
        self._monitor_interval = 2.0

    @property
    def is_connected(self) -> bool:
        """Check if connected to EASEE cloud."""
        return self._connected

    @property
    def charger_id(self) -> str:
        """Get charger ID."""
        return self._charger_id

    @property
    def last_status(self) -> ChargerStatus | None:
        """Get last known charger status."""
        return self._last_status

    @property
    def max_current(self) -> float:
        """Get configured max current."""
        return self._max_current

    # =========================================================================
    # CONNECTION
    # =========================================================================

    async def connect(self) -> bool:
        """Connect to EASEE cloud and initialize charger."""
        try:
            logger.info(f"Connecting to EASEE cloud for charger {self._charger_id}...")

            self._easee = Easee(self._username, self._password)

            # Get sites and find our charger with proper site/circuit hierarchy
            sites = await self._easee.get_sites()

            for site_data in sites:
                # Create Site object from site data
                site_obj = Site(site_data, self._easee)
                
                for circuit_data in site_data.get("circuits", []):
                    # Create Circuit object with site reference
                    circuit_obj = Circuit(circuit_data, site_obj, self._easee)
                    
                    for charger_data in circuit_data.get("chargers", []):
                        if charger_data.get("id") == self._charger_id:
                            # Store site and circuit for dynamic current control
                            self._site = site_obj
                            self._circuit = circuit_obj
                            self._charger = Charger(charger_data, self._easee, site=site_obj, circuit=circuit_obj)
                            self._connected = True
                            logger.info(f"Connected to charger {self._charger_id} (site: {site_obj.name}, circuit: {circuit_obj.id})")
                            return True

            # If not found in sites, try direct access (but warn about limited functionality)
            chargers = await self._easee.get_chargers()
            for charger in chargers:
                if charger.id == self._charger_id:
                    self._charger = charger
                    self._connected = True
                    logger.warning(f"Connected to charger {self._charger_id} without circuit info - dynamic current control may be limited")
                    return True

            logger.error(f"Charger {self._charger_id} not found in account")
            return False

        except Exception as e:
            logger.error(f"Failed to connect to EASEE: {e}")
            self._connected = False
            raise

    async def disconnect(self) -> None:
        """Disconnect from EASEE cloud."""
        await self.stop_monitoring()
        if self._easee:
            try:
                await self._easee.close()
            except Exception as e:
                logger.warning(f"Error during disconnect: {e}")
            finally:
                self._easee = None
                self._charger = None
                self._connected = False
                logger.info("Disconnected from EASEE cloud")

    # =========================================================================
    # STATE MONITORING
    # =========================================================================

    async def get_status(self) -> ChargerStatus:
        """Get current charger status from API."""
        if not self._connected or not self._charger:
            raise RuntimeError("Not connected to EASEE")

        state = await self._charger.get_state()
        
        # Parse session start time from EASEE API state (usually not available)
        session_start_dt = None
        session_start_str = state.get("sessionStart")
        if session_start_str:
            try:
                # EASEE API returns ISO format like "2026-01-04T17:30:00Z"
                session_start_dt = datetime.fromisoformat(session_start_str.replace("Z", "+00:00"))
            except (ValueError, AttributeError) as e:
                logger.warning(f"Failed to parse sessionStart '{session_start_str}': {e}")

        # Per-phase output currents - use circuitTotalPhaseConductorCurrentLX
        # NOT inCurrentT2/T3/T4 which are terminal currents (wiring-dependent)
        # circuitTotalPhaseConductorCurrentLX are the actual L1/L2/L3 phase currents
        ev_current_l1 = state.get("circuitTotalPhaseConductorCurrentL1", 0) or 0
        ev_current_l2 = state.get("circuitTotalPhaseConductorCurrentL2", 0) or 0
        ev_current_l3 = state.get("circuitTotalPhaseConductorCurrentL3", 0) or 0
        
        # Log raw per-phase currents for debugging
        if state.get("chargerOpMode", 0) == 3:  # Only log when CHARGING
            logger.info(
                f"EASEE per-phase currents: L1={ev_current_l1:.1f}A, L2={ev_current_l2:.1f}A, "
                f"L3={ev_current_l3:.1f}A (outputCurrent={state.get('outputCurrent', 0)}A)"
            )

        status = ChargerStatus(
            state=ChargerState.from_api_state(state.get("chargerOpMode", 1)),
            is_online=state.get("isOnline", False),
            power_watts=state.get("totalPower", 0) * 1000,
            current_amps=state.get("outputCurrent", 0),
            voltage_volts=state.get("voltage", 230),
            energy_session_kwh=state.get("sessionEnergy", 0),
            energy_total_kwh=state.get("lifetimeEnergy", 0),
            temperature_celsius=state.get("internalTemperature", 0),
            max_current_amps=state.get("maxChargerCurrent", self._max_current),
            dynamic_current_amps=state.get("dynamicCircuitCurrentP1", self._max_current),
            cable_locked=state.get("cableLocked", False),
            timestamp=datetime.now(),
            raw_data=state,
            output_phase=state.get("outputPhase", 0),
            session_start=session_start_dt,
            current_l1_amps=ev_current_l1,
            current_l2_amps=ev_current_l2,
            current_l3_amps=ev_current_l3,
        )

        self._last_status = status
        return status

    async def get_current_session(self) -> dict | None:
        """Get the current/ongoing charging session from EASEE API.
        
        Note: EASEE sessions API only returns COMPLETED sessions, not ongoing ones.
        For ongoing sessions, use FSM timestamps (car_connected_at, charging_started_at).
        
        This method is kept for future use when session ends.
        """
        if not self._connected or not self._charger:
            return None
        
        try:
            # Look for sessions starting from 24 hours ago to now
            now = datetime.now(timezone.utc)
            from_date = now - timedelta(hours=24)
            
            sessions = await self._charger.get_sessions_between_dates(from_date, now)
            
            if sessions:
                # Get the most recent session
                latest = sessions[-1] if isinstance(sessions, list) else sessions
                
                # Try various field names for session start
                session_start = (
                    latest.get("carConnected") or 
                    latest.get("sessionStart") or 
                    latest.get("chargeStartDateTime") or
                    latest.get("startDateTime")
                )
                if session_start:
                    if isinstance(session_start, str):
                        session_start = datetime.fromisoformat(session_start.replace("Z", "+00:00"))
                    
                    return {
                        "session_start": session_start,
                        "energy_kwh": latest.get("sessionEnergy") or latest.get("kiloWattHours", 0),
                        "is_complete": latest.get("sessionEnd") is not None or latest.get("chargeEndDateTime") is not None,
                    }
            return None
        except Exception as e:
            logger.warning(f"Failed to get current session: {e}")
            return None

    def on_state_change(self, callback: Callable) -> None:
        """Register callback for charger state changes."""
        self._state_change_callbacks.append(callback)

    async def start_monitoring(self, interval: float = 2.0) -> None:
        """Start monitoring charger state for changes."""
        if self._monitoring:
            return

        self._monitor_interval = interval
        self._monitoring = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info(f"Started charger state monitoring (interval={interval}s)")

    async def stop_monitoring(self) -> None:
        """Stop monitoring charger state."""
        self._monitoring = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
            self._monitor_task = None
        logger.info("Stopped charger state monitoring")

    async def _monitor_loop(self) -> None:
        """Internal loop that polls charger state and detects changes."""
        while self._monitoring:
            try:
                old_status = self._last_status
                new_status = await self.get_status()

                # Detect state change
                state_changed = (
                    old_status is None or
                    old_status.state != new_status.state
                )

                if state_changed:
                    logger.info(
                        f"Charger state: "
                        f"{old_status.state.name if old_status else 'None'} -> "
                        f"{new_status.state.name}"
                    )

                # Always notify callbacks so FSM gets latest power/current values
                for callback in self._state_change_callbacks:
                    try:
                        if asyncio.iscoroutinefunction(callback):
                            await callback(old_status, new_status)
                        else:
                            callback(old_status, new_status)
                    except Exception as e:
                        logger.warning(f"State change callback error: {e}")

                await asyncio.sleep(self._monitor_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"Error in charger monitor loop: {e}")
                await asyncio.sleep(self._monitor_interval * 2)

    # =========================================================================
    # ACTIONS - These are the ONLY ways to control the charger
    # Called by FSM, not directly by GUI/scheduler
    # =========================================================================

    async def do_charge(self, current: float | None = None, phases: int = 1) -> bool:
        """
        Start charging.

        Args:
            current: Initial charging current (default: MIN_CURRENT for safety)
            phases: Number of phases the car uses (1 or 3)

        Returns:
            True if command sent successfully
        """
        if not self._connected or not self._charger:
            raise RuntimeError("Not connected to EASEE")

        start_current = int(current or self.MIN_CURRENT)
        start_current = max(self.MIN_CURRENT, min(start_current, self._max_current))

        try:
            logger.info(f"do_charge: Starting at {start_current}A ({phases}-phase)")

            # Set dynamic current on circuit level (if circuit available)
            if self._circuit:
                try:
                    await asyncio.wait_for(
                        self._circuit.set_dynamic_current(
                            currentP1=start_current,
                            currentP2=start_current,
                            currentP3=start_current
                        ),
                        timeout=5.0
                    )
                    logger.info(f"do_charge: Circuit dynamic current set to {start_current}A")
                except asyncio.TimeoutError:
                    logger.warning("do_charge: set_dynamic_current timed out, continuing with start")
                except Exception as e:
                    logger.warning(f"do_charge: set_dynamic_current failed: {e}, continuing with start")
            else:
                logger.warning("do_charge: No circuit available, skipping dynamic current")

            # Override any schedule
            try:
                await asyncio.wait_for(self._charger.override_schedule(), timeout=5.0)
            except Exception:
                pass

            # Start charging
            await asyncio.wait_for(self._charger.start(), timeout=10.0)
            return True

        except asyncio.TimeoutError:
            logger.error("do_charge: start command timed out")
            return False
        except Exception as e:
            logger.error(f"do_charge failed: {e}")
            return False

    async def do_pause(self) -> bool:
        """
        Pause charging (car stays connected).

        Returns:
            True if command sent successfully
        """
        if not self._connected or not self._charger:
            raise RuntimeError("Not connected to EASEE")

        try:
            logger.info("do_pause: Pausing charging")
            await asyncio.wait_for(self._charger.pause(), timeout=10.0)
            return True
        except asyncio.TimeoutError:
            logger.error("do_pause: timed out")
            return False
        except Exception as e:
            logger.error(f"do_pause failed: {e}")
            return False

    async def do_resume(self) -> bool:
        """
        Resume charging after pause.

        Returns:
            True if command sent successfully
        """
        if not self._connected or not self._charger:
            raise RuntimeError("Not connected to EASEE")

        try:
            logger.info("do_resume: Resuming charging")
            await asyncio.wait_for(self._charger.resume(), timeout=10.0)
            return True
        except asyncio.TimeoutError:
            logger.error("do_resume: timed out")
            return False
        except Exception as e:
            logger.error(f"do_resume failed: {e}")
            return False

    async def do_stop(self) -> bool:
        """
        Stop charging completely.

        Returns:
            True if command sent successfully
        """
        if not self._connected or not self._charger:
            raise RuntimeError("Not connected to EASEE")

        try:
            logger.info("do_stop: Stopping charging")
            await asyncio.wait_for(self._charger.stop(), timeout=10.0)
            return True
        except asyncio.TimeoutError:
            logger.error("do_stop: timed out")
            return False
        except Exception as e:
            logger.error(f"do_stop failed: {e}")
            return False

    async def do_set_current(
        self, 
        current: float, 
        phases: int = 1,
        current_l1: float | None = None,
        current_l2: float | None = None,
        current_l3: float | None = None,
    ) -> bool:
        """
        Set charging current using dynamic circuit current for best utilization.

        For 3-phase charging, can set different limits per phase for optimal power.

        Args:
            current: Default current in amps (used if per-phase not specified)
            phases: Number of phases the car is using (1 or 3)
            current_l1: Optional per-phase limit for L1
            current_l2: Optional per-phase limit for L2
            current_l3: Optional per-phase limit for L3

        Returns:
            True if command sent successfully
        """
        if not self._connected or not self._charger:
            raise RuntimeError("Not connected to EASEE")

        # Clamp to valid range and convert to integer (Easee API requires int)
        def clamp(val: float) -> int:
            return int(max(self.MIN_CURRENT, min(val, self._max_current)))

        # Use per-phase currents if provided, otherwise use default
        p1 = clamp(current_l1 if current_l1 is not None else current)
        p2 = clamp(current_l2 if current_l2 is not None else current)
        p3 = clamp(current_l3 if current_l3 is not None else current)

        if not self._circuit:
            logger.warning("do_set_current: No circuit available")
            return False

        try:
            logger.info(f"Setting circuit dynamic current: P1={p1}A, P2={p2}A, P3={p3}A")
            await asyncio.wait_for(
                self._circuit.set_dynamic_current(
                    currentP1=p1,
                    currentP2=p2,
                    currentP3=p3
                ),
                timeout=5.0
            )
            return True
        except asyncio.TimeoutError:
            logger.error("do_set_current: timed out")
            return False
        except Exception as e:
            logger.error(f"do_set_current failed: {e}")
            return False

    async def do_detect_phases(self) -> PhaseDetectionResult | None:
        """
        Perform phase detection by brief charging.

        Starts charging briefly to detect how many phases the car uses,
        then pauses. This allows accurate scheduling based on actual power.

        Returns:
            PhaseDetectionResult with detected phases and power, or None if failed
        """
        from .charging_fsm import PhaseDetectionResult

        if not self._connected or not self._charger:
            raise RuntimeError("Not connected to EASEE")

        try:
            logger.info("do_detect_phases: Starting detection charge...")

            # Start charging at minimum current
            await self._charger.set_dynamic_charger_current(self.MIN_CURRENT)

            try:
                await self._charger.override_schedule()
            except Exception:
                pass

            await self._charger.start()

            # Wait for charging to stabilize and get accurate power reading
            # Cars need time to negotiate and start drawing full current
            await asyncio.sleep(15)

            # Read the state to detect phases
            state = await self._charger.get_state()

            power_kw = state.get("totalPower", 0)
            output_phase_raw = state.get("outputPhase", 0)
            output_current = state.get("outputCurrent", 0)

            # Get per-phase current measurements (actual L1/L2/L3, not terminal currents)
            current_l1 = state.get("circuitTotalPhaseConductorCurrentL1", 0) or 0
            current_l2 = state.get("circuitTotalPhaseConductorCurrentL2", 0) or 0
            current_l3 = state.get("circuitTotalPhaseConductorCurrentL3", 0) or 0

            logger.info(
                f"do_detect_phases: raw state - outputPhase={output_phase_raw}, "
                f"totalPower={power_kw:.2f}kW, outputCurrent={output_current}A, "
                f"L1={current_l1:.1f}A, L2={current_l2:.1f}A, L3={current_l3:.1f}A"
            )

            # Determine phases from POWER / CURRENT ratio (works regardless of current limit)
            # 1-phase: P = V × I → ratio = P / (V × I) ≈ 1
            # 3-phase: P = V × I × 3 → ratio = P / (V × I) ≈ 3
            num_phases = 0
            voltage = state.get("voltage", 230) or 230
            
            # Get actual current - use max of phase currents or outputCurrent
            actual_current = max(current_l1, current_l2, current_l3, output_current or 0)
            
            if power_kw > 0.5 and actual_current > 3:
                # Calculate ratio: actual_power / expected_single_phase_power
                expected_1phase_kw = (actual_current * voltage) / 1000
                ratio = power_kw / expected_1phase_kw
                
                logger.info(
                    f"do_detect_phases: current={actual_current:.1f}A, "
                    f"expected_1phase={expected_1phase_kw:.2f}kW, ratio={ratio:.2f}"
                )
                
                # ratio < 2 → 1-phase, ratio >= 2 → 3-phase
                if ratio < 2.0:
                    num_phases = 1
                else:
                    num_phases = 3
                logger.info(f"do_detect_phases: Detected {num_phases}-phase (ratio={ratio:.2f})")
            else:
                logger.warning(
                    f"do_detect_phases: Insufficient data (power={power_kw:.2f}kW, "
                    f"current={actual_current:.1f}A)"
                )

            # DEFAULT: Assume 1-phase if detection failed
            if num_phases == 0:
                num_phases = 1
                logger.warning("do_detect_phases: Defaulting to 1-phase")

            # Pause after detection
            await self._charger.pause()

            # Calculate max power based on detected phases
            max_power_kw = (self._max_current * 230 * num_phases) / 1000

            result = PhaseDetectionResult(
                detected_phases=num_phases,
                detected_power_kw=max_power_kw,
                max_current_per_phase=self._max_current,
            )

            logger.info(
                f"do_detect_phases: Detected {num_phases} phase(s), "
                f"max power {max_power_kw:.1f}kW"
            )

            return result

        except Exception as e:
            logger.error(f"do_detect_phases failed: {e}")
            # Try to pause in case we started charging
            try:
                await self._charger.pause()
            except Exception:
                pass
            return None

    # =========================================================================
    # CONTEXT MANAGER
    # =========================================================================

    async def __aenter__(self) -> EaseeController:
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.disconnect()
