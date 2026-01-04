"""
SaveEye Monitor Module.

Connects to SaveEye device to monitor real-time electricity consumption.
Supports two modes:
- MQTT: Subscribe to SaveEye data published to MQTT broker (recommended)
- HTTP: Poll SaveEye device directly via HTTP (legacy)
"""

from __future__ import annotations

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import aiohttp

from .config import get_config

logger = logging.getLogger(__name__)


@dataclass
class PowerReading:
    """Real-time power reading from SaveEye."""

    timestamp: datetime

    # Power values in Watts
    power_import: float  # Power being imported from grid
    power_export: float = 0.0  # Power being exported (e.g., solar)
    power_net: float = 0.0  # Net power (import - export)

    # Per-phase values (for 3-phase systems)
    power_l1: float = 0.0
    power_l2: float = 0.0
    power_l3: float = 0.0

    # Current values in Amps
    current_l1: float = 0.0
    current_l2: float = 0.0
    current_l3: float = 0.0

    # Voltage values in Volts
    voltage_l1: float = 230.0
    voltage_l2: float = 230.0
    voltage_l3: float = 230.0

    # Cumulative energy in kWh
    energy_import_total: float = 0.0
    energy_export_total: float = 0.0

    @property
    def power_import_kw(self) -> float:
        """Get import power in kW."""
        return self.power_import / 1000

    @property
    def max_phase_current(self) -> float:
        """Get the highest current on any phase."""
        return max(self.current_l1, self.current_l2, self.current_l3)

    @property
    def total_current(self) -> float:
        """Get total current across all phases."""
        return self.current_l1 + self.current_l2 + self.current_l3


@dataclass
class ConsumptionStats:
    """Consumption statistics over a time period."""

    period_start: datetime
    period_end: datetime
    readings_count: int

    # Power statistics in Watts
    avg_power: float
    max_power: float
    min_power: float

    # Current statistics (max per phase)
    avg_max_current: float
    peak_max_current: float

    # Energy in kWh
    energy_consumed: float


class SaveEyeMonitorBase(ABC):
    """Base class for SaveEye monitors."""

    def __init__(self):
        self._running = False
        self._last_reading: PowerReading | None = None
        self._readings_history: deque = deque(maxlen=1000)
        self._reading_callbacks: list[Callable] = []

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def last_reading(self) -> PowerReading | None:
        return self._last_reading

    @property
    def current_power_watts(self) -> float:
        if self._last_reading:
            return self._last_reading.power_import
        return 0.0

    @property
    def current_max_current_amps(self) -> float:
        if self._last_reading:
            return self._last_reading.max_phase_current
        return 0.0

    @abstractmethod
    async def connect(self) -> bool:
        pass

    @abstractmethod
    async def start(self) -> None:
        pass

    @abstractmethod
    async def stop(self) -> None:
        pass

    @abstractmethod
    async def get_reading(self) -> PowerReading | None:
        pass

    def _process_reading(self, reading: PowerReading) -> None:
        """Process a new reading."""
        self._last_reading = reading
        self._readings_history.append(reading)

        # Notify callbacks
        for callback in self._reading_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    asyncio.create_task(callback(reading))
                else:
                    callback(reading)
            except Exception as e:
                logger.warning(f"Reading callback error: {e}")

    def get_consumption_stats(self, period_minutes: int = 15) -> ConsumptionStats | None:
        """Get consumption statistics for a recent period."""
        if not self._readings_history:
            return None

        now = datetime.now()
        cutoff = now - timedelta(minutes=period_minutes)

        recent = [r for r in self._readings_history if r.timestamp >= cutoff]

        if len(recent) < 2:
            return None

        powers = [r.power_import for r in recent]
        currents = [r.max_phase_current for r in recent]

        # Estimate energy
        energy = 0.0
        for i in range(1, len(recent)):
            dt_hours = (recent[i].timestamp - recent[i-1].timestamp).total_seconds() / 3600
            avg_power_kw = (recent[i].power_import + recent[i-1].power_import) / 2000
            energy += avg_power_kw * dt_hours

        return ConsumptionStats(
            period_start=recent[0].timestamp,
            period_end=recent[-1].timestamp,
            readings_count=len(recent),
            avg_power=sum(powers) / len(powers),
            max_power=max(powers),
            min_power=min(powers),
            avg_max_current=sum(currents) / len(currents),
            peak_max_current=max(currents),
            energy_consumed=energy,
        )

    def get_available_current(self, main_fuse_amps: float) -> float:
        """Calculate available current for EV charging."""
        if not self._last_reading:
            return 0.0
        max_current = self._last_reading.max_phase_current
        available = main_fuse_amps - max_current
        return max(0.0, available)

    def on_reading(self, callback: Callable) -> None:
        """Register callback for new readings."""
        self._reading_callbacks.append(callback)

    def remove_reading_callback(self, callback: Callable) -> None:
        """Remove a reading callback."""
        if callback in self._reading_callbacks:
            self._reading_callbacks.remove(callback)

    async def __aenter__(self) -> SaveEyeMonitorBase:
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.stop()


class SaveEyeMQTTMonitor(SaveEyeMonitorBase):
    """
    SaveEye monitor using MQTT.

    Subscribes to MQTT topics where SaveEye publishes power data.
    """

    def __init__(
        self,
        mqtt_host: str = None,
        mqtt_port: int = None,
        mqtt_username: str = None,
        mqtt_password: str = None,
        topic: str = None,
    ):
        super().__init__()

        config = get_config()

        self._mqtt_host = mqtt_host or config.mqtt.host
        self._mqtt_port = mqtt_port or config.mqtt.port
        self._mqtt_username = mqtt_username or config.mqtt.username
        self._mqtt_password = mqtt_password or config.mqtt.password
        self._mqtt_client_id = config.mqtt.client_id

        self._topic = topic or config.saveeye.mqtt_topic

        self._mqtt_task: asyncio.Task | None = None
        self._connected = False

    @property
    def is_connected(self) -> bool:
        """Check if connected to MQTT broker."""
        return self._connected

    async def connect(self) -> bool:
        """Test MQTT connection."""
        try:
            import uuid

            import aiomqtt
            import paho.mqtt.client as paho

            logger.info(f"Testing MQTT connection to {self._mqtt_host}:{self._mqtt_port}...")
            logger.info(f"Topic: {self._topic}")

            test_client_id = f"{self._mqtt_client_id}-test-{uuid.uuid4().hex[:8]}"
            async with aiomqtt.Client(
                hostname=self._mqtt_host,
                port=self._mqtt_port,
                username=self._mqtt_username,
                password=self._mqtt_password,
                identifier=test_client_id,
                protocol=paho.MQTTv311,  # Use MQTT 3.1.1 for broader compatibility
                clean_session=True,
            ) as client:
                # Try to subscribe briefly to test connection
                await client.subscribe(self._topic)
                self._connected = True
                logger.info("MQTT connection successful")
                return True

        except Exception as e:
            logger.error(f"MQTT connection failed: {e}")
            self._connected = False
            return False

    async def start(self) -> None:
        """Start MQTT subscription."""
        if self._running:
            return

        self._running = True
        self._mqtt_task = asyncio.create_task(self._mqtt_loop())
        logger.info(f"SaveEye MQTT monitor started, subscribing to {self._topic}")

    async def stop(self) -> None:
        """Stop MQTT subscription."""
        self._running = False
        if self._mqtt_task:
            self._mqtt_task.cancel()
            try:
                await self._mqtt_task
            except asyncio.CancelledError:
                pass
        logger.info("SaveEye MQTT monitor stopped")

    async def _mqtt_loop(self) -> None:
        """Main MQTT subscription loop."""
        import uuid

        import aiomqtt
        import paho.mqtt.client as paho

        reconnect_delay = 5
        # Use unique client ID to avoid conflicts with other MQTT clients
        unique_client_id = f"{self._mqtt_client_id}-saveeye-{uuid.uuid4().hex[:8]}"

        while self._running:
            try:
                async with aiomqtt.Client(
                    hostname=self._mqtt_host,
                    port=self._mqtt_port,
                    username=self._mqtt_username,
                    password=self._mqtt_password,
                    identifier=unique_client_id,
                    protocol=paho.MQTTv311,  # Use MQTT 3.1.1 for broader compatibility
                    clean_session=True,
                    keepalive=60,
                ) as client:

                    # Subscribe to telemetry topic
                    await client.subscribe(self._topic)
                    self._connected = True

                    logger.info(f"Subscribed to MQTT topic: {self._topic}")

                    async for message in client.messages:
                        if not self._running:
                            break

                        try:
                            reading = self._parse_mqtt_message(message)
                            if reading:
                                self._process_reading(reading)
                        except Exception as e:
                            logger.warning(f"Error parsing MQTT message: {e}")

            except asyncio.CancelledError:
                break
            except Exception as e:
                self._connected = False
                logger.error(f"MQTT connection error: {e}")
                if self._running:
                    logger.info(f"Reconnecting in {reconnect_delay}s...")
                    await asyncio.sleep(reconnect_delay)

    def _parse_mqtt_message(self, message) -> PowerReading | None:
        """Parse MQTT message into PowerReading."""
        topic = str(message.topic)
        payload = message.payload.decode("utf-8")

        logger.debug(f"MQTT message on {topic}: {payload}")

        now = datetime.now()

        # Try to parse as JSON
        try:
            data = json.loads(payload)
            return self._parse_json_payload(data, now)
        except json.JSONDecodeError:
            pass

        # Try to parse as simple numeric value
        try:
            value = float(payload)

            # Determine what the value represents based on topic
            if "power" in topic.lower() or "watt" in topic.lower():
                return PowerReading(timestamp=now, power_import=value)
            elif "current" in topic.lower() or "amp" in topic.lower():
                # Assume it's max phase current, estimate power
                voltage = 230
                return PowerReading(
                    timestamp=now,
                    power_import=value * voltage,
                    current_l1=value,
                )
        except ValueError:
            pass

        logger.warning(f"Could not parse MQTT payload: {payload}")
        return None

    def _parse_json_payload(self, data: dict[str, Any], timestamp: datetime) -> PowerReading:
        """Parse JSON payload into PowerReading."""

        # Check for SaveEye format (nested structures)
        if "activeActualConsumption" in data:
            return self._parse_saveeye_format(data, timestamp)

        # Generic format: common field name variations
        power_fields = ["power", "power_import", "activePower", "active_power", "watt", "watts", "W", "import"]
        export_fields = ["power_export", "export", "powerExport"]

        power_import = 0.0
        for field in power_fields:
            if field in data:
                power_import = float(data[field])
                break

        power_export = 0.0
        for field in export_fields:
            if field in data:
                power_export = float(data[field])
                break

        # Per-phase power
        power_l1 = float(data.get("power_l1", data.get("powerL1", data.get("P1", 0))))
        power_l2 = float(data.get("power_l2", data.get("powerL2", data.get("P2", 0))))
        power_l3 = float(data.get("power_l3", data.get("powerL3", data.get("P3", 0))))

        # Current
        current_l1 = float(data.get("current_l1", data.get("currentL1", data.get("I1", 0))))
        current_l2 = float(data.get("current_l2", data.get("currentL2", data.get("I2", 0))))
        current_l3 = float(data.get("current_l3", data.get("currentL3", data.get("I3", 0))))

        # Voltage
        voltage_l1 = float(data.get("voltage_l1", data.get("voltageL1", data.get("U1", 230))))
        voltage_l2 = float(data.get("voltage_l2", data.get("voltageL2", data.get("U2", 230))))
        voltage_l3 = float(data.get("voltage_l3", data.get("voltageL3", data.get("U3", 230))))

        # Energy
        energy_import = float(data.get("energy_import", data.get("totalImport", data.get("energyImport", 0))))
        energy_export = float(data.get("energy_export", data.get("totalExport", data.get("energyExport", 0))))

        # If no per-phase current but we have power, estimate current
        if current_l1 == 0 and power_l1 > 0:
            current_l1 = power_l1 / voltage_l1
        if current_l2 == 0 and power_l2 > 0:
            current_l2 = power_l2 / voltage_l2
        if current_l3 == 0 and power_l3 > 0:
            current_l3 = power_l3 / voltage_l3

        # If no per-phase data, estimate from total
        if power_l1 == 0 and power_l2 == 0 and power_l3 == 0 and power_import > 0:
            # Assume balanced load across 3 phases
            power_l1 = power_l2 = power_l3 = power_import / 3
            current_l1 = current_l2 = current_l3 = (power_import / 3) / 230

        return PowerReading(
            timestamp=timestamp,
            power_import=power_import,
            power_export=power_export,
            power_net=power_import - power_export,
            power_l1=power_l1,
            power_l2=power_l2,
            power_l3=power_l3,
            current_l1=current_l1,
            current_l2=current_l2,
            current_l3=current_l3,
            voltage_l1=voltage_l1,
            voltage_l2=voltage_l2,
            voltage_l3=voltage_l3,
            energy_import_total=energy_import,
            energy_export_total=energy_export,
        )

    def _parse_saveeye_format(self, data: dict[str, Any], timestamp: datetime) -> PowerReading:
        """
        Parse SaveEye specific JSON format.

        SaveEye format:
        {
            "activeActualConsumption": {"total": 2622, "L1": 339, "L2": 1583, "L3": 699},  # Watts
            "activeActualProduction": {"total": 0, "L1": 0, "L2": 0, "L3": 0},  # Watts
            "rmsCurrent": {"L1": 1800, "L2": 7000, "L3": 3500},  # milliamps!
            "rmsVoltage": {"L1": 234, "L2": 234, "L3": 232},  # Volts
            "activeTotalConsumption": {"total": 69716471},  # Wh total
            "activeTotalProduction": {"total": 14707861}  # Wh total
        }
        """
        # Power consumption (in Watts)
        consumption = data.get("activeActualConsumption", {})
        power_import = float(consumption.get("total", 0))
        power_l1 = float(consumption.get("L1", 0))
        power_l2 = float(consumption.get("L2", 0))
        power_l3 = float(consumption.get("L3", 0))

        # Power production/export (in Watts)
        production = data.get("activeActualProduction", {})
        power_export = float(production.get("total", 0))

        # Current (in milliamps - need to convert to Amps!)
        rms_current = data.get("rmsCurrent", {})
        current_l1 = float(rms_current.get("L1", 0)) / 1000.0  # mA to A
        current_l2 = float(rms_current.get("L2", 0)) / 1000.0  # mA to A
        current_l3 = float(rms_current.get("L3", 0)) / 1000.0  # mA to A

        # Voltage (in Volts)
        rms_voltage = data.get("rmsVoltage", {})
        voltage_l1 = float(rms_voltage.get("L1", 230))
        voltage_l2 = float(rms_voltage.get("L2", 230))
        voltage_l3 = float(rms_voltage.get("L3", 230))

        # Total energy (in Wh - convert to kWh for storage)
        total_consumption = data.get("activeTotalConsumption", {})
        total_production = data.get("activeTotalProduction", {})
        energy_import = float(total_consumption.get("total", 0)) / 1000.0  # Wh to kWh
        energy_export = float(total_production.get("total", 0)) / 1000.0  # Wh to kWh

        logger.debug(
            f"SaveEye parsed: power={power_import}W, "
            f"current=[{current_l1:.1f}A, {current_l2:.1f}A, {current_l3:.1f}A], "
            f"voltage=[{voltage_l1}V, {voltage_l2}V, {voltage_l3}V]"
        )

        return PowerReading(
            timestamp=timestamp,
            power_import=power_import,
            power_export=power_export,
            power_net=power_import - power_export,
            power_l1=power_l1,
            power_l2=power_l2,
            power_l3=power_l3,
            current_l1=current_l1,
            current_l2=current_l2,
            current_l3=current_l3,
            voltage_l1=voltage_l1,
            voltage_l2=voltage_l2,
            voltage_l3=voltage_l3,
            energy_import_total=energy_import,
            energy_export_total=energy_export,
        )

    async def get_reading(self) -> PowerReading | None:
        """Get the most recent reading."""
        return self._last_reading


class SaveEyeHTTPMonitor(SaveEyeMonitorBase):
    """
    SaveEye monitor using HTTP polling (legacy mode).

    Polls SaveEye device directly via HTTP.
    """

    def __init__(
        self,
        host: str = None,
        port: int = None,
        poll_interval: int = None
    ):
        super().__init__()

        config = get_config()

        self._host = host or config.saveeye.http.host
        self._port = port or config.saveeye.http.port
        self._poll_interval = poll_interval or config.saveeye.http.poll_interval

        self._base_url = f"http://{self._host}:{self._port}"
        self._poll_task: asyncio.Task | None = None

    async def connect(self) -> bool:
        """Test HTTP connection."""
        try:
            logger.info(f"Testing HTTP connection to {self._base_url}...")
            reading = await self._fetch_reading()
            if reading:
                logger.info("HTTP connection successful")
                return True
            return False
        except Exception as e:
            logger.error(f"HTTP connection failed: {e}")
            return False

    async def start(self) -> None:
        """Start HTTP polling."""
        if self._running:
            return

        self._running = True
        self._poll_task = asyncio.create_task(self._poll_loop())
        logger.info(f"SaveEye HTTP monitor started (interval: {self._poll_interval}s)")

    async def stop(self) -> None:
        """Stop HTTP polling."""
        self._running = False
        if self._poll_task:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
        logger.info("SaveEye HTTP monitor stopped")

    async def _poll_loop(self) -> None:
        """HTTP polling loop."""
        while self._running:
            try:
                reading = await self._fetch_reading()
                if reading:
                    self._process_reading(reading)
                await asyncio.sleep(self._poll_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Poll error: {e}")
                await asyncio.sleep(self._poll_interval)

    async def _fetch_reading(self) -> PowerReading | None:
        """Fetch reading via HTTP."""
        try:
            async with aiohttp.ClientSession() as session:
                endpoints = ["/api/realtime", "/data.json", "/api/data", "/realtime", "/"]

                for endpoint in endpoints:
                    try:
                        async with session.get(
                            f"{self._base_url}{endpoint}",
                            timeout=aiohttp.ClientTimeout(total=10)
                        ) as response:
                            if response.status == 200:
                                content_type = response.headers.get("Content-Type", "")
                                if "json" in content_type:
                                    data = await response.json()
                                    return self._parse_json_response(data)
                                else:
                                    text = await response.text()
                                    return self._parse_text_response(text)
                    except aiohttp.ClientError:
                        continue

                return None
        except Exception as e:
            logger.error(f"Failed to fetch reading: {e}")
            return None

    def _parse_json_response(self, data: dict[str, Any]) -> PowerReading:
        """Parse JSON response."""
        now = datetime.now()

        if "power" in data or "activePower" in data:
            return PowerReading(
                timestamp=now,
                power_import=float(data.get("power", data.get("activePower", data.get("import", 0)))),
                power_export=float(data.get("powerExport", data.get("export", 0))),
                power_l1=float(data.get("powerL1", data.get("power_l1", 0))),
                power_l2=float(data.get("powerL2", data.get("power_l2", 0))),
                power_l3=float(data.get("powerL3", data.get("power_l3", 0))),
                current_l1=float(data.get("currentL1", data.get("current_l1", 0))),
                current_l2=float(data.get("currentL2", data.get("current_l2", 0))),
                current_l3=float(data.get("currentL3", data.get("current_l3", 0))),
                voltage_l1=float(data.get("voltageL1", data.get("voltage_l1", 230))),
                voltage_l2=float(data.get("voltageL2", data.get("voltage_l2", 230))),
                voltage_l3=float(data.get("voltageL3", data.get("voltage_l3", 230))),
                energy_import_total=float(data.get("energyImport", data.get("totalImport", 0))),
                energy_export_total=float(data.get("energyExport", data.get("totalExport", 0))),
            )

        return PowerReading(
            timestamp=now,
            power_import=float(data.get("power_import", data.get("consumption", data.get("watts", 0)))),
        )

    def _parse_text_response(self, text: str) -> PowerReading:
        """Parse text response."""
        import re

        now = datetime.now()
        power = 0.0

        patterns = [
            r"(\d+(?:\.\d+)?)\s*[wW]",
            r"power[:\s]+(\d+(?:\.\d+)?)",
            r"import[:\s]+(\d+(?:\.\d+)?)",
        ]

        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                power = float(match.group(1))
                break

        return PowerReading(timestamp=now, power_import=power)

    async def get_reading(self) -> PowerReading | None:
        """Get a single reading."""
        return await self._fetch_reading()


class MockSaveEyeMonitor(SaveEyeMonitorBase):
    """Mock SaveEye monitor for testing."""

    def __init__(self, base_load: float = 500, variance: float = 200):
        super().__init__()
        self._base_load = base_load
        self._variance = variance
        self._poll_task: asyncio.Task | None = None

    async def connect(self) -> bool:
        logger.info("Connected to mock SaveEye monitor")
        return True

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._poll_task = asyncio.create_task(self._generate_loop())
        logger.info("Mock SaveEye monitor started")

    async def stop(self) -> None:
        self._running = False
        if self._poll_task:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
        logger.info("Mock SaveEye monitor stopped")

    async def _generate_loop(self) -> None:
        """Generate mock readings."""
        while self._running:
            try:
                reading = await self.get_reading()
                if reading:
                    self._process_reading(reading)
                await asyncio.sleep(5)
            except asyncio.CancelledError:
                break

    async def get_reading(self) -> PowerReading:
        """Generate synthetic reading."""
        import random

        now = datetime.now()
        hour = now.hour

        # Time-based patterns
        if 17 <= hour <= 22:
            base = self._base_load * 1.5
        elif 6 <= hour <= 9:
            base = self._base_load * 1.2
        else:
            base = self._base_load

        power = base + random.uniform(-self._variance, self._variance)
        power = max(100, power)

        # Distribute across phases
        power_l1 = power * random.uniform(0.3, 0.4)
        power_l2 = power * random.uniform(0.3, 0.4)
        power_l3 = power - power_l1 - power_l2

        voltage = 230.0

        return PowerReading(
            timestamp=now,
            power_import=power,
            power_l1=power_l1,
            power_l2=power_l2,
            power_l3=power_l3,
            current_l1=power_l1 / voltage,
            current_l2=power_l2 / voltage,
            current_l3=power_l3 / voltage,
            voltage_l1=voltage,
            voltage_l2=voltage,
            voltage_l3=voltage,
        )


# Unified interface
class SaveEyeMonitor:
    """
    Unified SaveEye monitor interface.

    Automatically selects MQTT or HTTP mode based on configuration.
    """

    def __new__(cls, *args, **kwargs) -> SaveEyeMonitorBase:
        config = get_config()
        mode = config.saveeye.mode

        if mode == "mqtt":
            logger.info("Using MQTT mode for SaveEye")
            return SaveEyeMQTTMonitor(*args, **kwargs)
        else:
            logger.info("Using HTTP mode for SaveEye")
            return SaveEyeHTTPMonitor(*args, **kwargs)


# Global instance
_monitor: SaveEyeMonitorBase | None = None


def get_saveeye_monitor(use_mock: bool = False) -> SaveEyeMonitorBase:
    """
    Get the SaveEye monitor singleton.

    Args:
        use_mock: Use mock monitor for testing
    """
    global _monitor
    if _monitor is None:
        if use_mock:
            _monitor = MockSaveEyeMonitor()
        else:
            _monitor = SaveEyeMonitor()
    return _monitor


def set_saveeye_monitor(monitor: SaveEyeMonitorBase) -> None:
    """Set the global SaveEye monitor instance."""
    global _monitor
    _monitor = monitor


async def test_saveeye():
    """Test SaveEye connection."""
    config = get_config()
    print(f"SaveEye mode: {config.saveeye.mode}")

    if config.saveeye.mode == "mqtt":
        print(f"MQTT broker: {config.mqtt.host}:{config.mqtt.port}")
        print(f"Topic: {config.saveeye.mqtt_topic_power}")
    else:
        print(f"HTTP endpoint: http://{config.saveeye.host}:{config.saveeye.port}")

    monitor = get_saveeye_monitor()

    try:
        connected = await monitor.connect()
        if not connected:
            print("✗ Failed to connect to SaveEye")
            print("  Falling back to mock data for testing")
            monitor = MockSaveEyeMonitor()
            await monitor.connect()
        else:
            print("✓ Connected to SaveEye")

        print("\nStarting 15-second monitoring test...")

        readings_received = []

        def on_reading(r):
            readings_received.append(r)
            print(f"  {r.timestamp.strftime('%H:%M:%S')}: {r.power_import_kw:.2f} kW, "
                  f"L1={r.current_l1:.1f}A, L2={r.current_l2:.1f}A, L3={r.current_l3:.1f}A")

        monitor.on_reading(on_reading)

        async with monitor:
            await asyncio.sleep(15)

        print(f"\nReceived {len(readings_received)} readings")

        stats = monitor.get_consumption_stats(period_minutes=1)
        if stats:
            print(f"Stats: avg={stats.avg_power:.0f}W, max={stats.max_power:.0f}W")

    except Exception as e:
        print(f"✗ Test failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(test_saveeye())
