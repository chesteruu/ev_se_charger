"""
Configuration management for EASEE Smart Charger Controller.

Supports loading from:
1. config.yaml (primary, structured configuration)
2. Environment variables (overrides, useful for Docker)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class EaseeConfig:
    """EASEE charger configuration."""
    username: str = ""
    password: str = ""
    charger_id: str = ""
    max_current: int = 32


@dataclass
class MQTTConfig:
    """MQTT broker configuration."""
    host: str = "localhost"
    port: int = 1883
    username: str | None = None
    password: str | None = None
    client_id: str = "easee-controller"
    use_tls: bool = False


@dataclass
class SaveEyeHTTPConfig:
    """SaveEye HTTP configuration."""
    host: str = "192.168.1.100"
    port: int = 80
    poll_interval: int = 5


@dataclass
class SaveEyeConfig:
    """SaveEye monitoring configuration."""
    enabled: bool = True
    mode: str = "mqtt"
    mqtt_topic: str = "saveeye/telemetry"
    required: bool = False
    http: SaveEyeHTTPConfig = field(default_factory=SaveEyeHTTPConfig)


@dataclass
class HomeElectricalConfig:
    """Home electrical system configuration."""
    main_fuse_amps: int = 25
    safety_margin_amps: int = 3
    phases: int = 3
    voltage: int = 230

    @property
    def max_available_current(self) -> int:
        """Maximum available current for charging."""
        return self.main_fuse_amps - self.safety_margin_amps

    @property
    def max_power_watts(self) -> float:
        """Maximum power capacity in Watts."""
        return self.main_fuse_amps * self.voltage * self.phases


@dataclass
class PriceConfig:
    """Electricity price configuration."""
    source: str = "nordpool"
    nordpool_area: str = "NO1"  # Timezone auto-detected from area
    tibber_access_token: str | None = None
    currency: str = "NOK"
    vat_percent: float = 25.0  # VAT/moms percentage (25% in Sweden)


@dataclass
class GridTariffConfig:
    """Grid tariff (effektavgift) configuration - fees added on top of spot price."""
    enabled: bool = True
    day_rate_ore: float = 85.0  # öre/kWh during daytime (06-22)
    night_rate_ore: float = 42.0  # öre/kWh during nighttime (22-06)
    night_start_hour: int = 22  # Hour when night rate begins
    night_end_hour: int = 6  # Hour when night rate ends

    def get_rate_for_hour(self, hour: int) -> float:
        """Get the grid tariff rate (SEK/kWh) for a given hour."""
        if not self.enabled:
            return 0.0

        # Night hours: 22:00-05:59 (night_start to night_end-1)
        is_night = (hour >= self.night_start_hour or hour < self.night_end_hour)
        rate_ore = self.night_rate_ore if is_night else self.day_rate_ore
        return rate_ore / 100.0  # Convert öre to SEK

    def is_night_hour(self, hour: int) -> bool:
        """Check if the given hour falls within night rate period."""
        return hour >= self.night_start_hour or hour < self.night_end_hour


@dataclass
class ChargingConfig:
    """EV charging configuration."""
    target_kwh: float = 50.0  # Default target kWh per charge
    charging_power_kw: float = 11.0  # Default charging power (11kW for 3-phase 16A)


@dataclass
class SmartChargingConfig:
    """Smart charging configuration."""
    enabled: bool = True
    price_threshold: float = 1.5
    lookahead_hours: int = 24
    min_soc_target: int = 80
    preferred_hours: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 4, 5, 6])

    @property
    def preferred_hours_list(self) -> list[int]:
        """Get preferred hours as list."""
        return self.preferred_hours


@dataclass
class WebAuthConfig:
    """Web authentication configuration."""
    enabled: bool = False
    username: str = "admin"
    password: str = "changeme"


@dataclass
class WebConfig:
    """Web dashboard configuration."""
    host: str = "0.0.0.0"
    port: int = 8080
    auth: WebAuthConfig = field(default_factory=WebAuthConfig)

    # Compatibility properties
    @property
    def auth_enabled(self) -> bool:
        return self.auth.enabled

    @property
    def auth_username(self) -> str:
        return self.auth.username

    @property
    def auth_password(self) -> str:
        return self.auth.password


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    file: str | None = None


@dataclass
class DatabaseConfig:
    """Database configuration."""
    path: str = "data/easee_controller.db"


@dataclass
class SAJCloudConfig:
    """SAJ Cloud API configuration (eop.saj-electric.com / Elekeeper)."""
    username: str = ""  # Your eSolar/Elekeeper account email
    password: str = ""  # Your eSolar/Elekeeper account password
    device_sn: str = ""  # Your inverter serial number (from the device label or app)
    plant_id: str = ""  # Optional - not used in current implementation


@dataclass
class SAJModbusConfig:
    """SAJ Modbus TCP configuration."""
    host: str = "192.168.1.100"
    port: int = 502
    slave_id: int = 1


@dataclass
class SolarConfig:
    """Solar/inverter integration configuration."""
    enabled: bool = False
    source: str = "saj_cloud"  # Options: saj_cloud, saj_modbus, mqtt, mock

    # SAJ Cloud configuration
    saj_cloud: SAJCloudConfig = field(default_factory=SAJCloudConfig)

    # SAJ Modbus configuration (local)
    saj_modbus: SAJModbusConfig = field(default_factory=SAJModbusConfig)

    # MQTT configuration (if using external solar data via MQTT)
    mqtt_topic: str = "solar/production"

    # Solar charging settings
    min_solar_power_w: int = 1000  # Minimum solar power before enabling charging
    buffer_power_w: int = 200  # Buffer to ensure we don't import from grid
    update_interval_seconds: int = 30  # How often to check solar production


class AppConfig:
    """
    Main application configuration.

    Loads configuration from:
    1. config.yaml (if exists)
    2. Environment variables (overrides YAML values)
    """

    def __init__(self, config_file: str | None = None):
        """
        Initialize configuration.

        Args:
            config_file: Path to config.yaml. If None, searches default locations.
        """
        # Initialize with defaults
        self.easee = EaseeConfig()
        self.mqtt = MQTTConfig()
        self.saveeye = SaveEyeConfig()
        self.home = HomeElectricalConfig()
        self.price = PriceConfig()
        self.grid_tariff = GridTariffConfig()
        self.charging = ChargingConfig()
        self.smart_charging = SmartChargingConfig()
        self.web = WebConfig()
        self.logging = LoggingConfig()
        self.database = DatabaseConfig()
        self.solar = SolarConfig()

        # Load from YAML
        yaml_data = self._load_yaml(config_file)
        if yaml_data:
            self._apply_yaml(yaml_data)

        # Apply environment variable overrides
        self._apply_env_overrides()

    def _load_yaml(self, config_file: str | None) -> dict[str, Any] | None:
        """Load configuration from YAML file."""
        search_paths = [
            config_file,
            "config.yaml",
            "config.yml",
            "/app/config/config.yaml",  # Docker/NAS mount point
            "/app/config.yaml",
            "/etc/easee-controller/config.yaml",
            str(Path.home() / ".config" / "easee-controller" / "config.yaml"),
        ]

        for path in search_paths:
            if path and Path(path).exists():
                try:
                    with open(path) as f:
                        data = yaml.safe_load(f)
                        print(f"Loaded configuration from {path}")
                        return data
                except Exception as e:
                    print(f"Warning: Failed to load {path}: {e}")

        return None

    def _apply_yaml(self, data: dict[str, Any]) -> None:
        """Apply YAML configuration data."""
        if not data:
            return

        # EASEE
        if "easee" in data:
            e = data["easee"]
            self.easee.username = e.get("username", self.easee.username)
            self.easee.password = e.get("password", self.easee.password)
            self.easee.charger_id = e.get("charger_id", self.easee.charger_id)
            self.easee.max_current = e.get("max_current", self.easee.max_current)

        # MQTT
        if "mqtt" in data:
            m = data["mqtt"]
            self.mqtt.host = m.get("host", self.mqtt.host)
            self.mqtt.port = m.get("port", self.mqtt.port)
            self.mqtt.username = m.get("username") or None
            self.mqtt.password = m.get("password") or None
            self.mqtt.client_id = m.get("client_id", self.mqtt.client_id)
            self.mqtt.use_tls = m.get("use_tls", self.mqtt.use_tls)

        # SaveEye
        if "saveeye" in data:
            s = data["saveeye"]
            self.saveeye.enabled = s.get("enabled", self.saveeye.enabled)
            self.saveeye.mode = s.get("mode", self.saveeye.mode)
            self.saveeye.mqtt_topic = s.get("mqtt_topic", self.saveeye.mqtt_topic)
            self.saveeye.required = s.get("required", self.saveeye.required)
            if "http" in s:
                h = s["http"]
                self.saveeye.http.host = h.get("host", self.saveeye.http.host)
                self.saveeye.http.port = h.get("port", self.saveeye.http.port)
                self.saveeye.http.poll_interval = h.get("poll_interval", self.saveeye.http.poll_interval)

        # Home
        if "home" in data:
            h = data["home"]
            self.home.main_fuse_amps = h.get("main_fuse_amps", self.home.main_fuse_amps)
            self.home.safety_margin_amps = h.get("safety_margin_amps", self.home.safety_margin_amps)
            self.home.phases = h.get("phases", self.home.phases)
            self.home.voltage = h.get("voltage", self.home.voltage)

        # Price
        if "price" in data:
            p = data["price"]
            self.price.source = p.get("source", self.price.source)
            self.price.nordpool_area = p.get("nordpool_area", self.price.nordpool_area)
            self.price.tibber_access_token = p.get("tibber_access_token") or None
            self.price.currency = p.get("currency", self.price.currency)
            self.price.vat_percent = p.get("vat_percent", self.price.vat_percent)

        # Grid Tariff
        if "grid_tariff" in data:
            gt = data["grid_tariff"]
            self.grid_tariff.enabled = gt.get("enabled", self.grid_tariff.enabled)
            self.grid_tariff.day_rate_ore = gt.get("day_rate_ore", self.grid_tariff.day_rate_ore)
            self.grid_tariff.night_rate_ore = gt.get("night_rate_ore", self.grid_tariff.night_rate_ore)
            self.grid_tariff.night_start_hour = gt.get("night_start_hour", self.grid_tariff.night_start_hour)
            self.grid_tariff.night_end_hour = gt.get("night_end_hour", self.grid_tariff.night_end_hour)

        # Charging
        if "charging" in data:
            ch = data["charging"]
            self.charging.target_kwh = ch.get("target_kwh", self.charging.target_kwh)
            self.charging.charging_power_kw = ch.get("charging_power_kw", self.charging.charging_power_kw)

        # Smart Charging
        if "smart_charging" in data:
            sc = data["smart_charging"]
            self.smart_charging.enabled = sc.get("enabled", self.smart_charging.enabled)
            self.smart_charging.price_threshold = sc.get("price_threshold", self.smart_charging.price_threshold)
            self.smart_charging.lookahead_hours = sc.get("lookahead_hours", self.smart_charging.lookahead_hours)
            self.smart_charging.min_soc_target = sc.get("min_soc_target", self.smart_charging.min_soc_target)
            self.smart_charging.preferred_hours = sc.get("preferred_hours", self.smart_charging.preferred_hours)

        # Web
        if "web" in data:
            w = data["web"]
            self.web.host = w.get("host", self.web.host)
            self.web.port = w.get("port", self.web.port)
            if "auth" in w:
                a = w["auth"]
                self.web.auth.enabled = a.get("enabled", self.web.auth.enabled)
                self.web.auth.username = a.get("username", self.web.auth.username)
                self.web.auth.password = a.get("password", self.web.auth.password)

        # Logging
        if "logging" in data:
            log_cfg = data["logging"]
            self.logging.level = log_cfg.get("level", self.logging.level)
            self.logging.file = log_cfg.get("file") or None

        # Database
        if "database" in data:
            d = data["database"]
            self.database.path = d.get("path", self.database.path)

        # Solar
        if "solar" in data:
            s = data["solar"]
            self.solar.enabled = s.get("enabled", self.solar.enabled)
            self.solar.source = s.get("source", self.solar.source)
            self.solar.mqtt_topic = s.get("mqtt_topic", self.solar.mqtt_topic)
            self.solar.min_solar_power_w = s.get("min_solar_power_w", self.solar.min_solar_power_w)
            self.solar.buffer_power_w = s.get("buffer_power_w", self.solar.buffer_power_w)
            self.solar.update_interval_seconds = s.get("update_interval_seconds", self.solar.update_interval_seconds)
            if "saj_cloud" in s:
                c = s["saj_cloud"]
                self.solar.saj_cloud.username = c.get("username", self.solar.saj_cloud.username)
                self.solar.saj_cloud.password = c.get("password", self.solar.saj_cloud.password)
                self.solar.saj_cloud.plant_id = c.get("plant_id", self.solar.saj_cloud.plant_id)
                self.solar.saj_cloud.device_sn = c.get("device_sn", self.solar.saj_cloud.device_sn)
            if "saj_modbus" in s:
                m = s["saj_modbus"]
                self.solar.saj_modbus.host = m.get("host", self.solar.saj_modbus.host)
                self.solar.saj_modbus.port = m.get("port", self.solar.saj_modbus.port)
                self.solar.saj_modbus.slave_id = m.get("slave_id", self.solar.saj_modbus.slave_id)

    def _apply_env_overrides(self) -> None:
        """Apply environment variable overrides."""
        # EASEE
        self.easee.username = os.environ.get("EASEE_USERNAME", self.easee.username)
        self.easee.password = os.environ.get("EASEE_PASSWORD", self.easee.password)
        self.easee.charger_id = os.environ.get("EASEE_CHARGER_ID", self.easee.charger_id)
        if os.environ.get("EASEE_MAX_CURRENT"):
            self.easee.max_current = int(os.environ["EASEE_MAX_CURRENT"])

        # MQTT
        self.mqtt.host = os.environ.get("MQTT_HOST", self.mqtt.host)
        if os.environ.get("MQTT_PORT"):
            self.mqtt.port = int(os.environ["MQTT_PORT"])
        self.mqtt.username = os.environ.get("MQTT_USERNAME") or self.mqtt.username
        self.mqtt.password = os.environ.get("MQTT_PASSWORD") or self.mqtt.password
        self.mqtt.client_id = os.environ.get("MQTT_CLIENT_ID", self.mqtt.client_id)
        if os.environ.get("MQTT_USE_TLS"):
            self.mqtt.use_tls = os.environ["MQTT_USE_TLS"].lower() in ("true", "1", "yes")

        # SaveEye
        self.saveeye.mode = os.environ.get("SAVEEYE_MODE", self.saveeye.mode)
        self.saveeye.mqtt_topic = os.environ.get("SAVEEYE_MQTT_TOPIC", self.saveeye.mqtt_topic)
        if os.environ.get("SAVEEYE_REQUIRED"):
            self.saveeye.required = os.environ["SAVEEYE_REQUIRED"].lower() in ("true", "1", "yes")
        self.saveeye.http.host = os.environ.get("SAVEEYE_HOST", self.saveeye.http.host)
        if os.environ.get("SAVEEYE_PORT"):
            self.saveeye.http.port = int(os.environ["SAVEEYE_PORT"])
        if os.environ.get("SAVEEYE_POLL_INTERVAL"):
            self.saveeye.http.poll_interval = int(os.environ["SAVEEYE_POLL_INTERVAL"])

        # Home
        if os.environ.get("HOME_MAIN_FUSE_AMPS"):
            self.home.main_fuse_amps = int(os.environ["HOME_MAIN_FUSE_AMPS"])
        if os.environ.get("HOME_SAFETY_MARGIN_AMPS"):
            self.home.safety_margin_amps = int(os.environ["HOME_SAFETY_MARGIN_AMPS"])
        if os.environ.get("HOME_PHASES"):
            self.home.phases = int(os.environ["HOME_PHASES"])
        if os.environ.get("HOME_VOLTAGE"):
            self.home.voltage = int(os.environ["HOME_VOLTAGE"])

        # Price
        self.price.source = os.environ.get("PRICE_SOURCE", self.price.source)
        self.price.nordpool_area = os.environ.get("NORDPOOL_AREA", self.price.nordpool_area)
        self.price.tibber_access_token = os.environ.get("TIBBER_ACCESS_TOKEN") or self.price.tibber_access_token
        self.price.currency = os.environ.get("PRICE_CURRENCY", self.price.currency)
        if os.environ.get("PRICE_VAT_PERCENT"):
            self.price.vat_percent = float(os.environ["PRICE_VAT_PERCENT"])

        # Grid Tariff
        if os.environ.get("GRID_TARIFF_ENABLED"):
            self.grid_tariff.enabled = os.environ["GRID_TARIFF_ENABLED"].lower() in ("true", "1", "yes")
        if os.environ.get("GRID_TARIFF_DAY_RATE_ORE"):
            self.grid_tariff.day_rate_ore = float(os.environ["GRID_TARIFF_DAY_RATE_ORE"])
        if os.environ.get("GRID_TARIFF_NIGHT_RATE_ORE"):
            self.grid_tariff.night_rate_ore = float(os.environ["GRID_TARIFF_NIGHT_RATE_ORE"])
        if os.environ.get("GRID_TARIFF_NIGHT_START_HOUR"):
            self.grid_tariff.night_start_hour = int(os.environ["GRID_TARIFF_NIGHT_START_HOUR"])
        if os.environ.get("GRID_TARIFF_NIGHT_END_HOUR"):
            self.grid_tariff.night_end_hour = int(os.environ["GRID_TARIFF_NIGHT_END_HOUR"])

        # Smart Charging
        if os.environ.get("SMART_CHARGING_ENABLED"):
            self.smart_charging.enabled = os.environ["SMART_CHARGING_ENABLED"].lower() in ("true", "1", "yes")
        if os.environ.get("SMART_CHARGING_PRICE_THRESHOLD"):
            self.smart_charging.price_threshold = float(os.environ["SMART_CHARGING_PRICE_THRESHOLD"])
        if os.environ.get("SMART_CHARGING_LOOKAHEAD_HOURS"):
            self.smart_charging.lookahead_hours = int(os.environ["SMART_CHARGING_LOOKAHEAD_HOURS"])
        if os.environ.get("SMART_CHARGING_PREFERRED_CHARGING_HOURS"):
            hours_str = os.environ["SMART_CHARGING_PREFERRED_CHARGING_HOURS"]
            self.smart_charging.preferred_hours = [int(h.strip()) for h in hours_str.split(",")]

        # Web
        self.web.host = os.environ.get("WEB_HOST", self.web.host)
        if os.environ.get("WEB_PORT"):
            self.web.port = int(os.environ["WEB_PORT"])
        if os.environ.get("WEB_AUTH_ENABLED"):
            self.web.auth.enabled = os.environ["WEB_AUTH_ENABLED"].lower() in ("true", "1", "yes")
        self.web.auth.username = os.environ.get("WEB_AUTH_USERNAME", self.web.auth.username)
        self.web.auth.password = os.environ.get("WEB_AUTH_PASSWORD", self.web.auth.password)

        # Logging
        self.logging.level = os.environ.get("LOG_LEVEL", self.logging.level)
        self.logging.file = os.environ.get("LOG_FILE") or self.logging.file

        # Database
        self.database.path = os.environ.get("DATABASE_PATH", self.database.path)

        # Solar
        if os.environ.get("SOLAR_ENABLED"):
            self.solar.enabled = os.environ["SOLAR_ENABLED"].lower() in ("true", "1", "yes")
        self.solar.source = os.environ.get("SOLAR_SOURCE", self.solar.source)
        self.solar.mqtt_topic = os.environ.get("SOLAR_MQTT_TOPIC", self.solar.mqtt_topic)
        if os.environ.get("SOLAR_MIN_POWER_W"):
            self.solar.min_solar_power_w = int(os.environ["SOLAR_MIN_POWER_W"])
        if os.environ.get("SOLAR_BUFFER_POWER_W"):
            self.solar.buffer_power_w = int(os.environ["SOLAR_BUFFER_POWER_W"])
        self.solar.saj_cloud.username = os.environ.get("SAJ_USERNAME", self.solar.saj_cloud.username)
        self.solar.saj_cloud.password = os.environ.get("SAJ_PASSWORD", self.solar.saj_cloud.password)
        self.solar.saj_cloud.plant_id = os.environ.get("SAJ_PLANT_ID", self.solar.saj_cloud.plant_id)
        self.solar.saj_cloud.device_sn = os.environ.get("SAJ_DEVICE_SN", self.solar.saj_cloud.device_sn)
        self.solar.saj_modbus.host = os.environ.get("SAJ_MODBUS_HOST", self.solar.saj_modbus.host)
        if os.environ.get("SAJ_MODBUS_PORT"):
            self.solar.saj_modbus.port = int(os.environ["SAJ_MODBUS_PORT"])

    def validate(self) -> list[str]:
        """
        Validate configuration.

        Returns:
            List of validation error messages, empty if valid.
        """
        errors = []

        # EASEE credentials
        if not self.easee.username or self.easee.username == "your_email@example.com":
            errors.append("easee.username is not configured")
        if not self.easee.password or self.easee.password == "your_password":
            errors.append("easee.password is not configured")
        if not self.easee.charger_id or self.easee.charger_id == "EHXXXXXX":
            errors.append("easee.charger_id is not configured")

        # Tibber token
        if self.price.source == "tibber" and not self.price.tibber_access_token:
            errors.append("price.tibber_access_token is required when using Tibber")

        # Electrical sanity
        if self.home.safety_margin_amps >= self.home.main_fuse_amps:
            errors.append("home.safety_margin_amps must be less than home.main_fuse_amps")

        # Current limits
        if not 6 <= self.easee.max_current <= 32:
            errors.append("easee.max_current must be between 6 and 32")

        return errors

    def to_dict(self) -> dict[str, Any]:
        """Export configuration as dictionary (for debugging)."""
        return {
            "easee": {
                "username": self.easee.username,
                "charger_id": self.easee.charger_id,
                "max_current": self.easee.max_current,
            },
            "mqtt": {
                "host": self.mqtt.host,
                "port": self.mqtt.port,
                "client_id": self.mqtt.client_id,
            },
            "saveeye": {
                "mode": self.saveeye.mode,
                "mqtt_topic": self.saveeye.mqtt_topic,
                "required": self.saveeye.required,
            },
            "home": {
                "main_fuse_amps": self.home.main_fuse_amps,
                "safety_margin_amps": self.home.safety_margin_amps,
                "phases": self.home.phases,
            },
            "price": {
                "source": self.price.source,
                "nordpool_area": self.price.nordpool_area,
                "currency": self.price.currency,
            },
            "grid_tariff": {
                "enabled": self.grid_tariff.enabled,
                "day_rate_ore": self.grid_tariff.day_rate_ore,
                "night_rate_ore": self.grid_tariff.night_rate_ore,
                "night_hours": f"{self.grid_tariff.night_start_hour}:00-{self.grid_tariff.night_end_hour}:00",
            },
            "smart_charging": {
                "enabled": self.smart_charging.enabled,
                "price_threshold": self.smart_charging.price_threshold,
                "preferred_hours": self.smart_charging.preferred_hours,
            },
            "web": {
                "host": self.web.host,
                "port": self.web.port,
            },
            "solar": {
                "enabled": self.solar.enabled,
                "source": self.solar.source,
                "min_solar_power_w": self.solar.min_solar_power_w,
            },
        }


# Global configuration instance
_config: AppConfig | None = None


def get_config(config_file: str | None = None) -> AppConfig:
    """
    Get the application configuration singleton.

    Args:
        config_file: Path to config.yaml (only used on first call).

    Returns:
        Application configuration instance.
    """
    global _config
    if _config is None:
        _config = AppConfig(config_file)
    return _config


def reload_config(config_file: str | None = None) -> AppConfig:
    """
    Reload the application configuration.

    Args:
        config_file: Path to config.yaml.

    Returns:
        New application configuration instance.
    """
    global _config
    _config = AppConfig(config_file)
    return _config


def load_config(config_file: str | None = None) -> AppConfig:
    """
    Load configuration from a specific file.

    Args:
        config_file: Path to config.yaml.

    Returns:
        Application configuration instance.
    """
    return reload_config(config_file)
