"""
SAJ Solar Monitor - Fetches data from SAJ eSolar/Elekeeper Cloud API.

Uses eop.saj-electric.com with AES encryption and signatures.
Based on reverse-engineered SAJ web app API.
"""

import asyncio
import hashlib
import logging
import random
import string
from dataclasses import dataclass, field
from datetime import datetime, timezone

import aiohttp
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

from .config import get_config

logger = logging.getLogger(__name__)

# SAJ API Constants (reverse-engineered from SAJ web app)
BASE_URL = "https://eop.saj-electric.com"
# AES-128 key: 32 hex chars = 16 bytes (parsed as HEX, not UTF-8!)
AES_KEY_HEX = 'ec1840a7c53cf0709eb784be480379b6'
AES_KEY = bytes.fromhex(AES_KEY_HEX)  # 16 bytes for AES-128
SECRET_KEY = 'ktoKRLgQPjvNyUZO8lVc9kU1Bsip6XIe'
CLIENT_ID = 'esolar-monitor-admin'
APP_PROJECT = 'elekeeper'


def encrypt_password(password: str) -> str:
    """
    Encrypt password using AES-128-ECB with PKCS7 padding.
    
    Returns hex string (matching SAJ web app crypto-js implementation).
    """
    # PKCS7 padding
    block_size = 16
    padding_len = block_size - (len(password) % block_size)
    padded_password = password.encode('utf-8') + bytes([padding_len] * padding_len)

    # AES-ECB encryption with 16-byte key (AES-128)
    cipher = Cipher(
        algorithms.AES(AES_KEY),
        modes.ECB(),
        backend=default_backend()
    )
    encryptor = cipher.encryptor()
    encrypted = encryptor.update(padded_password) + encryptor.finalize()

    # Return as hex string (NOT base64!) - this is what crypto-js ciphertext.toString() does
    return encrypted.hex()


def generate_random_string(length: int = 32) -> str:
    """Generate random alphanumeric string."""
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))


def generate_signature(params: dict) -> str:
    """
    Generate API signature using MD5 -> SHA1 chain.

    Algorithm:
    1. Sort parameters alphabetically
    2. Concatenate as key=value&key=value...&key=SECRET_KEY
    3. MD5 hash the string
    4. SHA1 hash the MD5 result
    5. Return uppercase hex
    """
    # Sort keys alphabetically
    sorted_keys = sorted(params.keys())

    # Build parameter string
    param_str = '&'.join(f'{k}={params[k]}' for k in sorted_keys)
    param_str += f'&key={SECRET_KEY}'

    # MD5 -> SHA1 -> UPPERCASE
    md5_hash = hashlib.md5(param_str.encode('utf-8')).hexdigest()
    sha1_hash = hashlib.sha1(md5_hash.encode('utf-8')).hexdigest().upper()

    return sha1_hash


def build_request_params(extra_params: dict | None = None) -> dict:
    """Build common request parameters with signature."""
    timestamp = str(int(datetime.now().timestamp() * 1000))
    random_str = generate_random_string(32)
    client_date = datetime.now().strftime('%Y-%m-%d')

    params = {
        'appProjectName': APP_PROJECT,
        'clientDate': client_date,
        'lang': 'en',
        'timeStamp': timestamp,
        'random': random_str,
        'clientId': CLIENT_ID,
    }

    if extra_params:
        params.update(extra_params)

    # Generate signature from ALL params
    signature = generate_signature(params)

    # Add signature metadata
    params['signParams'] = ','.join(sorted(params.keys()))
    params['signature'] = signature

    return params


@dataclass
class SolarStatus:
    """Current solar system status."""

    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    is_online: bool = False

    # Power values in Watts
    pv_power: float = 0.0  # Solar panel production
    grid_power: float = 0.0  # Grid import (+) or export (-)
    load_power: float = 0.0  # Home consumption
    battery_power: float = 0.0  # Battery charge (-) or discharge (+)

    # Battery status
    battery_soc: float = 0.0  # State of charge (%)

    # Daily totals (kWh)
    pv_today_kwh: float = 0.0

    @property
    def is_exporting(self) -> bool:
        """Check if exporting to grid (negative grid_power)."""
        return self.grid_power < -50  # Allow 50W tolerance

    @property
    def available_solar_power(self) -> float:
        """Power available for EV charging (excess solar)."""
        if self.grid_power < 0:
            return abs(self.grid_power)
        return 0.0


class SAJMonitor:
    """
    Monitor for SAJ eSolar/Elekeeper Cloud API.

    Uses eop.saj-electric.com with AES encryption and signatures.
    """

    LOGIN_URL = f"{BASE_URL}/dev-api/api/v1/sys/login"
    CAPTCHA_URL = f"{BASE_URL}/dev-api/api/v1/sys/getCaptchaBase64"
    DEVICE_INFO_URL = f"{BASE_URL}/dev-api/api/v1/monitor/device/getOneDeviceInfo"

    def __init__(self) -> None:
        config = get_config()
        self._username = config.solar.saj_cloud.username
        self._password = config.solar.saj_cloud.password
        self._device_sn = config.solar.saj_cloud.device_sn
        self._enabled = config.solar.enabled and config.solar.source == "saj_cloud"

        self._session: aiohttp.ClientSession | None = None
        self._token: str | None = None

        self._last_status: SolarStatus = SolarStatus()
        self._running = False
        self._monitor_task: asyncio.Task | None = None
        self._update_interval = config.solar.update_interval_seconds

        # Login state tracking
        self._login_error: str | None = None
        self._needs_captcha: bool = False
        self._captcha_key: str | None = None
        self._captcha_image: str | None = None  # Base64 encoded image

    @property
    def enabled(self) -> bool:
        """Check if SAJ monitoring is enabled."""
        return self._enabled

    @property
    def last_status(self) -> SolarStatus:
        """Get last known solar status."""
        return self._last_status

    @property
    def login_error(self) -> str | None:
        """Get current login error message, if any."""
        return self._login_error

    @property
    def needs_captcha(self) -> bool:
        """Check if login requires CAPTCHA verification."""
        return self._needs_captcha

    @property
    def is_connected(self) -> bool:
        """Check if connected and authenticated."""
        return self._running and self._token is not None

    @property
    def captcha_image(self) -> str | None:
        """Get current CAPTCHA image (base64 encoded)."""
        return self._captcha_image

    async def get_captcha(self) -> dict:
        """
        Fetch a new CAPTCHA image from SAJ.
        
        Returns dict with:
        - success: bool
        - image: base64 string (if success)
        - error: string (if failure)
        """
        if not self._session:
            self._session = aiohttp.ClientSession()

        captcha_params = {
            'loginName': self._username,
            'type': 'pwdLogin',
        }
        params = build_request_params(captcha_params)
        headers = {'lang': 'en', 'accept': 'application/json'}

        try:
            async with self._session.get(
                self.CAPTCHA_URL,
                params=params,
                headers=headers,
            ) as resp:
                if resp.status != 200:
                    return {"success": False, "error": f"HTTP {resp.status}"}

                data = await resp.json()
                code = data.get('code') or data.get('errCode')

                if code == 200 or code == 0:
                    captcha_data = data.get('data', {})
                    self._captcha_key = captcha_data.get('captchaKey')
                    self._captcha_image = captcha_data.get('captchaBase64')
                    self._needs_captcha = True
                    logger.info(f"CAPTCHA fetched successfully, key={self._captcha_key[:20] if self._captcha_key else 'None'}...")
                    return {
                        "success": True,
                        "image": self._captcha_image,
                    }
                else:
                    error_msg = data.get('msg') or data.get('errMsg') or 'Unknown error'
                    return {"success": False, "error": error_msg}

        except Exception as e:
            logger.error(f"Failed to fetch CAPTCHA: {e}")
            return {"success": False, "error": str(e)}

    async def login_with_captcha(self, captcha_code: str) -> bool:
        """
        Login with CAPTCHA code.
        
        Args:
            captcha_code: The user-entered CAPTCHA text
            
        Returns:
            True if login successful, False otherwise
        """
        if not self._captcha_key:
            self._login_error = "No CAPTCHA key - fetch CAPTCHA first"
            return False

        if not self._session:
            self._session = aiohttp.ClientSession()

        # Encrypt password
        encrypted_password = encrypt_password(self._password)

        # Build login params with CAPTCHA
        # Note: From JS reverse-engineering, the CAPTCHA code param is 'code'
        # and the key param is 'captchaKey' (matching the getCaptchaBase64 response)
        # Note: JS deletes rememberMe, confirmPassword, uuid before signing
        login_params = {
            'username': self._username,
            'password': encrypted_password,
            'loginType': '1',
            'code': captcha_code,  # The CAPTCHA text entered by user
            'captchaKey': self._captcha_key,  # The CAPTCHA key from getCaptchaBase64 response
        }

        params = build_request_params(login_params)
        headers = {
            'Content-Type': 'application/x-www-form-urlencoded;charset=UTF-8',
            'lang': 'en',
        }

        logger.info(f"SAJ login with CAPTCHA: username={self._username}, captchaKey={self._captcha_key[:20] if self._captcha_key else 'None'}..., code={captcha_code}")

        try:
            async with self._session.post(
                self.LOGIN_URL,
                data=params,
                headers=headers,
            ) as resp:
                if resp.status != 200:
                    self._login_error = f"HTTP {resp.status}"
                    return False

                data = await resp.json()
                code = data.get('code') or data.get('errCode')

                if code != 200 and code != 0:
                    error_msg = data.get('msg') or data.get('errMsg') or 'Unknown error'
                    self._login_error = error_msg

                    # Check if still needs CAPTCHA (wrong code)
                    if 'captcha' in error_msg.lower():
                        self._needs_captcha = True
                        # Don't auto-fetch new CAPTCHA - let user retry or manually refresh

                    logger.error(f"SAJ login with CAPTCHA failed: {error_msg}")
                    return False

                # Success!
                self._login_error = None
                self._needs_captcha = False
                self._captcha_key = None
                self._captcha_image = None
                self._token = data['data']['token']
                expires_in = data['data'].get('expiresIn', 0)

                logger.info(f"SAJ login successful! Token expires in {expires_in // 3600} hours")

                # Start monitoring if not already running
                self._running = True
                if not self._monitor_task or self._monitor_task.done():
                    self._monitor_task = asyncio.create_task(self._monitor_loop())

                return True

        except Exception as e:
            self._login_error = str(e)
            logger.error(f"SAJ login with CAPTCHA error: {e}")
            return False

    async def start(self) -> None:
        """Start monitoring."""
        if not self._enabled:
            logger.info("SAJ monitor disabled")
            return

        if self._running:
            return

        self._session = aiohttp.ClientSession()

        # Try initial login
        try:
            success = await self._login()
            if success:
                logger.info("SAJ monitor started successfully")
                self._running = True
                self._monitor_task = asyncio.create_task(self._monitor_loop())
            else:
                # If CAPTCHA required, fetch it for the GUI
                if self._needs_captcha:
                    logger.info("CAPTCHA required - fetching for GUI")
                    await self.get_captcha()
                logger.warning("SAJ monitor needs CAPTCHA - waiting for user input")
        except Exception as e:
            logger.error(f"Failed to start SAJ monitor: {e}")
            self._login_error = str(e)

    async def stop(self) -> None:
        """Stop monitoring."""
        self._running = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
            self._monitor_task = None

        if self._session:
            await self._session.close()
            self._session = None

        logger.info("SAJ monitor stopped")

    async def retry_login(self) -> bool:
        """Retry login after user clears CAPTCHA via web browser."""
        logger.info("Retrying SAJ login...")
        self._login_error = None
        self._needs_captcha = False

        # Create session if needed
        if not self._session:
            self._session = aiohttp.ClientSession()

        success = await self._login()
        if success:
            logger.info("SAJ retry login successful!")
            self._running = True
            if not self._monitor_task or self._monitor_task.done():
                self._monitor_task = asyncio.create_task(self._monitor_loop())
        return success

    async def _login(self) -> bool:
        """Login to SAJ eSolar Cloud."""
        if not self._session:
            return False

        # Encrypt password
        encrypted_password = encrypt_password(self._password)

        # Build login params (these get included in signature)
        # Note: JS deletes rememberMe, confirmPassword, uuid before signing
        login_params = {
            'username': self._username,
            'password': encrypted_password,
            'loginType': '1',
        }

        # Get signed params (signature includes ALL params)
        params = build_request_params(login_params)

        headers = {
            'Content-Type': 'application/x-www-form-urlencoded;charset=UTF-8',
            'lang': 'en',
        }

        logger.debug(f"SAJ login: username={self._username}")

        try:
            async with self._session.post(
                self.LOGIN_URL,
                data=params,
                headers=headers,
            ) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    self._login_error = f"HTTP {resp.status}"
                    logger.error(f"SAJ login failed: HTTP {resp.status}, {text[:200]}")
                    return False

                data = await resp.json()
                logger.info(f"SAJ login response: {data}")

                # Handle both 'code' and 'errCode' response formats
                code = data.get('code')
                if code is None:
                    code = data.get('errCode')

                if code != 200 and code != 0:
                    error_msg = data.get('msg') or data.get('errMsg') or 'Unknown error'
                    self._login_error = error_msg

                    # Check for CAPTCHA requirement - error code 10003 is auth error
                    # which usually requires CAPTCHA (especially after failed attempts)
                    if 'captcha' in error_msg.lower() or code == 10003:
                        self._needs_captcha = True
                        logger.warning(
                            "SAJ requires CAPTCHA. Please login at "
                            "https://eop.saj-electric.com/login and retry."
                        )

                    logger.error(f"SAJ login failed (code={code}): {error_msg}")
                    return False

                # Success - clear errors
                self._login_error = None
                self._needs_captcha = False

                # Extract token
                self._token = data['data']['token']
                expires_in = data['data'].get('expiresIn', 0)

                logger.info(f"SAJ login successful! Token expires in {expires_in // 3600} hours")
                return True

        except Exception as e:
            self._login_error = str(e)
            logger.error(f"SAJ login error: {e}")
            return False

    async def _fetch_data(self) -> SolarStatus | None:
        """Fetch current solar data."""
        if not self._session or not self._token:
            return None

        # Build params with device SN
        params = build_request_params({'deviceSn': self._device_sn})

        headers = {
            'Authorization': f'Bearer {self._token}',
            'lang': 'en',
        }

        try:
            async with self._session.get(
                self.DEVICE_INFO_URL,
                params=params,
                headers=headers,
            ) as resp:
                if resp.status != 200:
                    logger.warning(f"SAJ data fetch failed: HTTP {resp.status}")
                    return None

                data = await resp.json()

                # Handle both 'code' and 'errCode' response formats
                code = data.get('code')
                if code is None:
                    code = data.get('errCode')

                if code != 200 and code != 0:
                    error_msg = data.get('msg') or data.get('errMsg') or 'Unknown error'
                    # Check if token expired
                    if 'token' in error_msg.lower() or 'login' in error_msg.lower():
                        logger.info("SAJ session expired, re-logging in...")
                        await self._login()
                    else:
                        logger.warning(f"SAJ data fetch error: {error_msg}")
                    return None

                return self._parse_data(data['data'])

        except Exception as e:
            logger.warning(f"SAJ data fetch error: {e}")
            return None

    def _parse_data(self, device_data: dict) -> SolarStatus:
        """Parse and format power data from device response."""
        stats = device_data.get('deviceStatisticsData', {})

        # Extract raw values using actual API field names
        # (field names reverse-engineered from SAJ web app)
        pv_power = float(stats.get('powerNow', 0) or 0)  # Current PV power in W
        load_power = float(stats.get('totalLoadPowerwatt', 0) or 0)  # Home consumption in W
        grid_power = float(stats.get('sysGridPowerwatt', 0) or 0)  # Grid power in W (+ = import)
        battery_power = float(stats.get('batPower', 0) or 0)  # Battery power (+ = charging)
        soc = float(stats.get('batCapcity', 0) or 0)  # Battery SOC % (note: API typo)
        today_kwh = float(stats.get('todayPvEnergy', 0) or 0)  # Today's PV in kWh
        is_online = stats.get('isOnline', 0) == 1

        # Convert battery sign: SAJ uses + for charging, we use + for discharging
        # So negate it to match our convention
        battery_power = -battery_power

        return SolarStatus(
            timestamp=datetime.now(timezone.utc),
            is_online=is_online,
            pv_power=pv_power,
            grid_power=grid_power,
            load_power=load_power,
            battery_power=battery_power,
            battery_soc=soc,
            pv_today_kwh=today_kwh,
        )

    async def _monitor_loop(self) -> None:
        """Background monitoring loop."""
        while self._running:
            try:
                status = await self._fetch_data()
                if status:
                    self._last_status = status
                    logger.debug(
                        f"SAJ: PV={status.pv_power:.0f}W, "
                        f"Grid={status.grid_power:.0f}W, "
                        f"Load={status.load_power:.0f}W, "
                        f"Battery={status.battery_soc:.0f}%"
                    )
                else:
                    # Mark as offline if we couldn't fetch data
                    self._last_status.is_online = False
            except Exception as e:
                logger.warning(f"SAJ monitor error: {e}")
                self._last_status.is_online = False

            await asyncio.sleep(self._update_interval)

    def get_status(self) -> dict:
        """Get solar status for API."""
        s = self._last_status
        return {
            "enabled": self._enabled,
            "is_online": s.is_online and self.is_connected,
            "error": self._login_error,
            "needs_captcha": self._needs_captcha,
            "captcha_image": self._captcha_image,
            "timestamp": s.timestamp.isoformat(),
            "power": {
                "pv_watts": s.pv_power,
                "grid_watts": s.grid_power,
                "load_watts": s.load_power,
                "battery_watts": s.battery_power,
            },
            "battery": {
                "soc_percent": s.battery_soc,
            },
            "today_kwh": {
                "pv": s.pv_today_kwh,
            },
            "is_exporting": s.is_exporting,
            "available_for_ev_watts": s.available_solar_power,
        }


# Singleton instance
_saj_monitor: SAJMonitor | None = None


def get_saj_monitor() -> SAJMonitor:
    """Get or create SAJ monitor singleton."""
    global _saj_monitor
    if _saj_monitor is None:
        _saj_monitor = SAJMonitor()
    return _saj_monitor
