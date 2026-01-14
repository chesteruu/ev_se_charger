"""
Web Application for FSM-based Charging Controller.

Frontend is a VIEWER with limited control:
- View status (read-only)
- Change mode (triggers FSM switch)
- Start charging
- Stop charging (always allowed for safety)

All control goes through FSMManager - no direct charger control from GUI.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

if TYPE_CHECKING:
    from ..easee_controller import EaseeController
    from ..fsm_manager import FSMManager
    from ..load_monitor import LoadMonitor
    from ..price_fetcher import PriceFetcher
    from ..schedule_provider import ScheduleProvider

logger = logging.getLogger(__name__)


class ModeChangeRequest(BaseModel):
    """Request to change charging mode."""

    mode: str  # "smart", "immediate", "manual", "solar", "scheduled"


class CaptchaLoginRequest(BaseModel):
    """Request to login with CAPTCHA."""

    captcha_code: str


class TargetKwhRequest(BaseModel):
    """Request to set target kWh."""

    target_kwh: float


def create_app(
    fsm_manager: FSMManager | None = None,
    schedule_provider: ScheduleProvider | None = None,
    load_monitor: LoadMonitor | None = None,
    charger: EaseeController | None = None,
    price_fetcher: PriceFetcher | None = None,
    saj_monitor=None,  # SAJMonitor | None
) -> FastAPI:
    """Create the FastAPI application."""

    app = FastAPI(
        title="EV Charging Controller",
        description="FSM-based EV charging controller",
        version="2.0.0",
    )

    # Store references
    app.state.fsm_manager = fsm_manager
    app.state.schedule_provider = schedule_provider
    app.state.load_monitor = load_monitor
    app.state.charger = charger
    app.state.price_fetcher = price_fetcher
    app.state.saj_monitor = saj_monitor
    app.state.websocket_connections: list[WebSocket] = []
    app.state.broadcast_task = None

    async def broadcast_status() -> None:
        """Background task to broadcast status to all WebSocket clients."""
        while True:
            try:
                if app.state.websocket_connections:
                    status = await get_status()
                    dead_connections = []
                    for ws in app.state.websocket_connections:
                        try:
                            await ws.send_json(status)
                        except Exception:
                            dead_connections.append(ws)
                    # Clean up dead connections
                    for ws in dead_connections:
                        if ws in app.state.websocket_connections:
                            app.state.websocket_connections.remove(ws)
                await asyncio.sleep(1)  # Update every second
            except Exception:
                await asyncio.sleep(1)

    @app.on_event("startup")
    async def start_broadcast():
        """Start the broadcast task."""
        app.state.broadcast_task = asyncio.create_task(broadcast_status())

    @app.on_event("shutdown")
    async def stop_broadcast():
        """Stop the broadcast task."""
        if app.state.broadcast_task:
            app.state.broadcast_task.cancel()
            try:
                await app.state.broadcast_task
            except asyncio.CancelledError:
                pass

    # =========================================================================
    # STATUS ENDPOINTS (Read-only)
    # =========================================================================

    @app.get("/api/status")
    async def get_status() -> dict:
        """Get complete system status."""
        response: dict = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # FSM status (from FSMManager)
        if app.state.fsm_manager:
            response["fsm"] = app.state.fsm_manager.get_status()
        else:
            response["fsm"] = {"state": "unknown", "error": "FSM not initialized"}

        # Schedule status
        if app.state.schedule_provider:
            response["schedule"] = app.state.schedule_provider.get_status()
        else:
            response["schedule"] = {"available": False}

        # Load status
        if app.state.load_monitor:
            response["load"] = app.state.load_monitor.get_status()
        else:
            response["load"] = {"available": False}

        # Charger status
        if app.state.charger and app.state.charger.last_status:
            status = app.state.charger.last_status

            # Note: EASEE sessions API only returns COMPLETED sessions
            # For ongoing sessions, we use FSM timestamps (car_connected_at, charging_started_at)
            # which are already included in the FSM status response

            response["charger"] = {
                "state": status.state.name,
                "is_online": status.is_online,
                "power_watts": status.power_watts,
                "current_amps": status.current_amps,
                "session_energy_kwh": status.energy_session_kwh,
            }
        else:
            response["charger"] = {"available": False}

        # Prices
        if app.state.price_fetcher:
            try:
                from ..config import get_config

                config = get_config()

                forecast = await app.state.price_fetcher.get_prices()
                current = forecast.current_price

                def price_to_dict(price) -> dict | None:
                    if not price:
                        return None
                    return {
                        "start_time": price.start_time.isoformat(),
                        "end_time": price.end_time.isoformat(),
                        "price": price.price,
                        "currency": price.currency,
                        "level": price.level.value,
                    }

                # Get scheduled windows for highlighting
                scheduled_hours = set()
                if app.state.schedule_provider and app.state.schedule_provider.current_plan:
                    for window in app.state.schedule_provider.current_plan.windows:
                        scheduled_hours.add(window.start_time.isoformat())

                # Build hourly data with date info
                # Use price fetcher's timezone (derived from nordpool_area) for local time
                local_tz = app.state.price_fetcher.timezone
                now_local = datetime.now(local_tz)
                hourly_data = []
                for h in sorted(forecast.prices, key=lambda x: x.start_time):
                    local_time = h.start_time.astimezone(local_tz)
                    hour = local_time.hour
                    is_night = config.grid_tariff.is_night_hour(hour)

                    hourly_data.append(
                        {
                            "start": h.start_time.isoformat(),
                            "price_sek": h.price,
                            "is_night_rate": is_night,
                            "hour": hour,
                            "date": local_time.strftime("%m-%d"),
                            "is_today": local_time.date() == now_local.date(),
                            "is_current": h.start_time <= datetime.now(timezone.utc) < h.end_time,
                            "is_scheduled": h.start_time.isoformat() in scheduled_hours,
                        }
                    )

                response["prices"] = {
                    "current": {
                        "price_sek": current.price if current else None,
                        "is_night_rate": current.is_cheap if current else False,
                    }
                    if current
                    else None,
                    "average_24h": forecast.average_price,
                    "min_24h": price_to_dict(forecast.min_price),
                    "max_24h": price_to_dict(forecast.max_price),
                    "hourly": hourly_data,
                }
            except Exception as e:
                response["prices"] = {"error": str(e)}

        # Solar
        if app.state.saj_monitor:
            response["solar"] = app.state.saj_monitor.get_status()
        else:
            response["solar"] = {"enabled": False}

        return response

    @app.get("/api/solar")
    async def get_solar() -> dict:
        """Get solar production data."""
        if not app.state.saj_monitor:
            return {"enabled": False, "message": "Solar monitoring not configured"}
        return app.state.saj_monitor.get_status()

    @app.get("/api/fsm")
    async def get_fsm_status() -> dict:
        """Get FSM state."""
        if not app.state.fsm_manager:
            raise HTTPException(status_code=503, detail="FSM not initialized")
        return app.state.fsm_manager.get_status()

    @app.get("/api/schedule")
    async def get_schedule() -> dict:
        """Get schedule information."""
        if not app.state.schedule_provider:
            raise HTTPException(status_code=503, detail="Schedule not initialized")
        return app.state.schedule_provider.get_status()

    @app.get("/api/load")
    async def get_load() -> dict:
        """Get load information."""
        if not app.state.load_monitor:
            raise HTTPException(status_code=503, detail="Load monitor not initialized")
        return app.state.load_monitor.get_status()

    @app.get("/api/prices")
    async def get_prices() -> dict:
        """Get price forecast with all 48 hours."""
        if not app.state.price_fetcher:
            raise HTTPException(status_code=503, detail="Price fetcher not initialized")

        from ..config import get_config

        config = get_config()

        forecast = await app.state.price_fetcher.get_prices()
        current = forecast.current_price

        # Get scheduled windows for highlighting
        scheduled_hours = set()
        if app.state.schedule_provider and app.state.schedule_provider.current_plan:
            for window in app.state.schedule_provider.current_plan.windows:
                scheduled_hours.add(window.start_time.isoformat())

        # Build hourly data with date info
        # Use price fetcher's timezone (derived from nordpool_area) for local time
        local_tz = app.state.price_fetcher.timezone
        now_local = datetime.now(local_tz)
        hourly_data = []
        for h in sorted(forecast.prices, key=lambda x: x.start_time):
            local_time = h.start_time.astimezone(local_tz)
            hour = local_time.hour
            is_night = config.grid_tariff.is_night_hour(hour)

            hourly_data.append(
                {
                    "start": h.start_time.isoformat(),
                    "end": h.end_time.isoformat(),
                    "price_sek": h.price,
                    "is_night_rate": is_night,
                    "hour": hour,
                    "date": local_time.strftime("%m-%d"),
                    "day_name": local_time.strftime("%a"),
                    "is_today": local_time.date() == now_local.date(),
                    "is_current": h.start_time <= datetime.now(timezone.utc) < h.end_time,
                    "is_scheduled": h.start_time.isoformat() in scheduled_hours,
                    "level": h.level.value,
                }
            )

        return {
            "current": {
                "start": current.start_time.isoformat() if current else None,
                "end": current.end_time.isoformat() if current else None,
                "price_sek": current.price if current else None,
                "is_night_rate": current.is_cheap if current else False,
            }
            if current
            else None,
            "hourly": hourly_data,
            "statistics": {
                "average": forecast.average_price,
                "min": forecast.min_price.price if forecast.min_price else None,
                "max": forecast.max_price.price if forecast.max_price else None,
            },
        }

    # =========================================================================
    # CONTROL ENDPOINTS - All go through FSMManager
    # =========================================================================

    @app.post("/api/mode")
    async def set_mode(request: ModeChangeRequest) -> dict:
        """
        Change charging mode.

        This may trigger an FSM switch:
        - SMART, SCHEDULED, SOLAR -> SmartChargingFSM
        - MANUAL, IMMEDIATE -> ManualChargingFSM
        """
        if not app.state.fsm_manager:
            raise HTTPException(status_code=503, detail="FSM not initialized")

        from ..charging_fsm_base import ChargingMode

        mode_map = {
            "smart": ChargingMode.SMART,
            "immediate": ChargingMode.IMMEDIATE,
            "manual": ChargingMode.MANUAL,
            "solar": ChargingMode.SOLAR,
            "scheduled": ChargingMode.SCHEDULED,
        }

        mode = mode_map.get(request.mode.lower())
        if not mode:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid mode: {request.mode}. Valid: {list(mode_map.keys())}",
            )

        app.state.fsm_manager.set_mode(mode)

        return {
            "status": "ok",
            "mode": mode.value,
            "active_fsm": app.state.fsm_manager.active_fsm.__class__.__name__,
        }

    @app.post("/api/charger/start")
    async def start_charging() -> dict:
        """
        Request to start charging.

        FSMManager routes to active FSM which decides based on state + mode.
        """
        if not app.state.fsm_manager:
            raise HTTPException(status_code=503, detail="FSM not initialized")

        result = await app.state.fsm_manager.request_start()

        if result["success"]:
            return {"status": "ok", "message": result["reason"]}
        raise HTTPException(status_code=400, detail=result["reason"])

    @app.post("/api/charger/stop")
    async def stop_charging() -> dict:
        """
        Request to stop charging.

        FSMManager routes to active FSM which decides based on state + mode.
        """
        if not app.state.fsm_manager:
            raise HTTPException(status_code=503, detail="FSM not initialized")

        result = await app.state.fsm_manager.request_stop()

        if result["success"]:
            return {"status": "ok", "message": result["reason"]}
        raise HTTPException(status_code=400, detail=result["reason"])

    @app.post("/api/charger/resume")
    async def resume_charging() -> dict:
        """
        Request to resume charging.

        FSMManager routes to active FSM which decides based on state + mode.
        """
        if not app.state.fsm_manager:
            raise HTTPException(status_code=503, detail="FSM not initialized")

        result = await app.state.fsm_manager.request_resume()

        if result["success"]:
            return {"status": "ok", "message": result["reason"]}
        raise HTTPException(status_code=400, detail=result["reason"])

    @app.post("/api/schedule/target")
    async def set_target_kwh(request: TargetKwhRequest) -> dict:
        """Set target kWh for charging schedule."""
        if not app.state.schedule_provider:
            raise HTTPException(status_code=503, detail="Schedule provider not initialized")

        try:
            await app.state.schedule_provider.set_target_kwh(request.target_kwh)
            return {
                "status": "ok",
                "target_kwh": request.target_kwh,
            }
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    # =========================================================================
    # SAJ SOLAR
    # =========================================================================

    @app.get("/api/solar/status")
    async def get_solar_status() -> dict:
        """Get SAJ solar monitor status."""
        if not app.state.saj_monitor:
            return {
                "enabled": False,
                "connected": False,
                "error": None,
                "needs_captcha": False,
            }

        saj = app.state.saj_monitor
        return {
            "enabled": saj.enabled,
            "connected": saj.is_connected,
            "error": saj.login_error,
            "needs_captcha": saj.needs_captcha,
            "login_url": "https://eop.saj-electric.com/login",
        }

    @app.post("/api/solar/retry")
    async def retry_solar_login() -> dict:
        """Retry SAJ login (without CAPTCHA)."""
        if not app.state.saj_monitor:
            raise HTTPException(status_code=503, detail="SAJ monitor not configured")

        success = await app.state.saj_monitor.retry_login()
        if success:
            return {"status": "ok", "message": "SAJ login successful"}
        raise HTTPException(status_code=401, detail=app.state.saj_monitor.login_error or "Login failed")

    @app.get("/api/solar/captcha")
    async def get_solar_captcha() -> dict:
        """Get a new CAPTCHA image for SAJ login."""
        if not app.state.saj_monitor:
            raise HTTPException(status_code=503, detail="SAJ monitor not configured")

        result = await app.state.saj_monitor.get_captcha()
        if result["success"]:
            return {
                "status": "ok",
                "image": result["image"],
            }
        raise HTTPException(status_code=500, detail=result.get("error", "Failed to get CAPTCHA"))

    @app.post("/api/solar/login")
    async def solar_login_with_captcha(request: CaptchaLoginRequest) -> dict:
        """Login to SAJ with CAPTCHA code."""
        if not app.state.saj_monitor:
            raise HTTPException(status_code=503, detail="SAJ monitor not configured")

        logger.info(
            f"CAPTCHA login request: code={request.captcha_code}, has_key={app.state.saj_monitor._captcha_key is not None}"
        )
        success = await app.state.saj_monitor.login_with_captcha(request.captcha_code)
        if success:
            return {"status": "ok", "message": "SAJ login successful"}
        raise HTTPException(status_code=401, detail=app.state.saj_monitor.login_error or "Login failed")

    # =========================================================================
    # WEBSOCKET
    # =========================================================================

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket) -> None:
        """WebSocket endpoint - updates pushed via broadcast task."""
        await websocket.accept()
        app.state.websocket_connections.append(websocket)

        try:
            # Keep connection alive, handle client messages
            while True:
                try:
                    data = await websocket.receive_text()
                    if data == "ping":
                        await websocket.send_text("pong")
                except WebSocketDisconnect:
                    break
        except Exception:
            pass
        finally:
            if websocket in app.state.websocket_connections:
                app.state.websocket_connections.remove(websocket)

    # =========================================================================
    # STATIC FILES
    # =========================================================================

    @app.get("/", response_model=None)
    async def index():
        """Serve frontend."""
        static_path = Path(__file__).parent / "static" / "index.html"
        if static_path.exists():
            return FileResponse(static_path)
        return JSONResponse(
            content={"message": "API running", "docs": "/docs"},
            status_code=200,
        )

    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    return app
