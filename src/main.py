"""
FSM-based EV Charging Controller - Main Entry Point.

Architecture:
- EaseeController: Polls charger state, emits state changes
- FSMManager: Manages FSM instances, handles mode switching
  - SmartChargingFSM: Schedule-aware charging (SMART, SCHEDULED, SOLAR)
  - ManualChargingFSM: User-controlled charging (MANUAL, IMMEDIATE)
- ScheduleProvider: Calculates charging windows, notifies FSM
- LoadMonitor: Monitors home load, notifies FSM
- Web App: Read-only viewer + mode changes

Flow:
1. EaseeController detects charger state change -> FSMManager.on_charger_state_change()
2. ScheduleProvider detects schedule change -> FSMManager.on_schedule_update()
3. LoadMonitor detects load change -> FSMManager.on_load_update()
4. Active FSM evaluates state and takes action (start/stop/pause charging)
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys
from typing import TYPE_CHECKING

import uvicorn

from .charging_fsm import set_charging_fsm
from .config import get_config, load_config
from .easee_controller import EaseeController
from .fsm_manager import FSMManager, set_fsm_manager
from .load_monitor import LoadMonitor, set_load_monitor
from .price_fetcher import PriceFetcher, set_price_fetcher
from .saj_monitor import SAJMonitor, get_saj_monitor
from .saveeye_monitor import SaveEyeMonitor, set_saveeye_monitor
from .schedule_provider import ScheduleProvider, set_schedule_provider
from .web.app import create_app

if TYPE_CHECKING:
    from .saveeye_monitor import PowerReading

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)

logger = logging.getLogger("app")


class ChargingApplication:
    """Main application that wires all components together."""

    def __init__(self, config_path: str | None = None) -> None:
        # Load configuration
        if config_path:
            load_config(config_path)

        self._config = get_config()

        # Components (initialized in start())
        self._charger: EaseeController | None = None
        self._fsm_manager: FSMManager | None = None
        self._schedule_provider: ScheduleProvider | None = None
        self._load_monitor: LoadMonitor | None = None
        self._price_fetcher: PriceFetcher | None = None
        self._saveeye: SaveEyeMonitor | None = None
        self._saj_monitor: SAJMonitor | None = None
        self._web_app = None

        self._running = False
        self._shutdown_event = asyncio.Event()

    async def start(self) -> None:
        """Start all components."""
        logger.info("Starting FSM-based charging controller...")

        # 1. Initialize EaseeController
        logger.info("Initializing EASEE controller...")
        self._charger = EaseeController()
        await self._charger.connect()

        # 2. Initialize FSMManager (manages both Smart and Manual FSMs)
        logger.info("Initializing FSM Manager...")
        self._fsm_manager = FSMManager(self._charger)
        set_fsm_manager(self._fsm_manager)
        # Set global FSM singleton for backward compatibility
        set_charging_fsm(self._fsm_manager.active_fsm)

        # 3. Initialize PriceFetcher
        logger.info("Initializing price fetcher...")
        self._price_fetcher = PriceFetcher()
        set_price_fetcher(self._price_fetcher)

        # 4. Initialize ScheduleProvider
        logger.info("Initializing schedule provider...")
        self._schedule_provider = ScheduleProvider(self._price_fetcher)
        self._schedule_provider.set_fsm(self._fsm_manager)
        set_schedule_provider(self._schedule_provider)
        
        # Wire FSMManager to ScheduleProvider for mode-based enable/disable
        self._fsm_manager.set_schedule_provider(self._schedule_provider)

        # 5. Initialize LoadMonitor
        logger.info("Initializing load monitor...")
        self._load_monitor = LoadMonitor()
        # LoadMonitor directly controls current via controller (not FSM)
        self._load_monitor.set_current_controller(self._charger)
        # LoadMonitor notifies FSM about should_pause signal
        self._load_monitor.set_fsm(self._fsm_manager)
        # LoadMonitor notifies ScheduleProvider about current changes (affects charging plan)
        self._load_monitor.set_current_change_consumer(self._schedule_provider)
        set_load_monitor(self._load_monitor)

        # 6. Initialize SaveEyeMonitor (if configured)
        if self._config.saveeye.enabled:
            logger.info("Initializing SaveEye monitor...")
            self._saveeye = SaveEyeMonitor()
            set_saveeye_monitor(self._saveeye)

            # Wire SaveEye readings to LoadMonitor
            self._saveeye.on_reading(self._on_power_reading)

        # 6b. Initialize SAJ Solar Monitor (if configured)
        if self._config.solar.enabled:
            logger.info("Initializing SAJ solar monitor...")
            self._saj_monitor = get_saj_monitor()

        # 7. Wire charger state changes to FSMManager
        self._charger.on_state_change(self._fsm_manager.on_charger_state_change)

        # 8. Wire phase detection: FSMManager -> ScheduleProvider
        # Both FSMs notify ScheduleProvider when phases are detected
        self._fsm_manager.add_phase_detection_consumer(self._schedule_provider)

        # 9. Create web app
        logger.info("Creating web application...")
        self._web_app = create_app(
            fsm_manager=self._fsm_manager,
            schedule_provider=self._schedule_provider,
            load_monitor=self._load_monitor,
            charger=self._charger,
            price_fetcher=self._price_fetcher,
            saj_monitor=self._saj_monitor,
        )

        # 10. Start all components
        await self._fsm_manager.start()
        await self._schedule_provider.start()
        self._load_monitor.start()

        if self._saveeye:
            await self._saveeye.start()

        if self._saj_monitor:
            await self._saj_monitor.start()

        # 11. Start charger monitoring
        await self._charger.start_monitoring(interval=2.0)

        self._running = True
        logger.info("All components started successfully")

    async def stop(self) -> None:
        """Stop all components."""
        logger.info("Stopping charging controller...")
        self._running = False

        # Stop in reverse order
        if self._charger:
            await self._charger.stop_monitoring()

        if self._saveeye:
            await self._saveeye.stop()

        if self._saj_monitor:
            await self._saj_monitor.stop()

        if self._load_monitor:
            self._load_monitor.stop()

        if self._schedule_provider:
            await self._schedule_provider.stop()

        if self._fsm_manager:
            await self._fsm_manager.stop()

        if self._charger:
            await self._charger.disconnect()

        logger.info("All components stopped")

    async def _on_power_reading(self, reading: PowerReading) -> None:
        """Handle power reading from SaveEye."""
        if not self._running or not self._load_monitor or not self._charger:
            return

        # Get charger status for EV current calculation
        charger_status = self._charger.last_status

        # Pass to load monitor
        await self._load_monitor.on_power_reading(reading, charger_status)

    async def run_web_server(self, host: str = "0.0.0.0", port: int = 8080) -> None:
        """Run the web server."""
        config = uvicorn.Config(
            self._web_app,
            host=host,
            port=port,
            log_level="info",
        )
        server = uvicorn.Server(config)
        await server.serve()

    async def wait_for_shutdown(self) -> None:
        """Wait for shutdown signal."""
        await self._shutdown_event.wait()

    def request_shutdown(self) -> None:
        """Request application shutdown."""
        self._shutdown_event.set()


async def main(config_path: str | None = None) -> None:
    """Main entry point."""
    app = ChargingApplication(config_path)

    # Setup signal handlers
    loop = asyncio.get_event_loop()

    def signal_handler() -> None:
        logger.info("Received shutdown signal")
        app.request_shutdown()

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, signal_handler)

    try:
        # Start application
        await app.start()

        # Get web server config
        config = get_config()
        host = config.web.host
        port = config.web.port

        logger.info(f"Starting web server on {host}:{port}")

        # Run web server and wait for shutdown
        web_task = asyncio.create_task(app.run_web_server(host, port))
        shutdown_task = asyncio.create_task(app.wait_for_shutdown())

        # Wait for either to complete
        done, pending = await asyncio.wait(
            [web_task, shutdown_task],
            return_when=asyncio.FIRST_COMPLETED,
        )

        # Cancel pending tasks
        for task in pending:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    finally:
        await app.stop()


def run() -> None:
    """Entry point for console script."""
    parser = argparse.ArgumentParser(description="FSM-based EV Charging Controller")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config.yaml file",
    )
    args = parser.parse_args()

    try:
        asyncio.run(main(args.config))
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Application error: {e}")
        raise


if __name__ == "__main__":
    run()
