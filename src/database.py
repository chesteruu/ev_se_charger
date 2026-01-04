"""
Database module for EASEE Smart Charger Controller.

Provides SQLite database for storing:
- Charging sessions history
- Price history
- Consumption statistics
- System events
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    and_,
    select,
)
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from .config import get_config

Base = declarative_base()


class ChargingSession(Base):
    """Charging session record."""

    __tablename__ = "charging_sessions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(100), unique=True, nullable=True)
    charger_id = Column(String(50), nullable=False)

    start_time = Column(DateTime, nullable=False)
    end_time = Column(DateTime, nullable=True)

    energy_kwh = Column(Float, default=0)
    avg_power_kw = Column(Float, default=0)
    max_power_kw = Column(Float, default=0)

    cost = Column(Float, nullable=True)
    avg_price = Column(Float, nullable=True)
    currency = Column(String(10), nullable=True)

    was_smart_charged = Column(Boolean, default=False)
    was_load_limited = Column(Boolean, default=False)

    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class PriceRecord(Base):
    """Hourly price record."""

    __tablename__ = "price_records"

    id = Column(Integer, primary_key=True, autoincrement=True)

    timestamp = Column(DateTime, nullable=False)
    area = Column(String(20), nullable=False)

    price = Column(Float, nullable=False)
    currency = Column(String(10), nullable=False)
    level = Column(String(20), nullable=True)

    source = Column(String(50), nullable=False)

    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    __table_args__ = (
        # Unique constraint on timestamp + area
        {"sqlite_autoincrement": True},
    )


class ConsumptionRecord(Base):
    """Consumption statistics record."""

    __tablename__ = "consumption_records"

    id = Column(Integer, primary_key=True, autoincrement=True)

    timestamp = Column(DateTime, nullable=False)

    # Power readings (Watts)
    power_import = Column(Float, default=0)
    power_export = Column(Float, default=0)
    power_ev = Column(Float, default=0)
    power_home = Column(Float, default=0)

    # Current readings (Amps)
    current_l1 = Column(Float, default=0)
    current_l2 = Column(Float, default=0)
    current_l3 = Column(Float, default=0)

    # Load balancer state
    load_zone = Column(String(20), nullable=True)
    utilization_percent = Column(Float, default=0)

    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class SystemEvent(Base):
    """System event log."""

    __tablename__ = "system_events"

    id = Column(Integer, primary_key=True, autoincrement=True)

    timestamp = Column(DateTime, nullable=False)
    event_type = Column(String(50), nullable=False)
    severity = Column(String(20), default="info")

    message = Column(Text, nullable=False)
    details = Column(Text, nullable=True)  # JSON string

    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class Database:
    """
    Database manager.

    Provides async interface for database operations.
    """

    def __init__(self, db_path: str = None):
        """
        Initialize database.

        Args:
            db_path: Path to SQLite database file.
        """
        config = get_config()
        self.db_path = db_path or config.database.path

        # Ensure directory exists
        db_dir = Path(self.db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)

        # Create async engine
        self.engine = create_async_engine(
            f"sqlite+aiosqlite:///{self.db_path}",
            echo=False,
        )

        # Create session factory
        self.async_session = sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

    async def init_db(self) -> None:
        """Initialize database schema."""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def close(self) -> None:
        """Close database connection."""
        await self.engine.dispose()

    # ==================== Charging Sessions ====================

    async def save_charging_session(self, session: ChargingSession) -> int:
        """Save a charging session."""
        async with self.async_session() as db:
            db.add(session)
            await db.commit()
            await db.refresh(session)
            return session.id

    async def get_charging_sessions(
        self,
        limit: int = 100,
        charger_id: str = None,
    ) -> list[ChargingSession]:
        """Get charging sessions."""
        async with self.async_session() as db:
            query = select(ChargingSession).order_by(
                ChargingSession.start_time.desc()
            ).limit(limit)

            if charger_id:
                query = query.where(ChargingSession.charger_id == charger_id)

            result = await db.execute(query)
            return result.scalars().all()

    async def get_session_stats(
        self,
        days: int = 30,
        charger_id: str = None,
    ) -> dict[str, Any]:
        """Get charging session statistics."""
        async with self.async_session() as db:
            cutoff = datetime.now(timezone.utc).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            cutoff = cutoff.replace(day=cutoff.day - days)

            query = select(ChargingSession).where(
                ChargingSession.start_time >= cutoff
            )

            if charger_id:
                query = query.where(ChargingSession.charger_id == charger_id)

            result = await db.execute(query)
            sessions = result.scalars().all()

            if not sessions:
                return {
                    "total_sessions": 0,
                    "total_energy_kwh": 0,
                    "total_cost": 0,
                    "avg_session_kwh": 0,
                    "smart_charged_percent": 0,
                }

            total_energy = sum(s.energy_kwh for s in sessions)
            total_cost = sum(s.cost or 0 for s in sessions)
            smart_count = sum(1 for s in sessions if s.was_smart_charged)

            return {
                "total_sessions": len(sessions),
                "total_energy_kwh": round(total_energy, 2),
                "total_cost": round(total_cost, 2),
                "avg_session_kwh": round(total_energy / len(sessions), 2),
                "smart_charged_percent": round(smart_count / len(sessions) * 100, 1),
            }

    # ==================== Price Records ====================

    async def save_price_records(self, records: list[PriceRecord]) -> None:
        """Save multiple price records."""
        async with self.async_session() as db:
            for record in records:
                # Check if already exists
                existing = await db.execute(
                    select(PriceRecord).where(
                        and_(
                            PriceRecord.timestamp == record.timestamp,
                            PriceRecord.area == record.area,
                        )
                    )
                )
                if not existing.scalar():
                    db.add(record)
            await db.commit()

    async def get_price_history(
        self,
        area: str,
        start: datetime,
        end: datetime,
    ) -> list[PriceRecord]:
        """Get price history for a period."""
        async with self.async_session() as db:
            query = select(PriceRecord).where(
                and_(
                    PriceRecord.area == area,
                    PriceRecord.timestamp >= start,
                    PriceRecord.timestamp <= end,
                )
            ).order_by(PriceRecord.timestamp)

            result = await db.execute(query)
            return result.scalars().all()

    # ==================== Consumption Records ====================

    async def save_consumption_record(self, record: ConsumptionRecord) -> None:
        """Save a consumption record."""
        async with self.async_session() as db:
            db.add(record)
            await db.commit()

    async def get_consumption_history(
        self,
        start: datetime,
        end: datetime,
        resolution_minutes: int = 15,
    ) -> list[dict[str, Any]]:
        """
        Get consumption history with optional aggregation.

        For high-resolution data, returns raw records.
        For lower resolution, aggregates data.
        """
        async with self.async_session() as db:
            query = select(ConsumptionRecord).where(
                and_(
                    ConsumptionRecord.timestamp >= start,
                    ConsumptionRecord.timestamp <= end,
                )
            ).order_by(ConsumptionRecord.timestamp)

            result = await db.execute(query)
            records = result.scalars().all()

            # Simple aggregation by time bucket
            if resolution_minutes <= 5:
                return [
                    {
                        "timestamp": r.timestamp.isoformat(),
                        "power_import": r.power_import,
                        "power_ev": r.power_ev,
                        "power_home": r.power_home,
                        "utilization": r.utilization_percent,
                    }
                    for r in records
                ]

            # Aggregate into buckets
            from collections import defaultdict
            buckets = defaultdict(list)

            for r in records:
                bucket_key = r.timestamp.replace(
                    minute=(r.timestamp.minute // resolution_minutes) * resolution_minutes,
                    second=0,
                    microsecond=0,
                )
                buckets[bucket_key].append(r)

            aggregated = []
            for bucket_time, bucket_records in sorted(buckets.items()):
                aggregated.append({
                    "timestamp": bucket_time.isoformat(),
                    "power_import": sum(r.power_import for r in bucket_records) / len(bucket_records),
                    "power_ev": sum(r.power_ev for r in bucket_records) / len(bucket_records),
                    "power_home": sum(r.power_home for r in bucket_records) / len(bucket_records),
                    "utilization": sum(r.utilization_percent for r in bucket_records) / len(bucket_records),
                })

            return aggregated

    # ==================== System Events ====================

    async def log_event(
        self,
        event_type: str,
        message: str,
        severity: str = "info",
        details: dict[str, Any] | None = None,
    ) -> None:
        """Log a system event."""
        import json

        async with self.async_session() as db:
            event = SystemEvent(
                timestamp=datetime.now(timezone.utc),
                event_type=event_type,
                severity=severity,
                message=message,
                details=json.dumps(details) if details else None,
            )
            db.add(event)
            await db.commit()

    async def get_events(
        self,
        limit: int = 100,
        event_type: str = None,
        severity: str = None,
    ) -> list[SystemEvent]:
        """Get system events."""
        async with self.async_session() as db:
            query = select(SystemEvent).order_by(
                SystemEvent.timestamp.desc()
            ).limit(limit)

            if event_type:
                query = query.where(SystemEvent.event_type == event_type)
            if severity:
                query = query.where(SystemEvent.severity == severity)

            result = await db.execute(query)
            return result.scalars().all()


# Global instance
_db: Database | None = None


def get_database() -> Database:
    """Get database singleton."""
    global _db
    if _db is None:
        _db = Database()
    return _db


async def init_database() -> None:
    """Initialize the database."""
    db = get_database()
    await db.init_db()


if __name__ == "__main__":
    # Initialize database when run directly
    asyncio.run(init_database())
    print("Database initialized successfully")
