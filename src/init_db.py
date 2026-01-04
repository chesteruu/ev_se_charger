"""
Database initialization utility.

Creates the SQLite database and all required tables.
"""

import asyncio
import sys
from pathlib import Path

from .config import get_config
from .database import init_database


async def main() -> int:
    """Initialize the database."""
    print("EASEE Smart Charger Controller - Database Initialization")
    print("=" * 60)

    config = get_config()
    db_path = config.database.path

    print(f"Database path: {db_path}")

    # Check if database already exists
    if Path(db_path).exists():
        print(f"Database already exists at {db_path}")
        response = input("Reinitialize? This will NOT delete existing data. (y/n): ")
        if response.lower() != "y":
            print("Cancelled.")
            return 0

    # Create directory if needed
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    print("Creating database schema...")
    await init_database()

    print()
    print("✓ Database initialized successfully!")
    print(f"  Location: {db_path}")

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
