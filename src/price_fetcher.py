"""
Electricity Price Fetcher Module.

Fetches and manages electricity price forecasts from various sources:
- Nord Pool (Nordic/Baltic markets)
- Tibber (if you have a Tibber subscription)
- ENTSO-E (European markets)
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum

import aiohttp
import httpx

from .config import get_config

logger = logging.getLogger(__name__)


class PriceLevel(Enum):
    """Price level classification."""

    VERY_CHEAP = "very_cheap"
    CHEAP = "cheap"
    NORMAL = "normal"
    EXPENSIVE = "expensive"
    VERY_EXPENSIVE = "very_expensive"


@dataclass
class HourlyPrice:
    """Electricity price for a specific hour."""

    start_time: datetime
    end_time: datetime
    price: float  # Price per kWh
    currency: str
    level: PriceLevel = PriceLevel.NORMAL

    @property
    def price_per_kwh(self) -> float:
        """Get price per kWh."""
        return self.price

    @property
    def is_cheap(self) -> bool:
        """Check if price is cheap or very cheap."""
        return self.level in (PriceLevel.VERY_CHEAP, PriceLevel.CHEAP)

    @property
    def is_expensive(self) -> bool:
        """Check if price is expensive or very expensive."""
        return self.level in (PriceLevel.EXPENSIVE, PriceLevel.VERY_EXPENSIVE)


@dataclass
class PriceForecast:
    """Collection of hourly prices with analysis."""

    prices: list[HourlyPrice] = field(default_factory=list)
    fetched_at: datetime = field(default_factory=datetime.now)
    source: str = "unknown"
    area: str = ""
    currency: str = ""

    @property
    def current_price(self) -> HourlyPrice | None:
        """Get current hour's price."""
        now = datetime.now(timezone.utc)
        for price in self.prices:
            if price.start_time <= now < price.end_time:
                return price
        return None

    @property
    def min_price(self) -> HourlyPrice | None:
        """Get lowest price in forecast."""
        if not self.prices:
            return None
        return min(self.prices, key=lambda p: p.price)

    @property
    def max_price(self) -> HourlyPrice | None:
        """Get highest price in forecast."""
        if not self.prices:
            return None
        return max(self.prices, key=lambda p: p.price)

    @property
    def average_price(self) -> float:
        """Get average price."""
        if not self.prices:
            return 0.0
        return sum(p.price for p in self.prices) / len(self.prices)

    def get_prices_for_period(
        self,
        start: datetime,
        end: datetime
    ) -> list[HourlyPrice]:
        """Get prices within a time period."""
        return [
            p for p in self.prices
            if p.start_time >= start and p.end_time <= end
        ]

    def get_cheapest_hours(
        self,
        count: int,
        start_from: datetime | None = None,
        end_before: datetime | None = None
    ) -> list[HourlyPrice]:
        """
        Get the N cheapest hours.

        Args:
            count: Number of hours to return
            start_from: Only consider hours starting from this time
            end_before: Only consider hours ending before this time

        Returns:
            List of cheapest hours, sorted by time.
        """
        filtered = self.prices

        if start_from:
            filtered = [p for p in filtered if p.start_time >= start_from]
        if end_before:
            filtered = [p for p in filtered if p.end_time <= end_before]

        # Sort by price and take cheapest
        sorted_by_price = sorted(filtered, key=lambda p: p.price)[:count]

        # Return sorted by time
        return sorted(sorted_by_price, key=lambda p: p.start_time)

    def get_consecutive_cheap_hours(
        self,
        min_hours: int,
        max_price: float,
        start_from: datetime | None = None
    ) -> list[list[HourlyPrice]]:
        """
        Find consecutive periods of cheap hours.

        Args:
            min_hours: Minimum consecutive hours needed
            max_price: Maximum acceptable price
            start_from: Only consider hours from this time

        Returns:
            List of consecutive cheap hour periods.
        """
        filtered = self.prices
        if start_from:
            filtered = [p for p in filtered if p.start_time >= start_from]

        # Sort by time
        sorted_prices = sorted(filtered, key=lambda p: p.start_time)

        periods = []
        current_period = []

        for price in sorted_prices:
            if price.price <= max_price:
                if not current_period or \
                   price.start_time == current_period[-1].end_time:
                    current_period.append(price)
                else:
                    if len(current_period) >= min_hours:
                        periods.append(current_period)
                    current_period = [price]
            else:
                if len(current_period) >= min_hours:
                    periods.append(current_period)
                current_period = []

        if len(current_period) >= min_hours:
            periods.append(current_period)

        return periods


class PriceFetcherBase(ABC):
    """Base class for price fetchers."""

    @abstractmethod
    async def fetch_prices(self, area: str) -> PriceForecast:
        """Fetch price forecast."""
        pass

    def _classify_price(
        self,
        price: float,
        all_prices: list[float]
    ) -> PriceLevel:
        """
        Classify a price based on its position in the distribution.

        Args:
            price: The price to classify
            all_prices: All prices to compare against

        Returns:
            PriceLevel classification.
        """
        if not all_prices:
            return PriceLevel.NORMAL

        sorted_prices = sorted(all_prices)
        n = len(sorted_prices)

        # Find percentile
        below = sum(1 for p in sorted_prices if p < price)
        percentile = below / n * 100

        if percentile < 10:
            return PriceLevel.VERY_CHEAP
        elif percentile < 30:
            return PriceLevel.CHEAP
        elif percentile < 70:
            return PriceLevel.NORMAL
        elif percentile < 90:
            return PriceLevel.EXPENSIVE
        else:
            return PriceLevel.VERY_EXPENSIVE


class NordPoolFetcher(PriceFetcherBase):
    """
    Fetch prices from Nord Pool Data Portal API.

    Supports areas: NO1-NO5 (Norway), SE1-SE4 (Sweden),
    DK1-DK2 (Denmark), FI (Finland), EE, LV, LT (Baltic)
    """

    # Nord Pool Data Portal API
    BASE_URL = "https://dataportal-api.nordpoolgroup.com/api/DayAheadPrices"

    def __init__(self, currency: str = "SEK"):
        self.currency = currency

    async def fetch_prices(self, area: str = "SE3") -> PriceForecast:
        """
        Fetch Nord Pool prices from the Data Portal API.

        Args:
            area: Price area code (e.g., NO1, SE3)

        Returns:
            PriceForecast with hourly prices.
        """
        hourly_prices = []
        all_raw_prices = []

        # Fetch today and tomorrow
        for day_offset in [0, 1]:
            target_date = datetime.now() + timedelta(days=day_offset)
            date_str = target_date.strftime('%Y-%m-%d')

            params = {
                'date': date_str,
                'market': 'DayAhead',
                'deliveryArea': area,
                'currency': self.currency,
            }

            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(self.BASE_URL, params=params, timeout=aiohttp.ClientTimeout(total=30)) as response:
                        if response.status != 200:
                            logger.warning(f"Nord Pool API returned {response.status} for {date_str}")
                            continue

                        data = await response.json()

                # Parse the multiAreaEntries (15-minute intervals)
                entries = data.get('multiAreaEntries', [])

                # Group 15-minute intervals into hourly prices
                hourly_data = {}

                for entry in entries:
                    start_str = entry.get('deliveryStart')
                    price_value = entry.get('entryPerArea', {}).get(area)

                    if not start_str or price_value is None:
                        continue

                    # Parse timestamp
                    start_time = datetime.fromisoformat(start_str.replace('Z', '+00:00'))

                    # Round down to hour
                    hour_key = start_time.replace(minute=0, second=0, microsecond=0)

                    if hour_key not in hourly_data:
                        hourly_data[hour_key] = []

                    # Price is in currency/MWh, convert to currency/kWh
                    price_kwh = price_value / 1000
                    hourly_data[hour_key].append(price_kwh)

                # Average the 15-minute prices into hourly prices
                for hour_start, prices in sorted(hourly_data.items()):
                    avg_price = sum(prices) / len(prices)
                    all_raw_prices.append(avg_price)

                    hourly_prices.append(HourlyPrice(
                        start_time=hour_start,
                        end_time=hour_start + timedelta(hours=1),
                        price=avg_price,
                        currency=self.currency,
                    ))

            except Exception as e:
                logger.warning(f"Failed to fetch prices for {date_str}: {e}")
                continue

        # Remove duplicates and sort by time
        seen = set()
        unique_prices = []
        for price in sorted(hourly_prices, key=lambda p: p.start_time):
            if price.start_time not in seen:
                seen.add(price.start_time)
                unique_prices.append(price)
        hourly_prices = unique_prices

        # Classify price levels
        for price in hourly_prices:
            price.level = self._classify_price(price.price, all_raw_prices)

        return PriceForecast(
            prices=hourly_prices,
            fetched_at=datetime.now(timezone.utc),
            source="nordpool",
            area=area,
            currency=self.currency,
        )

    async def _fetch_via_api_legacy(self, area: str) -> PriceForecast:
        """Legacy API - no longer working."""
        # This is kept for reference but the old API is deprecated
        raise NotImplementedError("Legacy Nord Pool API is deprecated")



class TibberFetcher(PriceFetcherBase):
    """
    Fetch prices from Tibber API.

    Requires a Tibber account and API token.
    """

    API_URL = "https://api.tibber.com/v1-beta/gql"

    def __init__(self, access_token: str, currency: str = "NOK"):
        self.access_token = access_token
        self.currency = currency

    async def fetch_prices(self, area: str = "") -> PriceForecast:
        """
        Fetch Tibber prices.

        Note: Area is not used for Tibber as it's determined by your home.

        Returns:
            PriceForecast with hourly prices.
        """
        query = """
        {
            viewer {
                homes {
                    currentSubscription {
                        priceInfo {
                            current {
                                total
                                energy
                                tax
                                startsAt
                                level
                            }
                            today {
                                total
                                energy
                                tax
                                startsAt
                                level
                            }
                            tomorrow {
                                total
                                energy
                                tax
                                startsAt
                                level
                            }
                        }
                    }
                }
            }
        }
        """

        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.API_URL,
                json={"query": query},
                headers=headers,
            )
            response.raise_for_status()
            data = response.json()

        hourly_prices = []
        all_raw_prices = []

        # Parse response
        homes = data.get("data", {}).get("viewer", {}).get("homes", [])
        if not homes:
            raise ValueError("No homes found in Tibber account")

        price_info = homes[0].get("currentSubscription", {}).get("priceInfo", {})

        # Combine today and tomorrow prices
        for period in ["today", "tomorrow"]:
            for entry in price_info.get(period, []):
                price = entry.get("total", 0)
                all_raw_prices.append(price)

                start_time = datetime.fromisoformat(
                    entry["startsAt"].replace("Z", "+00:00")
                )

                # Tibber level mapping
                level_map = {
                    "VERY_CHEAP": PriceLevel.VERY_CHEAP,
                    "CHEAP": PriceLevel.CHEAP,
                    "NORMAL": PriceLevel.NORMAL,
                    "EXPENSIVE": PriceLevel.EXPENSIVE,
                    "VERY_EXPENSIVE": PriceLevel.VERY_EXPENSIVE,
                }

                hourly_prices.append(HourlyPrice(
                    start_time=start_time,
                    end_time=start_time + timedelta(hours=1),
                    price=price,
                    currency=self.currency,
                    level=level_map.get(entry.get("level", "NORMAL"), PriceLevel.NORMAL),
                ))

        return PriceForecast(
            prices=hourly_prices,
            fetched_at=datetime.now(timezone.utc),
            source="tibber",
            area="tibber_home",
            currency=self.currency,
        )


class PriceFetcher:
    """
    Unified price fetcher with caching.

    Automatically selects the appropriate data source based on configuration.
    Applies VAT and grid tariff to spot prices.
    """

    def __init__(self):
        config = get_config()
        self._source = config.price.source
        self._area = config.price.nordpool_area
        self._currency = config.price.currency
        self._tibber_token = config.price.tibber_access_token

        # VAT and tariff settings
        self._vat_percent = config.price.vat_percent
        self._grid_tariff = config.grid_tariff

        logger.info(
            f"Price fetcher initialized: VAT={self._vat_percent}%, "
            f"day_rate={self._grid_tariff.day_rate_ore}öre, "
            f"night_rate={self._grid_tariff.night_rate_ore}öre "
            f"(night: {self._grid_tariff.night_start_hour}:00-{self._grid_tariff.night_end_hour}:00)"
        )

        # Cache
        self._cache: PriceForecast | None = None
        self._cache_expiry: datetime | None = None
        self._cache_duration = timedelta(minutes=30)

    async def get_prices(self, force_refresh: bool = False) -> PriceForecast:
        """
        Get price forecast (from cache if available).

        Args:
            force_refresh: Force fetching new data

        Returns:
            PriceForecast with hourly prices.
        """
        now = datetime.now(timezone.utc)

        # Check cache
        if not force_refresh and self._cache and self._cache_expiry:
            if now < self._cache_expiry:
                logger.debug("Returning cached prices")
                return self._cache

        # Fetch new prices
        forecast = await self._fetch()

        # Update cache
        self._cache = forecast
        self._cache_expiry = now + self._cache_duration

        return forecast

    async def _fetch(self) -> PriceForecast:
        """Fetch prices from configured source and apply VAT + grid tariff."""
        if self._source == "tibber":
            if not self._tibber_token:
                raise ValueError("Tibber token not configured")
            fetcher = TibberFetcher(self._tibber_token, self._currency)
            forecast = await fetcher.fetch_prices()
        else:
            # Default to Nord Pool
            fetcher = NordPoolFetcher(self._currency)
            forecast = await fetcher.fetch_prices(self._area)

        # Apply VAT and grid tariff to all prices
        self._apply_vat_and_tariff(forecast)

        return forecast

    def _apply_vat_and_tariff(self, forecast: PriceForecast) -> None:
        """
        Apply VAT and grid tariff to all prices in the forecast.

        Nord Pool prices are spot prices excluding VAT.
        Total price = (spot_price * (1 + VAT%/100)) + grid_tariff
        """
        vat_multiplier = 1 + (self._vat_percent / 100)

        for price in forecast.prices:
            # Get the hour in local time for tariff calculation
            local_hour = price.start_time.astimezone().hour

            # Original spot price (excluding VAT)
            spot_price = price.price

            # Add VAT to spot price
            price_with_vat = spot_price * vat_multiplier

            # Add grid tariff (day or night rate)
            tariff = self._grid_tariff.get_rate_for_hour(local_hour)
            is_night = self._grid_tariff.is_night_hour(local_hour)

            # Total price
            total_price = price_with_vat + tariff

            # Update price
            price.price = total_price

            # Log first few for debugging
            logger.debug(
                f"Hour {local_hour:02d}: spot={spot_price:.4f} + VAT={price_with_vat - spot_price:.4f} "
                f"+ tariff={tariff:.4f} ({'night' if is_night else 'day'}) = {total_price:.4f} SEK/kWh"
            )

    @property
    def current_price(self) -> HourlyPrice | None:
        """Get current price from cache."""
        if self._cache:
            return self._cache.current_price
        return None

    def get_optimal_charging_windows(
        self,
        hours_needed: int,
        deadline: datetime | None = None,
        max_price: float | None = None
    ) -> list[HourlyPrice]:
        """
        Find optimal charging windows.

        Args:
            hours_needed: Number of hours of charging needed
            deadline: Car must be ready by this time
            max_price: Maximum acceptable price (optional)

        Returns:
            List of recommended charging hours.
        """
        if not self._cache:
            return []

        now = datetime.now(timezone.utc)

        return self._cache.get_cheapest_hours(
            count=hours_needed,
            start_from=now,
            end_before=deadline,
        )


# Global instance
_price_fetcher: PriceFetcher | None = None


def get_price_fetcher() -> PriceFetcher:
    """Get the price fetcher singleton."""
    global _price_fetcher
    if _price_fetcher is None:
        _price_fetcher = PriceFetcher()
    return _price_fetcher


def set_price_fetcher(fetcher: PriceFetcher) -> None:
    """Set the global price fetcher instance."""
    global _price_fetcher
    _price_fetcher = fetcher


async def test_price_fetcher():
    """Test the price fetcher."""
    fetcher = get_price_fetcher()
    config = get_config()

    try:
        print("=" * 60)
        print("PRICE FETCHER TEST")
        print("=" * 60)

        # Show configuration
        print("\n📋 Configuration:")
        print(f"  VAT: {config.price.vat_percent}%")
        print(f"  Grid tariff (day):   {config.grid_tariff.day_rate_ore} öre/kWh")
        print(f"  Grid tariff (night): {config.grid_tariff.night_rate_ore} öre/kWh")
        print(f"  Night hours: {config.grid_tariff.night_start_hour}:00 - {config.grid_tariff.night_end_hour}:00")

        forecast = await fetcher.get_prices(force_refresh=True)

        print(f"\n📊 Fetched {len(forecast.prices)} hourly prices from {forecast.source}")
        print(f"  Area: {forecast.area}")
        print(f"  Currency: {forecast.currency}")

        if forecast.current_price:
            print("\n💰 Current hour:")
            print(f"  Total price: {forecast.current_price.price:.4f} {forecast.currency}/kWh")
            print(f"  Price level: {forecast.current_price.level.value}")

        if forecast.min_price:
            local_time = forecast.min_price.start_time.astimezone()
            print(f"\n⬇️  Lowest price: {forecast.min_price.price:.4f} {forecast.currency}/kWh at {local_time.strftime('%Y-%m-%d %H:%M')}")

        if forecast.max_price:
            local_time = forecast.max_price.start_time.astimezone()
            print(f"⬆️  Highest price: {forecast.max_price.price:.4f} {forecast.currency}/kWh at {local_time.strftime('%Y-%m-%d %H:%M')}")

        print(f"📈 Average price: {forecast.average_price:.4f} {forecast.currency}/kWh")

        # Show all hours with breakdown
        print("\n📅 All hourly prices (total = spot×VAT + tariff):")
        print("-" * 70)
        print(f"{'Time':<12} {'Tariff':<8} {'Total':>12} {'Level':<15}")
        print("-" * 70)

        for price in sorted(forecast.prices, key=lambda p: p.start_time):
            local_time = price.start_time.astimezone()
            hour = local_time.hour
            is_night = config.grid_tariff.is_night_hour(hour)
            tariff_type = "🌙night" if is_night else "☀️day"

            print(
                f"{local_time.strftime('%m-%d %H:%M'):<12} "
                f"{tariff_type:<8} "
                f"{price.price:>10.4f} SEK "
                f"{price.level.value:<15}"
            )

        # Show cheapest 5 hours
        cheap_hours = forecast.get_cheapest_hours(5)
        print("\n🏆 Cheapest 5 hours for charging:")
        print("-" * 50)
        for hour in cheap_hours:
            local_time = hour.start_time.astimezone()
            is_night = config.grid_tariff.is_night_hour(local_time.hour)
            tariff_type = "🌙" if is_night else "☀️"
            print(f"  {tariff_type} {local_time.strftime('%m-%d %H:%M')} - {hour.price:.4f} {forecast.currency}/kWh")

    except Exception as e:
        print(f"✗ Price fetch failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(test_price_fetcher())
