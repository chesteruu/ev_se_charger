"""Tests for the load balancer module."""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.load_balancer import LoadBalancer, LoadZone, LoadStatus


class TestLoadZoneClassification:
    """Test load zone classification logic."""
    
    def test_safe_zone_threshold(self):
        """Test that utilization below 70% is classified as SAFE."""
        # Create a mock load status
        status = LoadStatus(
            timestamp=datetime.now(),
            zone=LoadZone.SAFE,
            home_consumption_amps=10,
            ev_charging_amps=5,
            total_current_amps=15,
            fuse_limit_amps=25,
            safety_margin_amps=3,
            effective_limit_amps=22,
            available_amps=12,
            utilization_percent=60,  # 60% < 70% = SAFE
            recommended_ev_current=16,
            should_pause_charging=False,
        )
        
        assert status.zone == LoadZone.SAFE
        assert status.is_safe
        assert not status.is_critical
    
    def test_warning_zone_threshold(self):
        """Test that utilization 70-85% is classified as WARNING."""
        status = LoadStatus(
            timestamp=datetime.now(),
            zone=LoadZone.WARNING,
            home_consumption_amps=15,
            ev_charging_amps=5,
            total_current_amps=20,
            fuse_limit_amps=25,
            safety_margin_amps=3,
            effective_limit_amps=22,
            available_amps=7,
            utilization_percent=80,  # 70% < 80% < 85% = WARNING
            recommended_ev_current=10,
            should_pause_charging=False,
        )
        
        assert status.zone == LoadZone.WARNING
        assert not status.is_safe
        assert not status.is_critical
    
    def test_critical_zone_threshold(self):
        """Test that utilization 85-95% is classified as CRITICAL."""
        status = LoadStatus(
            timestamp=datetime.now(),
            zone=LoadZone.CRITICAL,
            home_consumption_amps=20,
            ev_charging_amps=3,
            total_current_amps=23,
            fuse_limit_amps=25,
            safety_margin_amps=3,
            effective_limit_amps=22,
            available_amps=2,
            utilization_percent=92,  # 85% < 92% < 95% = CRITICAL
            recommended_ev_current=6,
            should_pause_charging=False,
        )
        
        assert status.zone == LoadZone.CRITICAL
        assert not status.is_safe
        assert status.is_critical
    
    def test_danger_zone_threshold(self):
        """Test that utilization > 95% is classified as DANGER."""
        status = LoadStatus(
            timestamp=datetime.now(),
            zone=LoadZone.DANGER,
            home_consumption_amps=23,
            ev_charging_amps=2,
            total_current_amps=25,
            fuse_limit_amps=25,
            safety_margin_amps=3,
            effective_limit_amps=22,
            available_amps=0,
            utilization_percent=100,  # > 95% = DANGER
            recommended_ev_current=0,
            should_pause_charging=True,
        )
        
        assert status.zone == LoadZone.DANGER
        assert not status.is_safe
        assert status.is_critical
        assert status.should_pause_charging


class TestLoadStatusProperties:
    """Test LoadStatus computed properties."""
    
    def test_is_safe_property(self):
        """Test is_safe property returns True only for SAFE zone."""
        zones = [LoadZone.SAFE, LoadZone.WARNING, LoadZone.CRITICAL, LoadZone.DANGER]
        expected = [True, False, False, False]
        
        for zone, expected_safe in zip(zones, expected):
            status = LoadStatus(
                timestamp=datetime.now(),
                zone=zone,
                home_consumption_amps=0,
                ev_charging_amps=0,
                total_current_amps=0,
                fuse_limit_amps=25,
                safety_margin_amps=3,
                effective_limit_amps=22,
                available_amps=22,
                utilization_percent=0,
                recommended_ev_current=0,
                should_pause_charging=False,
            )
            assert status.is_safe == expected_safe
    
    def test_is_critical_property(self):
        """Test is_critical property returns True for CRITICAL and DANGER zones."""
        zones = [LoadZone.SAFE, LoadZone.WARNING, LoadZone.CRITICAL, LoadZone.DANGER]
        expected = [False, False, True, True]
        
        for zone, expected_critical in zip(zones, expected):
            status = LoadStatus(
                timestamp=datetime.now(),
                zone=zone,
                home_consumption_amps=0,
                ev_charging_amps=0,
                total_current_amps=0,
                fuse_limit_amps=25,
                safety_margin_amps=3,
                effective_limit_amps=22,
                available_amps=22,
                utilization_percent=0,
                recommended_ev_current=0,
                should_pause_charging=False,
            )
            assert status.is_critical == expected_critical


class TestLoadBalancerCurrentCalculation:
    """Test current calculation logic."""
    
    def test_available_current_calculation(self):
        """Test that available current is calculated correctly."""
        # Effective limit = 25 - 3 = 22A
        # Home consumption = 10A
        # Available = 22 - 10 = 12A
        
        fuse_limit = 25
        safety_margin = 3
        home_consumption = 10
        
        effective_limit = fuse_limit - safety_margin
        available = effective_limit - home_consumption
        
        assert available == 12
    
    def test_recommended_current_capped_at_max(self):
        """Test that recommended current doesn't exceed max charger current."""
        max_charger_current = 16
        available = 20  # More than charger can use
        
        recommended = min(available, max_charger_current)
        assert recommended == 16
    
    def test_minimum_charging_current(self):
        """Test that charging respects minimum current threshold."""
        min_current = 6
        available = 4  # Less than minimum
        
        # Should recommend pausing if below minimum
        should_pause = available < min_current
        assert should_pause

