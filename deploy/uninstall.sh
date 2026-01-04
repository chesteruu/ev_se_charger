#!/bin/bash
#
# EASEE Smart Charger Controller - Uninstallation Script
#

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Configuration
INSTALL_DIR="/home/pi/easee-controller"
SERVICE_NAME="easee-controller"
LOG_DIR="/var/log/easee-controller"
DATA_DIR="/var/lib/easee-controller"

echo -e "${YELLOW}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${YELLOW}║    EASEE Smart Charger Controller - Uninstallation Script      ║${NC}"
echo -e "${YELLOW}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check if running as root
if [[ $EUID -ne 0 ]]; then
   echo -e "${RED}This script must be run as root (use sudo)${NC}"
   exit 1
fi

read -p "Are you sure you want to uninstall the EASEE Controller? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Uninstall cancelled"
    exit 1
fi

echo -e "${GREEN}[1/4]${NC} Stopping service..."
systemctl stop "$SERVICE_NAME" 2>/dev/null || true
systemctl disable "$SERVICE_NAME" 2>/dev/null || true

echo -e "${GREEN}[2/4]${NC} Removing systemd service..."
rm -f /etc/systemd/system/"$SERVICE_NAME".service
systemctl daemon-reload

echo -e "${GREEN}[3/4]${NC} Removing application files..."
rm -rf "$INSTALL_DIR"

read -p "Remove log files? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    rm -rf "$LOG_DIR"
fi

read -p "Remove database and data files? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    rm -rf "$DATA_DIR"
fi

echo ""
echo -e "${GREEN}Uninstallation complete!${NC}"

