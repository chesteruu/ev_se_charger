#!/bin/bash
#
# EASEE Smart Charger Controller - Installation Script
#
# This script installs the controller on a Raspberry Pi
#

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
INSTALL_DIR="/home/pi/easee-controller"
SERVICE_NAME="easee-controller"
LOG_DIR="/var/log/easee-controller"
DATA_DIR="/var/lib/easee-controller"
USER="pi"
GROUP="pi"

echo -e "${GREEN}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║     EASEE Smart Charger Controller - Installation Script       ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check if running as root
if [[ $EUID -ne 0 ]]; then
   echo -e "${RED}This script must be run as root (use sudo)${NC}"
   exit 1
fi

# Check if running on Raspberry Pi
if [[ ! -f /proc/device-tree/model ]] || ! grep -q "Raspberry Pi" /proc/device-tree/model 2>/dev/null; then
    echo -e "${YELLOW}Warning: This doesn't appear to be a Raspberry Pi${NC}"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo -e "${GREEN}[1/7]${NC} Updating system packages..."
apt-get update
apt-get install -y python3 python3-pip python3-venv git

echo -e "${GREEN}[2/7]${NC} Creating directories..."
mkdir -p "$LOG_DIR"
mkdir -p "$DATA_DIR"
mkdir -p "$INSTALL_DIR"

chown -R "$USER:$GROUP" "$LOG_DIR"
chown -R "$USER:$GROUP" "$DATA_DIR"
chown -R "$USER:$GROUP" "$INSTALL_DIR"

echo -e "${GREEN}[3/7]${NC} Copying application files..."
if [[ -d "$(dirname "$0")/../src" ]]; then
    cp -r "$(dirname "$0")/../"* "$INSTALL_DIR/"
else
    echo -e "${YELLOW}Source files not found, please clone the repository manually${NC}"
fi

echo -e "${GREEN}[4/7]${NC} Creating virtual environment..."
sudo -u "$USER" python3 -m venv "$INSTALL_DIR/venv"

echo -e "${GREEN}[5/7]${NC} Installing Python dependencies..."
sudo -u "$USER" "$INSTALL_DIR/venv/bin/pip" install --upgrade pip
sudo -u "$USER" "$INSTALL_DIR/venv/bin/pip" install -r "$INSTALL_DIR/requirements.txt"

echo -e "${GREEN}[6/7]${NC} Setting up configuration..."
if [[ ! -f "$INSTALL_DIR/config.env" ]]; then
    if [[ -f "$INSTALL_DIR/config.example.env" ]]; then
        cp "$INSTALL_DIR/config.example.env" "$INSTALL_DIR/config.env"
        chown "$USER:$GROUP" "$INSTALL_DIR/config.env"
        chmod 600 "$INSTALL_DIR/config.env"
        echo -e "${YELLOW}Please edit $INSTALL_DIR/config.env with your settings${NC}"
    fi
fi

echo -e "${GREEN}[7/7]${NC} Installing systemd service..."
cp "$INSTALL_DIR/deploy/easee-controller.service" /etc/systemd/system/
systemctl daemon-reload
systemctl enable "$SERVICE_NAME"

echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                    Installation Complete!                       ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "Next steps:"
echo -e "  1. Edit the configuration file:"
echo -e "     ${YELLOW}sudo nano $INSTALL_DIR/config.env${NC}"
echo -e ""
echo -e "  2. Start the service:"
echo -e "     ${YELLOW}sudo systemctl start $SERVICE_NAME${NC}"
echo -e ""
echo -e "  3. Check the status:"
echo -e "     ${YELLOW}sudo systemctl status $SERVICE_NAME${NC}"
echo -e ""
echo -e "  4. View the logs:"
echo -e "     ${YELLOW}sudo journalctl -u $SERVICE_NAME -f${NC}"
echo -e ""
echo -e "  5. Access the dashboard at:"
echo -e "     ${YELLOW}http://$(hostname -I | awk '{print $1}'):8080${NC}"
echo ""

