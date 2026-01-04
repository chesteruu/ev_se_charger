#!/bin/bash
#
# Quick start script for running EASEE Controller in Docker
#

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Check if .env file exists
if [[ ! -f ".env" ]]; then
    echo -e "${YELLOW}No .env file found. Creating from template...${NC}"
    
    if [[ -f "config.example.env" ]]; then
        cp config.example.env .env
        echo -e "${YELLOW}Please edit .env with your credentials:${NC}"
        echo "  nano .env"
        echo ""
        echo "Required settings:"
        echo "  - EASEE_USERNAME"
        echo "  - EASEE_PASSWORD"  
        echo "  - EASEE_CHARGER_ID"
        echo "  - SAVEEYE_HOST"
        echo ""
        exit 1
    else
        echo -e "${RED}config.example.env not found!${NC}"
        exit 1
    fi
fi

# Check required environment variables
source .env

if [[ -z "$EASEE_USERNAME" ]] || [[ "$EASEE_USERNAME" == "your_easee_email@example.com" ]]; then
    echo -e "${RED}EASEE_USERNAME not configured in .env${NC}"
    exit 1
fi

if [[ -z "$EASEE_CHARGER_ID" ]] || [[ "$EASEE_CHARGER_ID" == "EHXXXXXX" ]]; then
    echo -e "${RED}EASEE_CHARGER_ID not configured in .env${NC}"
    exit 1
fi

# Build if image doesn't exist
if ! docker images | grep -q "easee-controller"; then
    echo -e "${GREEN}Building Docker image...${NC}"
    ./scripts/docker-build.sh
fi

# Start with docker-compose
echo -e "${GREEN}Starting EASEE Controller...${NC}"
docker-compose up -d

echo ""
echo -e "${GREEN}Container started!${NC}"
echo ""
echo "Dashboard: http://localhost:${WEB_PORT:-8080}"
echo ""
echo "Useful commands:"
echo "  View logs:     docker-compose logs -f"
echo "  Stop:          docker-compose down"
echo "  Restart:       docker-compose restart"
echo "  Shell access:  docker-compose exec easee-controller /bin/bash"
