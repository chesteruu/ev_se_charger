#!/bin/bash
#
# Build Docker images for EASEE Smart Charger Controller
#

set -e

# Configuration
IMAGE_NAME="easee-controller"
VERSION="${VERSION:-latest}"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}Building EASEE Smart Charger Controller Docker image${NC}"
echo "Image: ${IMAGE_NAME}:${VERSION}"
echo ""

# Detect architecture
ARCH=$(uname -m)
echo "Detected architecture: ${ARCH}"

# Choose Dockerfile based on architecture
if [[ "$ARCH" == "aarch64" ]] || [[ "$ARCH" == "arm64" ]] || [[ "$ARCH" == "armv7l" ]]; then
    echo -e "${YELLOW}Using ARM-optimized Dockerfile${NC}"
    DOCKERFILE="Dockerfile.arm"
else
    echo "Using standard Dockerfile"
    DOCKERFILE="Dockerfile"
fi

# Build the image
echo ""
echo "Building image..."
docker build \
    -f "${DOCKERFILE}" \
    -t "${IMAGE_NAME}:${VERSION}" \
    -t "${IMAGE_NAME}:latest" \
    .

echo ""
echo -e "${GREEN}Build complete!${NC}"
echo ""
echo "To run the container:"
echo "  docker-compose up -d"
echo ""
echo "Or manually:"
echo "  docker run -d \\"
echo "    --name easee-controller \\"
echo "    -p 8080:8080 \\"
echo "    -e EASEE_USERNAME=your_email \\"
echo "    -e EASEE_PASSWORD=your_password \\"
echo "    -e EASEE_CHARGER_ID=EHXXXXXX \\"
echo "    -e SAVEEYE_HOST=192.168.1.100 \\"
echo "    -v easee-data:/data \\"
echo "    ${IMAGE_NAME}:${VERSION}"
