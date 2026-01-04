# EASEE Smart Charger Controller

A Raspberry Pi-based system for intelligent EV charging management with the EASEE charger.

## Features

- 🔌 **Smart Charging**: Automatically schedules charging during low electricity price periods
- ⚡ **Load Balancing**: Prevents home main fuse overload by monitoring real-time consumption via SaveEye
- 💰 **Price Optimization**: Integrates with Nord Pool/Tibber for electricity price forecasts
- 📊 **Web Dashboard**: Monitor and control your charger from any device
- 🔄 **Automatic Adjustment**: Dynamically adjusts charging current based on home consumption
- 📈 **Statistics & History**: Track charging sessions, costs, and energy consumption

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Raspberry Pi                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                 │
│  │ Price       │  │ SaveEye     │  │ EASEE       │                 │
│  │ Fetcher     │  │ Monitor     │  │ Controller  │                 │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘                 │
│         │                │                │                         │
│         ▼                ▼                ▼                         │
│  ┌─────────────────────────────────────────────┐                   │
│  │           Charging Scheduler                 │                   │
│  │  - Price-based scheduling                    │                   │
│  │  - Load balancing logic                      │                   │
│  │  - Fuse protection                           │                   │
│  └──────────────────┬──────────────────────────┘                   │
│                     │                                               │
│                     ▼                                               │
│  ┌─────────────────────────────────────────────┐                   │
│  │           Web Dashboard (FastAPI)            │                   │
│  └─────────────────────────────────────────────┘                   │
└─────────────────────────────────────────────────────────────────────┘
         │                │                │
         ▼                ▼                ▼
    Nord Pool        SaveEye           EASEE
    /Tibber          Device           Charger
```

## Requirements

### Hardware
- Raspberry Pi 3B+ or newer (4GB+ RAM recommended)
- EASEE charger (Home or Charge)
- SaveEye electricity monitor (connected to your meter's P1 port or via IR)
- Stable Wi-Fi or Ethernet connection

### Software
- Python 3.10+
- Docker & Docker Compose (recommended)
- Or: Raspberry Pi OS (64-bit recommended)

## Installation

### Option 1: Docker (Recommended)

The easiest way to run the controller is with Docker.

#### Quick Start

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/easee-controller.git
cd easee-controller

# Create configuration file
cp config.example.yaml config.yaml
nano config.yaml  # Edit with your settings

# Start with Docker Compose
docker-compose up -d

# View logs
docker-compose logs -f
```

#### Configuration (config.yaml)

All settings are in one structured YAML file:

```yaml
easee:
  username: "your_email@example.com"
  password: "your_password"
  charger_id: "EHXXXXXX"
  max_current: 32

mqtt:
  host: "192.168.1.50"    # Your MQTT broker
  port: 1883

saveeye:
  mode: "mqtt"
  mqtt_topic: "saveeye/telemetry"
  required: false         # Set true to require SaveEye

home:
  main_fuse_amps: 25
  safety_margin_amps: 3
  phases: 3

price:
  source: "nordpool"
  nordpool_area: "NO1"    # Your price area
  currency: "NOK"

smart_charging:
  enabled: true
  price_threshold: 1.5
  preferred_hours: [0, 1, 2, 3, 4, 5, 6]

web:
  port: 8080
```

See `config.example.yaml` for all options with documentation.

#### Building for Raspberry Pi

For ARM-based systems (Raspberry Pi):

```bash
# Build ARM-optimized image
docker build -f Dockerfile.arm -t easee-controller .

# Or use the build script
./scripts/docker-build.sh
```

#### Docker Commands

```bash
# Start
docker-compose up -d

# Stop
docker-compose down

# View logs
docker-compose logs -f

# Restart
docker-compose restart

# Shell access
docker-compose exec easee-controller /bin/bash

# Update
docker-compose pull
docker-compose up -d
```

#### Network Configuration

If the container cannot reach your MQTT broker on the local network:

**Option 1: Host networking**
```yaml
# In docker-compose.override.yml
services:
  easee-controller:
    network_mode: host
```

**Option 2: Access host's localhost**
```yaml
# In docker-compose.override.yml
services:
  easee-controller:
    extra_hosts:
      - "host.docker.internal:host-gateway"
```
Then use `mqtt.host: "host.docker.internal"` in config.yaml.

#### Environment Variable Overrides

Environment variables can override config.yaml (useful for secrets):

```bash
# Override sensitive values
export EASEE_PASSWORD="secret"
export MQTT_PASSWORD="secret"
docker-compose up -d
```

---

### Option 2: Native Installation

#### 1. Clone the Repository

```bash
cd /home/pi
git clone https://github.com/chesteruu/ev_se_charger.git
cd easee-controller
```

#### 2. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

#### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

#### 4. Configure Environment

```bash
cp config.example.env config.env
nano config.env  # Edit with your credentials
```

#### 5. Initialize Database

```bash
python -m src.init_db
```

#### 6. Test the Connection

```bash
python -m src.test_connection
```

#### 7. Run the Application

```bash
python -m src.main
```

## Running as a Service

### Install systemd Service

```bash
sudo cp deploy/easee-controller.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable easee-controller
sudo systemctl start easee-controller
```

### View Logs

```bash
sudo journalctl -u easee-controller -f
```

## Configuration

### Main Configuration Options

| Option | Description | Default |
|--------|-------------|---------|
| `HOME_MAIN_FUSE_AMPS` | Your home's main fuse rating | 25 |
| `HOME_SAFETY_MARGIN_AMPS` | Safety buffer below max | 3 |
| `PRICE_THRESHOLD` | Max price to allow charging | 1.5 |
| `SMART_CHARGING_ENABLED` | Enable price-based scheduling | true |

### Load Balancing

The system continuously monitors your home's electricity consumption via SaveEye.
When total consumption approaches the main fuse limit:

1. **Warning Zone (80%)**: Reduces charging current gradually
2. **Critical Zone (90%)**: Significantly reduces or pauses charging
3. **Danger Zone (95%)**: Immediately stops charging

### Price Optimization

The scheduler analyzes electricity prices for the next 24 hours and:
- Identifies the cheapest charging windows
- Schedules charging during off-peak hours
- Respects your preferred charging times
- Ensures your car is ready by your departure time

## Web Dashboard

Access the dashboard at: `http://raspberry-pi-ip:8080`

Features:
- Real-time charger status
- Current electricity price
- Home consumption graph
- Charging history
- Manual override controls
- Configuration interface

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/status` | GET | Current system status |
| `/api/charger` | GET | Charger state |
| `/api/charger/start` | POST | Start charging |
| `/api/charger/stop` | POST | Stop charging |
| `/api/charger/current` | POST | Set charging current |
| `/api/prices` | GET | Current and upcoming prices |
| `/api/consumption` | GET | Real-time consumption data |
| `/api/schedule` | GET/POST | Charging schedule |
| `/api/history` | GET | Charging history |

## Troubleshooting

### EASEE Connection Issues
- Verify your credentials in `.env`
- Check if your charger is online in the EASEE app
- Ensure your charger ID is correct

### SaveEye Connection Issues
- Verify SaveEye is on the same network
- Check the IP address and port
- Ensure P1 port is properly connected

### Price Data Issues
- Verify your area code (e.g., NO1, SE3)
- Check internet connectivity
- Some areas may have delayed data

## Contributing

Contributions are welcome! Please read our contributing guidelines and submit pull requests.

## License

MIT License - see LICENSE file for details

## Acknowledgments

- [pyeasee](https://github.com/nordicopen/pyeasee) - EASEE API wrapper
- [nordpool](https://github.com/custom-components/nordpool) - Nord Pool integration
- [SaveEye](https://saveeye.eu/) - Real-time electricity monitoring
