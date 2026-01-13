# Harper - Dockerized Setup

## Quick Start

```bash
# Build and start all services
make build
make up

# View logs
make logs

# Stop services
make down
```

## Services

- **harper-server**: FastAPI backend on port 5000
  - STT: Groq Whisper API
  - TTS: Resemble AI Streaming API
  - WebSocket endpoint: `ws://localhost:5000/ws`

- **harper-client**: Next.js frontend on port 3007
  - Access: http://localhost:3007

## Configuration

1. Copy `.env.example` to `server/.env`:
```bash
cp server/.env.example server/.env
```

2. Edit `server/.env` with your API keys:
```env
# LLM Settings
OPENAI_API_KEY=your_openai_key

# STT Settings
STT_MODEL=groq
STT__GROQ_API_KEY=your_groq_key

# TTS Settings
TTS_MODEL=resemble
TTS__RESEMBLE_API_KEY=your_resemble_key
TTS__RESEMBLE_VOICE_UUID=your_voice_uuid
```

## Available Commands

```bash
make build          # Build Docker images
make up             # Start services in detached mode
make down           # Stop and remove containers
make restart        # Restart all services
make logs           # Follow logs from all services
make logs-server    # Follow server logs only
make logs-client    # Follow client logs only
make ps             # Show running containers
make shell-server   # Open bash in server container
make shell-client   # Open sh in client container
make clean          # Remove everything (containers, volumes, images)
```

## Development

For development with hot reload:

```bash
# Server with auto-reload
docker compose up harper-server

# Client with hot reload (if configured)
docker compose up harper-client
```

## Troubleshooting

### Server not starting
```bash
make logs-server
```

### Client build fails
```bash
docker compose build --no-cache harper-client
```

### Reset everything
```bash
make clean
make build
make up
```
