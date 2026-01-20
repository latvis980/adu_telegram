# ADUmedia Telegram Publisher

Standalone Railway service that publishes architecture news to Telegram.

## Overview

This service fetches today's processed articles from Cloudflare R2 storage and sends them to the ADUmedia Telegram channel with proper rate limiting and flood control.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     ADUmedia News System                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  adu_rss (18:45)  →  ┐                                         │
│                      ├──→  R2 Storage  ──→  adu_telegram       │
│  adu_scrapers (18:30)┘     (candidates)     (19:00)            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Features

- Fetches articles from R2 candidates or selected digest
- Sends to Telegram with hero images
- Rate limiting (3.5s between messages)
- Automatic retry with exponential backoff
- Flood control for Telegram API limits

## Project Structure

```
adu_telegram/
├── run_telegram.py      # Main entry point
├── telegram_bot.py      # Telegram bot with flood control
├── storage/
│   └── r2.py           # R2 storage client (read-only)
├── config/
│   └── sources.py      # Source name mappings
├── requirements.txt
└── README.md
```

## Environment Variables

Set these in Railway dashboard:

| Variable | Description |
|----------|-------------|
| `TELEGRAM_BOT_TOKEN` | Bot token from @BotFather |
| `TELEGRAM_CHANNEL_ID` | Channel ID (@channel or -100xxx) |
| `R2_ACCOUNT_ID` | Cloudflare R2 account ID |
| `R2_ACCESS_KEY_ID` | R2 access key |
| `R2_SECRET_ACCESS_KEY` | R2 secret key |
| `R2_BUCKET_NAME` | R2 bucket name (adumedia) |
| `R2_PUBLIC_URL` | Public URL for images (optional) |

## Railway Configuration

### Start Command
```
python run_telegram.py
```

### Cron Schedule
Run after RSS and custom scrapers complete:
```
0 19 * * *
```
(19:00 UTC daily, adjust for timezone)

## Usage

### Run normally (today's articles)
```bash
python run_telegram.py
```

### Test connections
```bash
python run_telegram.py --test
```

### Publish specific date
```bash
python run_telegram.py --date 2026-01-20
```

### Dry run (no sending)
```bash
python run_telegram.py --dry-run
```

### Limit articles
```bash
python run_telegram.py --limit 5
```

### Use curated selection
```bash
python run_telegram.py --use-selected
```

## R2 Folder Structure

Articles are stored in this structure:

```
adumedia/
└── 2026/
    └── January/
        └── Week-3/
            └── 2026-01-20/
                ├── candidates/
                │   ├── manifest.json
                │   ├── archdaily_001.json
                │   ├── dezeen_001.json
                │   └── images/
                │       ├── archdaily_001.jpg
                │       └── dezeen_001.jpg
                └── selected/
                    └── digest.json
```

## Telegram Rate Limits

- Channel limit: 20 messages/minute
- Safe interval: 3.5 seconds between messages
- 10 articles ≈ 35 seconds
- 50 articles ≈ 3 minutes

## License

MIT
