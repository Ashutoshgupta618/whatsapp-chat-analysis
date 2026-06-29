# WhatsApp Sentiment Analysis

This project parses a WhatsApp chat export and performs basic sentiment analysis on each message.

## Structure
- `main.py` runs the full workflow.
- `src/chat_parser.py` parses the chat format.
- `src/sentiment_analysis.py` computes sentiment scores.
- `src/visualization.py` creates simple charts.

## Setup

```bash
python -m pip install -r requirements.txt
```

## Run

```bash
python main.py --input data/whatsapp.txt
```

Place your exported WhatsApp chat file in the `data/` folder before running the script.
