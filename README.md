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

## mendatory for file run
Place your exported WhatsApp chat file in the `data/` folder before running the script.

# For iPhone:
- Open your chat with a person or a group
- Just tap on the profile of the person or the group
- You will see an option to export chat down below
# For Android:
- Open your chat with a person or a group 
- Click on the three dots above 
- Click on more
- Click on the export chat

  
--- save ---
as **whatsapp.txt** and paste it on **data folder**
## Run


```bash
python main.py --input data/whatsapp.txt
```
 


