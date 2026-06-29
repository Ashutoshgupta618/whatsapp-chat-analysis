from pathlib import Path
import argparse

from src.chat_parser import parse_whatsapp_chat
from src.sentiment_analysis import analyze_messages
from src.visualization import save_sentiment_summary, save_wordcloud


def main():
    parser = argparse.ArgumentParser(description="Analyze sentiment in a WhatsApp chat export")
    parser.add_argument("--input", default="data/whatsapp.txt", help="Path to the WhatsApp chat export")
    parser.add_argument("--output-csv", default="output/messages.csv", help="Where to save the analyzed messages")
    parser.add_argument("--output-plot", default="output/sentiment_summary.png", help="Where to save the sentiment summary plot")
    parser.add_argument("--output-wordcloud", default="output/wordcloud.png", help="Where to save the word cloud")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Chat file not found: {input_path}")

    messages = parse_whatsapp_chat(input_path)
    df = analyze_messages(messages)    cd whatsapp_sentiment_analysis
    git add .
    git commit -m "Initial commit"
    git branch -M main
    git remote add origin <your-github-repo-url>
    git push -u origin main

    output_dir = Path(args.output_csv).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(args.output_csv, index=False)
    save_sentiment_summary(df, args.output_plot)
    save_wordcloud(df, args.output_wordcloud)

    print(df.head())
    print(f"Saved results to {args.output_csv}")
    print(f"Saved summary plot to {args.output_plot}")
    print(f"Saved word cloud to {args.output_wordcloud}")


if __name__ == "__main__":
    main()
