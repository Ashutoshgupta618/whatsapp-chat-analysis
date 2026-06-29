from pathlib import Path
import argparse

from src.chat_parser import parse_whatsapp_chat
from src.sentiment_analysis import analyze_messages
from src.visualization import (
    save_sentiment_summary,
    save_wordcloud,
)


def overall_sentiment(df):
    positive = df["Positive"].sum()
    negative = df["Negative"].sum()
    neutral = df["Neutral"].sum()

    print("\nOverall Chat Sentiment")

    if positive > negative and positive > neutral:
        print("😊 Positive")

    elif negative > positive and negative > neutral:
        print("😠 Negative")

    else:
        print("🙂 Neutral")


def main():

    parser = argparse.ArgumentParser(
        description="WhatsApp Chat Sentiment Analyzer"
    )

    parser.add_argument(
        "--input",
        default="data/whatsapp.txt",
        help="Path to WhatsApp chat export",
    )

    parser.add_argument(
        "--output-csv",
        default="output/messages.csv",
        help="CSV output file",
    )

    parser.add_argument(
        "--output-plot",
        default="output/sentiment_summary.png",
        help="Sentiment graph image",
    )

    parser.add_argument(
        "--output-wordcloud",
        default="output/wordcloud.png",
        help="Word cloud image",
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help="Display graph and word cloud",
    )

    args = parser.parse_args()

    input_path = Path(args.input)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Chat file not found: {input_path}"
        )

    messages = parse_whatsapp_chat(input_path)

    df = analyze_messages(messages)

    output_dir = Path(args.output_csv).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(args.output_csv, index=False)

    save_sentiment_summary(
        df,
        args.output_plot,
        
    )

    save_wordcloud(
        df,
        args.output_wordcloud,
        
    )

    print("\nFirst 5 Messages")
    print(df.head())

    print(f"\nCSV saved to: {args.output_csv}")
    print(f"Graph saved to: {args.output_plot}")
    print(f"Word cloud saved to: {args.output_wordcloud}")

    overall_sentiment(df)


if __name__ == "__main__":
    main()