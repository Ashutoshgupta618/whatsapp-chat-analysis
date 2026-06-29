import nltk
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download("vader_lexicon", quiet=True)


def analyze_messages(messages):
    df = pd.DataFrame(messages, columns=["Date", "Time", "Author", "Message"])
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Message"]).copy()

    sentiments = SentimentIntensityAnalyzer()
    df["Positive"] = [sentiments.polarity_scores(message)["pos"] for message in df["Message"]]
    df["Negative"] = [sentiments.polarity_scores(message)["neg"] for message in df["Message"]]
    df["Neutral"] = [sentiments.polarity_scores(message)["neu"] for message in df["Message"]]

    return df
