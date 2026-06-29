import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from wordcloud import WordCloud


def save_sentiment_summary(df, output_path):
    summary = df[["Positive", "Negative", "Neutral"]].sum()
    fig, ax = plt.subplots(figsize=(6, 4))
    summary.plot(kind="bar", ax=ax, color=["green", "red", "gray"])
    ax.set_title("Sentiment Summary")
    ax.set_ylabel("Score")
    ax.set_xticks(range(len(summary.index)))
    ax.set_xticklabels(summary.index, rotation=0)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_wordcloud(df, output_path):
    text = " ".join(df["Message"].dropna().tolist())
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
