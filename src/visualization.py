import matplotlib.pyplot as plt
from wordcloud import WordCloud


def save_sentiment_summary(df, output_path):
    """Save and display sentiment summary graph."""

    summary = df[["Positive", "Negative", "Neutral"]].sum()

    fig, ax = plt.subplots(figsize=(7, 5))

    summary.plot(
        kind="bar",
        ax=ax,
        color=["green", "red", "gray"]
    )

    ax.set_title("Sentiment Summary")
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Score")
    ax.set_xticklabels(summary.index, rotation=0)

    plt.tight_layout()

    fig.savefig(output_path, dpi=300)

    # Display graph automatically
    plt.show()

    plt.close(fig)


def save_wordcloud(df, output_path):
    """Save and display word cloud."""

    text = " ".join(df["Message"].dropna().astype(str))

    wordcloud = WordCloud(
        width=1000,
        height=500,
        background_color="white"
    ).generate(text)

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    ax.set_title("Word Cloud")

    plt.tight_layout()

    fig.savefig(output_path, dpi=300)

    # Display word cloud automatically
    plt.show()

    plt.close(fig)