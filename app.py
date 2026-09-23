import io
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from wordcloud import WordCloud

from src.chat_parser import parse_whatsapp_chat
from src.sentiment_analysis import analyze_messages


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="WhatsApp Chat Analyzer",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------
# Custom styling
# ---------------------------------------------------------
st.markdown(
    """
    <style>
        .main-title {
            font-size: 2.6rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
        }

        .subtitle {
            color: #6b7280;
            font-size: 1.05rem;
            margin-bottom: 1.5rem;
        }

        .metric-card {
            padding: 1rem 1.1rem;
            border-radius: 12px;
            border: 1px solid rgba(128,128,128,0.20);
            background: rgba(128,128,128,0.05);
            min-height: 115px;
        }

        .metric-label {
            font-size: 0.85rem;
            color: #6b7280;
            margin-bottom: 0.3rem;
        }

        .metric-value {
            font-size: 1.65rem;
            font-weight: 700;
        }

        .section-title {
            font-size: 1.35rem;
            font-weight: 650;
            margin-top: 0.5rem;
            margin-bottom: 0.7rem;
        }

        div[data-testid="stMetric"] {
            border: 1px solid rgba(128,128,128,0.20);
            border-radius: 12px;
            padding: 12px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------
def save_uploaded_file(uploaded_file):
    """Save an uploaded Streamlit file to a temporary file."""
    suffix = Path(uploaded_file.name).suffix or ".txt"

    temp_file = tempfile.NamedTemporaryFile(
        mode="wb",
        suffix=suffix,
        delete=False,
    )

    temp_file.write(uploaded_file.getvalue())
    temp_file.close()

    return Path(temp_file.name)


def calculate_overall_sentiment(df):
    """Match the sentiment logic used in the original main.py."""
    positive = df["Positive"].sum()
    negative = df["Negative"].sum()
    neutral = df["Neutral"].sum()

    if positive > negative and positive > neutral:
        return "😊 Positive"
    elif negative > positive and negative > neutral:
        return "😠 Negative"
    return "🙂 Neutral"


def prepare_dataframe(df):
    """Add useful analysis columns without changing the original sentiment columns."""
    result = df.copy()

    result["Date"] = pd.to_datetime(result["Date"], errors="coerce")
    result["Message"] = result["Message"].fillna("").astype(str)

    # Convert time information into a useful hour column.
    result["Hour"] = pd.to_datetime(
        result["Time"].astype(str),
        errors="coerce",
    ).dt.hour

    # Some WhatsApp exports may use 12-hour time strings.
    if result["Hour"].isna().all():
        result["Hour"] = pd.to_datetime(
            result["Time"].astype(str),
            format="%I:%M %p",
            errors="coerce",
        ).dt.hour

    result["Day"] = result["Date"].dt.day_name()
    result["Month"] = result["Date"].dt.to_period("M").astype(str)

    return result


def show_metric(label, value):
    """Render a consistent dashboard metric."""
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def create_sentiment_chart(df):
    summary = df[["Positive", "Negative", "Neutral"]].sum()

    fig, ax = plt.subplots(figsize=(7, 4.5))
    summary.plot(kind="bar", ax=ax)
    ax.set_title("Overall Sentiment Scores")
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Score")
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()

    return fig


def create_user_chart(df):
    user_counts = (
        df["Author"]
        .fillna("System / Unknown")
        .value_counts()
        .head(15)
        .sort_values()
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    user_counts.plot(kind="barh", ax=ax)
    ax.set_title("Messages by Participant")
    ax.set_xlabel("Number of Messages")
    ax.set_ylabel("Participant")
    fig.tight_layout()

    return fig


def create_daily_chart(df):
    daily = (
        df.dropna(subset=["Date"])
        .groupby("Date")
        .size()
    )

    fig, ax = plt.subplots(figsize=(9, 4.5))
    daily.plot(ax=ax)
    ax.set_title("Messages Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Messages")
    fig.tight_layout()

    return fig


def create_hourly_chart(df):
    hourly = (
        df.dropna(subset=["Hour"])
        .groupby("Hour")
        .size()
        .reindex(range(24), fill_value=0)
    )

    fig, ax = plt.subplots(figsize=(9, 4.5))
    hourly.plot(kind="bar", ax=ax)
    ax.set_title("Messages by Hour")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Messages")
    ax.set_xticks(range(24))
    ax.set_xticklabels([str(i) for i in range(24)], rotation=0)
    fig.tight_layout()

    return fig


def create_wordcloud(df):
    text = " ".join(
        df["Message"]
        .dropna()
        .astype(str)
        .tolist()
    ).strip()

    if not text:
        return None

    wordcloud = WordCloud(
        width=1200,
        height=600,
        background_color="white",
    ).generate(text)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    ax.set_title("WhatsApp Chat Word Cloud")
    fig.tight_layout()

    return fig


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("## 💬 WhatsApp Analyzer")
    st.caption("Chat analytics and sentiment analysis")

    uploaded_file = st.file_uploader(
        "Upload WhatsApp chat export",
        type=["txt"],
        help="Export your WhatsApp conversation as a .txt file and upload it here.",
    )

    analyze_button = st.button(
        "🚀 Analyze Chat",
        type="primary",
        width="stretch",
    )

    st.divider()

    st.markdown("### 📌 About")
    st.write(
        "This dashboard uses the project's existing WhatsApp parser "
        "and VADER sentiment analysis pipeline."
    )

    st.markdown("### 🛠️ Technologies")
    st.write(
        "Python • Pandas • NLTK/VADER • Matplotlib • WordCloud • Streamlit"
    )

    st.divider()
    st.caption("WhatsApp Chat Analysis Project")


# ---------------------------------------------------------
# Header
# ---------------------------------------------------------
st.markdown(
    '<div class="main-title">💬 WhatsApp Chat Analysis</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="subtitle">Analyze conversation patterns, participant activity, sentiment, and frequently used words.</div>',
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Empty state
# ---------------------------------------------------------
if uploaded_file is None:
    st.info(
        "👈 Upload a WhatsApp `.txt` chat export from the sidebar to begin analysis."
    )

    st.markdown("### How it works")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("**1. 📂 Upload**")
        st.caption("Upload your exported WhatsApp chat.")

    with col2:
        st.markdown("**2. 🔎 Parse**")
        st.caption("Extract dates, times, participants, and messages.")

    with col3:
        st.markdown("**3. 🧠 Analyze**")
        st.caption("Calculate sentiment scores using VADER.")

    with col4:
        st.markdown("**4. 📊 Visualize**")
        st.caption("Explore charts, statistics, and a word cloud.")

    st.stop()


# ---------------------------------------------------------
# Analyze uploaded chat
# ---------------------------------------------------------
if analyze_button:
    try:
        with st.spinner("Analyzing your WhatsApp chat..."):
            temp_path = save_uploaded_file(uploaded_file)

            try:
                messages = parse_whatsapp_chat(temp_path)
            finally:
                try:
                    temp_path.unlink(missing_ok=True)
                except Exception:
                    pass

            if not messages:
                st.error(
                    "No messages were detected. Please check that the uploaded "
                    "file is a WhatsApp exported `.txt` chat."
                )
                st.stop()

            df = analyze_messages(messages)
            df = prepare_dataframe(df)

            st.session_state["analysis_df"] = df
            st.session_state["uploaded_name"] = uploaded_file.name

        st.success("✅ Chat analysis completed successfully.")

    except UnicodeDecodeError:
        st.error(
            "The uploaded file could not be decoded as UTF-8. "
            "Please upload the original WhatsApp exported `.txt` file."
        )
        st.stop()

    except Exception as exc:
        st.error(f"Analysis failed: {exc}")
        st.exception(exc)
        st.stop()


# ---------------------------------------------------------
# Use previously analyzed data after Streamlit reruns
# ---------------------------------------------------------
if "analysis_df" not in st.session_state:
    st.warning("Click **Analyze Chat** to process the uploaded file.")
    st.stop()

df = st.session_state["analysis_df"]
uploaded_name = st.session_state.get("uploaded_name", uploaded_file.name)

# ---------------------------------------------------------
# Dashboard metrics
# ---------------------------------------------------------
total_messages = len(df)
participants = df["Author"].dropna().nunique()
overall = calculate_overall_sentiment(df)

valid_dates = df["Date"].dropna()

if not valid_dates.empty:
    date_range = f"{valid_dates.min().strftime('%d %b %Y')} – {valid_dates.max().strftime('%d %b %Y')}"
else:
    date_range = "Not available"

st.markdown("### 📊 Dashboard Overview")

m1, m2, m3, m4 = st.columns(4)

with m1:
    show_metric("💬 Total Messages", f"{total_messages:,}")

with m2:
    show_metric("👥 Participants", f"{participants:,}")

with m3:
    show_metric("😊 Overall Sentiment", overall)

with m4:
    show_metric("📅 Chat Period", date_range)


st.caption(f"Analyzed file: **{uploaded_name}**")


# ---------------------------------------------------------
# Tabs
# ---------------------------------------------------------
tab_overview, tab_sentiment, tab_activity, tab_words, tab_data = st.tabs(
    [
        "📊 Overview",
        "😊 Sentiment",
        "📈 Activity",
        "☁️ Word Cloud",
        "🗃️ Data",
    ]
)


# ---------------------------------------------------------
# Overview tab
# ---------------------------------------------------------
with tab_overview:
    st.markdown("### Participant Activity")

    left, right = st.columns([1.4, 1])

    with left:
        if df["Author"].notna().any():
            user_chart = create_user_chart(df)
            st.pyplot(user_chart, width="stretch")
            plt.close(user_chart)

    with right:
        st.markdown("#### Most Active Participants")

        participant_counts = (
            df["Author"]
            .fillna("System / Unknown")
            .value_counts()
            .rename_axis("Participant")
            .reset_index(name="Messages")
        )

        st.dataframe(
            participant_counts.head(10),
            hide_index=True,
            width="stretch",
        )

        st.markdown("#### Chat Statistics")

        stats = pd.DataFrame(
            {
                "Metric": [
                    "Total messages",
                    "Participants",
                    "Average message count/user",
                    "Messages with author",
                    "Messages without author",
                ],
                "Value": [
                    f"{len(df):,}",
                    f"{participants:,}",
                    f"{(len(df) / participants):.2f}" if participants else "0",
                    f"{df['Author'].notna().sum():,}",
                    f"{df['Author'].isna().sum():,}",
                ],
            }
        )

        st.dataframe(
            stats,
            hide_index=True,
            width="stretch",
        )


# ---------------------------------------------------------
# Sentiment tab
# ---------------------------------------------------------
with tab_sentiment:
    st.markdown("### 😊 Sentiment Analysis")

    positive_score = df["Positive"].sum()
    negative_score = df["Negative"].sum()
    neutral_score = df["Neutral"].sum()

    s1, s2, s3 = st.columns(3)

    with s1:
        show_metric("Positive Score", f"{positive_score:.2f}")

    with s2:
        show_metric("Negative Score", f"{negative_score:.2f}")

    with s3:
        show_metric("Neutral Score", f"{neutral_score:.2f}")

    st.markdown("#### Sentiment Summary")

    sentiment_fig = create_sentiment_chart(df)
    st.pyplot(sentiment_fig, width="stretch")
    plt.close(sentiment_fig)

    st.info(
        "The sentiment calculation follows your existing project code, "
        "which uses NLTK's VADER SentimentIntensityAnalyzer to calculate "
        "positive, negative, and neutral scores."
    )

    if df["Author"].notna().any():
        st.markdown("#### Participant-wise Sentiment Scores")

        user_sentiment = (
            df.groupby(df["Author"].fillna("System / Unknown"))[
                ["Positive", "Negative", "Neutral"]
            ]
            .sum()
            .sort_values("Positive", ascending=False)
        )

        st.dataframe(
            user_sentiment,
            width="stretch",
        )


# ---------------------------------------------------------
# Activity tab
# ---------------------------------------------------------
with tab_activity:
    st.markdown("### 📈 Chat Activity")

    activity_left, activity_right = st.columns(2)

    with activity_left:
        st.markdown("#### Daily Activity")
        daily_fig = create_daily_chart(df)
        st.pyplot(daily_fig, width="stretch")
        plt.close(daily_fig)

    with activity_right:
        st.markdown("#### Hourly Activity")
        hourly_fig = create_hourly_chart(df)
        st.pyplot(hourly_fig, width="stretch")
        plt.close(hourly_fig)

    if df["Day"].notna().any():
        st.markdown("#### Messages by Day of Week")

        weekday_order = [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ]

        weekday_counts = (
            df["Day"]
            .value_counts()
            .reindex(weekday_order, fill_value=0)
        )

        fig, ax = plt.subplots(figsize=(9, 4.5))
        weekday_counts.plot(kind="bar", ax=ax)
        ax.set_title("Messages by Day of Week")
        ax.set_xlabel("Day")
        ax.set_ylabel("Messages")
        ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()

        st.pyplot(fig, width="stretch")
        plt.close(fig)


# ---------------------------------------------------------
# Word cloud tab
# ---------------------------------------------------------
with tab_words:
    st.markdown("### ☁️ Frequently Used Words")

    wordcloud_fig = create_wordcloud(df)

    if wordcloud_fig is None:
        st.warning("No message text is available for the word cloud.")
    else:
        st.pyplot(wordcloud_fig, width="stretch")
        plt.close(wordcloud_fig)

        st.caption(
            "The word cloud is generated from the message text using the "
            "WordCloud library."
        )


# ---------------------------------------------------------
# Data tab
# ---------------------------------------------------------
with tab_data:
    st.markdown("### 🗃️ Analyzed Messages")

    col1, col2 = st.columns([3, 1])

    with col1:
        st.write(f"Showing **{len(df):,}** analyzed messages.")

    with col2:
        csv_data = df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="⬇️ Download CSV",
            data=csv_data,
            file_name="whatsapp_analysis.csv",
            mime="text/csv",
            type="primary",
            width="stretch",
        )

    st.dataframe(
        df,
        width="stretch",
        height=500,
        hide_index=True,
    )


# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.divider()

st.caption(
    "WhatsApp Chat Analysis • Python • Pandas • NLTK/VADER • "
    "Matplotlib • WordCloud • Streamlit"
)
