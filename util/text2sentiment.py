from dataclasses import dataclass
from typing import Dict, List, Literal
import os
import nltk

# Optional transformers sentiment pipeline
try:
    from transformers import pipeline  # type: ignore
    _sentiment_classifier = pipeline("sentiment-analysis")
except Exception:
    _sentiment_classifier = None  # Fallback if transformers/torch not installed

from textblob import TextBlob


def setup_nltk_data():
    # Construct relative path to your local NLTK data folder, e.g. "./data/nltk_data"
    relative_path = "datasets/textblob/nltk_data"

    # Append this path to NLTK data search paths
    nltk.data.path.append(relative_path)

    # Alternatively, you can set the environment variable (before any NLTK/TextBlob calls)
    os.environ["NLTK_DATA"] = relative_path


setup_nltk_data()


@dataclass
class SentimentSummary:
    """
    A summary of sentiment analysis results combining TextBlob and Hugging Face outputs.

    Attributes:
        textblob_polarity (float): Polarity score from TextBlob, ranging from -1.0 to 1.0.
            -1.0 means extremely negative sentiment,
            0 means neutral sentiment,
            1.0 means extremely positive sentiment.
        textblob_subjectivity (float): Subjectivity score from TextBlob, ranging from 0.0 to 1.0.
            0.0 means very objective (fact-based),
            1.0 means very subjective (opinionated).
        huggingface_label (str): Sentiment label predicted by the Hugging Face model.
            Examples: "POSITIVE" indicates positive sentiment,
                    "NEGATIVE" indicates negative sentiment.
        huggingface_score (float): Confidence score of the Hugging Face prediction, between 0 and 1.
            0 means no confidence,
            1 means full confidence in the predicted label.
    """

    textblob_polarity: float
    textblob_subjectivity: float
    huggingface_label: str
    huggingface_score: float

    def __str__(self) -> str:
        """
        Return a human-readable summary string of the combined sentiment analysis,
        showing polarity, subjectivity, Hugging Face label, and Hugging Face score.
        """
        return (
            f"TextBlob polarity={self.textblob_polarity:.2f}, "
            f"subjectivity={self.textblob_subjectivity:.2f}; "
            f"HuggingFace label={self.huggingface_label}, "
            f"score={self.huggingface_score:.2f}"
        )


def text2sentiment(text_list: List[str]) -> List[SentimentSummary]:
    results: List[SentimentSummary] = []
    for text in text_list:
        text = (text or "").strip()
        if not text:
            results.append(
                SentimentSummary(
                    textblob_polarity=0.0,
                    textblob_subjectivity=0.0,
                    huggingface_label="NEUTRAL",
                    huggingface_score=0.0,
                )
            )
            continue

        # TextBlob sentiment
        sentiment_classical = TextBlob(text).sentiment
        polarity: float = float(sentiment_classical.polarity)
        subjectivity: float = float(sentiment_classical.subjectivity)

        # Hugging Face sentiment (optional)
        if _sentiment_classifier is not None:
            try:
                sentiment_deep: List[Dict] = _sentiment_classifier(text)
                hf_result = sentiment_deep[0]
                hf_label = str(hf_result.get("label", "NEUTRAL")).upper()
                hf_score = float(hf_result.get("score", 0.0))
            except Exception:
                # Fallback to TextBlob-only inference
                hf_label = "POSITIVE" if polarity > 0.05 else ("NEGATIVE" if polarity < -0.05 else "NEUTRAL")
                hf_score = min(1.0, max(0.0, abs(polarity)))
        else:
            # Transformers not available
            hf_label = "POSITIVE" if polarity > 0.05 else ("NEGATIVE" if polarity < -0.05 else "NEUTRAL")
            hf_score = min(1.0, max(0.0, abs(polarity)))

        # Combine into SentimentSummary
        summary = SentimentSummary(
            textblob_polarity=polarity,
            textblob_subjectivity=subjectivity,
            huggingface_label=hf_label,
            huggingface_score=hf_score,
        )
        results.append(summary)
    return results


# Example usage
if __name__ == "__main__":
    sample_texts = [
        "The service was terrible and the food was cold.",
        "I absolutely loved the movie!",
        "It was okay, nothing special.",
    ]

    sentiment_summaries = text2sentiment(sample_texts)
    for summary in sentiment_summaries:
        print(summary)
