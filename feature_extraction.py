"""
Feature Extraction Pipeline
Extracts textual, URL-based, and behavioral features from email content.
"""

import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix


# Phishing-related keywords
PHISHING_KEYWORDS = [
    "urgent", "immediately", "verify", "suspend", "account", "password",
    "click", "confirm", "update", "expire", "unauthorized", "security",
    "alert", "warning", "locked", "compromised", "unusual", "activity",
    "billing", "payment", "invoice", "refund", "prize", "winner",
    "congratulations", "gift", "offer", "limited", "act now", "free",
    "login", "credentials", "ssn", "social security", "bank",
    "final notice", "failure", "restricted", "disabled",
]

URGENCY_PHRASES = [
    "act now", "immediate action", "within 24 hours", "within 48 hours",
    "don't delay", "time is running out", "final warning", "last chance",
    "before it's too late", "hurry", "expires soon", "limited time",
    "right away", "as soon as possible", "without delay",
]

URL_SHORTENERS = ["bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "is.gd", "buff.ly"]


class FeatureExtractor:
    """
    Extracts features from email text for phishing detection.

    Features extracted:
    - url_count: Number of URLs in the email
    - suspicious_url_count: URLs with IP addresses, shorteners, or typosquatting
    - has_https: Whether any URL uses HTTPS
    - phishing_keyword_count: Count of phishing-related keywords
    - urgency_score: Count of urgency-related phrases
    - capital_ratio: Ratio of uppercase letters to total letters
    - special_char_count: Count of special characters
    - exclamation_count: Count of exclamation marks
    - body_length: Length of the email body
    - has_html_tags: Whether HTML tags are present
    - has_attachment_mention: Whether attachments are mentioned
    - TF-IDF features: Top textual patterns from email content
    """

    def __init__(self, max_tfidf_features=100):
        self.tfidf = TfidfVectorizer(
            max_features=max_tfidf_features,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
        )
        self._fitted = False
        self.feature_names_ = []

    def _extract_urls(self, text):
        """Extract all URLs from text."""
        url_pattern = r'https?://[^\s<>"\']+' 
        return re.findall(url_pattern, text, re.IGNORECASE)

    def _count_suspicious_urls(self, urls):
        """Count URLs that exhibit suspicious patterns."""
        suspicious = 0
        for url in urls:
            # Check for IP address in URL
            if re.search(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', url):
                suspicious += 1
                continue
            # Check for URL shorteners
            if any(shortener in url.lower() for shortener in URL_SHORTENERS):
                suspicious += 1
                continue
            # Check for excessive subdomains (3+)
            domain_part = url.split("//")[-1].split("/")[0]
            if domain_part.count(".") >= 4:
                suspicious += 1
                continue
            # Check for suspicious TLDs
            suspicious_tlds = [".xyz", ".tk", ".ru", ".cn", ".top", ".info", ".co"]
            if any(url.lower().endswith(tld) or tld + "/" in url.lower() for tld in suspicious_tlds):
                suspicious += 1
                continue
            # Check for very long URLs (>75 chars)
            if len(url) > 75:
                suspicious += 1
        return suspicious

    def _count_keywords(self, text, keywords):
        """Count occurrences of keywords in text."""
        text_lower = text.lower()
        return sum(1 for kw in keywords if kw in text_lower)

    def _capital_ratio(self, text):
        """Calculate ratio of uppercase letters."""
        letters = [c for c in text if c.isalpha()]
        if not letters:
            return 0.0
        return sum(1 for c in letters if c.isupper()) / len(letters)

    def _count_special_chars(self, text):
        """Count special characters."""
        return sum(1 for c in text if not c.isalnum() and not c.isspace())

    def _extract_handcrafted_features(self, texts):
        """Extract all handcrafted features from a list of texts."""
        features = []
        for text in texts:
            urls = self._extract_urls(text)
            feat = {
                "url_count": len(urls),
                "suspicious_url_count": self._count_suspicious_urls(urls),
                "has_https": int(any("https://" in u for u in urls)),
                "phishing_keyword_count": self._count_keywords(text, PHISHING_KEYWORDS),
                "urgency_score": self._count_keywords(text, URGENCY_PHRASES),
                "capital_ratio": self._capital_ratio(text),
                "special_char_count": self._count_special_chars(text),
                "exclamation_count": text.count("!"),
                "body_length": len(text),
                "has_html_tags": int(bool(re.search(r"<[^>]+>", text))),
                "has_attachment_mention": int(
                    any(w in text.lower() for w in ["attached", "attachment", "enclosed", "see attached"])
                ),
            }
            features.append(feat)
        return pd.DataFrame(features)

    def fit_transform(self, texts):
        """
        Fit the feature extractor and transform texts.

        Args:
            texts: List or Series of email text strings.

        Returns:
            Feature matrix (scipy sparse matrix).
        """
        texts = list(texts)

        # Handcrafted features
        handcrafted_df = self._extract_handcrafted_features(texts)

        # TF-IDF features
        tfidf_matrix = self.tfidf.fit_transform(texts)

        self._fitted = True

        # Build feature names list
        self.feature_names_ = list(handcrafted_df.columns) + [
            f"tfidf_{name}" for name in self.tfidf.get_feature_names_out()
        ]

        # Combine: handcrafted + TF-IDF
        handcrafted_sparse = csr_matrix(handcrafted_df.values)
        return hstack([handcrafted_sparse, tfidf_matrix])

    def transform(self, texts):
        """
        Transform texts using the fitted extractor.

        Args:
            texts: List or Series of email text strings.

        Returns:
            Feature matrix (scipy sparse matrix).
        """
        if not self._fitted:
            raise RuntimeError("FeatureExtractor must be fitted before calling transform().")

        texts = list(texts)
        handcrafted_df = self._extract_handcrafted_features(texts)
        tfidf_matrix = self.tfidf.transform(texts)

        handcrafted_sparse = csr_matrix(handcrafted_df.values)
        return hstack([handcrafted_sparse, tfidf_matrix])
