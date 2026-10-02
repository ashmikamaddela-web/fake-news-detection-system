import streamlit as st
import pandas as pd
import re
from urllib.parse import urlparse

# ==========================================
# PAGE SETTINGS
# ==========================================

st.set_page_config(
    page_title="Fake News Detection System",
    page_icon="📰",
    layout="wide"
)

# ==========================================
# LOAD DATASET
# ==========================================

fake_df = pd.read_csv("politifact_fake.csv")
real_df = pd.read_csv("politifact_real.csv")

fake_df["label"] = "fake"
real_df["label"] = "real"

news_df = pd.concat(
    [fake_df, real_df],
    ignore_index=True
)

news_df = news_df[["id", "news_url", "title", "label"]]

# ==========================================
# TEXT PREPROCESSING
# ==========================================

def remove_punctuation(text):
    return re.sub(r"[^a-zA-Z0-9\s]", "", text)

# ==========================================
# SUSPICIOUS WORDS
# ==========================================

suspicious_words = [
    "breaking",
    "shocking",
    "unbelievable",
    "you wont believe",
    "secret",
    "miracle",
    "exposed",
    "scandal",
    "urgent",
    "must see",
    "revealed",
    "amazing",
    "incredible",
    "alert",
    "doctors dont want you to know"
]

def find_suspicious_words(text):
    found_words = []

    for word in suspicious_words:
        if word in text:
            found_words.append(word)

    return found_words

# ==========================================
# DOMAIN EXTRACTION
# ==========================================

def extract_domain(url):
    if pd.isna(url):
        return ""

    url = str(url).strip()

    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    domain = urlparse(url).netloc.lower()

    domain = domain.replace("www.", "")

    return domain

# ==========================================
# DOMAIN ANALYSIS
# ==========================================

news_df["domain"] = news_df["news_url"].apply(extract_domain)

domain_label_counts = pd.crosstab(
    news_df["domain"],
    news_df["label"]
)

domain_stats = domain_label_counts.copy()

if "fake" not in domain_stats.columns:
    domain_stats["fake"] = 0

if "real" not in domain_stats.columns:
    domain_stats["real"] = 0

domain_stats["total"] = (
    domain_stats["fake"] +
    domain_stats["real"]
)

domain_stats["real_ratio"] = (
    domain_stats["real"] /
    domain_stats["total"]
)

def source_evidence(row):
    if row["total"] < 3:
        return "insufficient_data"
    elif row["real"] == row["total"]:
        return "real_only_in_reference"
    elif row["fake"] == row["total"]:
        return "fake_only_in_reference"
    else:
        return "mixed"

domain_stats["source_evidence"] = domain_stats.apply(
    source_evidence,
    axis=1
)

# ==========================================
# SOURCE RULE
# ==========================================

def source_rule(evidence):
    if evidence == "fake_only_in_reference":
        return 1
    elif evidence == "real_only_in_reference":
        return -1
    else:
        return 0

# ==========================================
# CAPITAL LETTER CHECK
# ==========================================

def count_caps_words(text):
    words = text.split()
    count = 0

    for word in words:
        if len(word) > 1 and word.isupper():
            count += 1

    return count

# ==========================================
# EXCESSIVE PUNCTUATION CHECK
# ==========================================

def count_excessive_punctuation(text):
    count = 0

    if "!!" in text:
        count += 1

    if "??" in text:
        count += 1

    return count

# ==========================================
# HEADLINE ANALYSIS
# ==========================================

def detect_headline_only(title):

    clean_title = title.lower()

    clean_title = remove_punctuation(clean_title)

    clean_title = re.sub(
        r"\s+",
        " ",
        clean_title
    ).strip()

    # Suspicious keywords
    suspicious_found = find_suspicious_words(
        clean_title
    )

    keyword_score = min(
        len(suspicious_found) * 5,
        20
    )

    # Capital letters
    caps_count = count_caps_words(title)

    if caps_count >= 3:
        caps_score = 10

    elif caps_count > 0:
        caps_score = 5

    else:
        caps_score = 0

    # Excessive punctuation
    punctuation_flag = count_excessive_punctuation(
        title
    )

    if punctuation_flag == 1:
        punctuation_score = 10

    else:
        punctuation_score = 0

    # Total headline score
    headline_score = (
        keyword_score +
        caps_score +
        punctuation_score
    )

    source_evidence = "Source not provided"

    return (
        headline_score,
        suspicious_found,
        caps_count,
        punctuation_flag,
        source_evidence
    )

# ==========================================
# CLAIM CHECK
# ==========================================

def calculate_claim_score(title):

    claim_words = [
        "record",
        "first",
        "highest",
        "lowest",
        "only",
        "never",
        "always",
        "100%",
        "all",
        "none"
    ]

    clean_title = title.lower()

    claims_found = []

    for word in claim_words:

        if word in clean_title:
            claims_found.append(word)

    claim_score = min(
        len(claims_found) * 5,
        20
    )

    if claims_found:
        claim_status = "Claim requires verification"

    else:
        claim_status = "No strong factual claim detected"

    return (
        claim_score,
        claim_status,
        claims_found
    )

# ==========================================
# SOURCE CHECK
# ==========================================

def calculate_source_score(domain=""):

    if domain == "" or domain is None:
        return 0, "Source not provided"

    domain = extract_domain(domain)

    if domain in domain_stats.index:

        source_evidence = domain_stats.loc[
            domain,
            "source_evidence"
        ]

        if source_evidence == "mixed":

            source_score = 10

            source_status = "Mixed source evidence"

        elif source_evidence == "fake_only_in_reference":

            source_score = 20

            source_status = "Source has fake-news evidence"

        elif source_evidence == "real_only_in_reference":

            source_score = 0

            source_status = "Source has real-news evidence"

        else:

            source_score = 5

            source_status = "Insufficient source evidence"

    else:

        source_score = 5

        source_status = "Source not found in reference data"

    return source_score, source_status

# ==========================================
# FINAL DETECTION FUNCTION
# ==========================================

def detect_fake_news(title, domain=""):

    headline_score, suspicious_found, caps_count, punctuation_flag, source_evidence = detect_headline_only(title)

    claim_score, claim_status, claims_found = calculate_claim_score(title)

    source_score, source_status = calculate_source_score(domain)

    final_score = (
        headline_score +
        claim_score +
        source_score
    )

    final_score = round(
        (final_score / 80) * 100,
        2
    )

    if final_score >= 60:

        prediction = "Likely Fake"

    elif final_score >= 30:

        prediction = "Needs Verification"

    else:

        prediction = "Likely Genuine"

    return {
        "prediction": prediction,
        "risk_score": final_score,
        "headline_score": headline_score,
        "claim_score": claim_score,
        "source_score": source_score,
        "suspicious_words": suspicious_found,
        "claim_status": claim_status,
        "claim_indicators": claims_found,
        "source_status": source_status
    }

# ==========================================
# REASONING
# ==========================================

def generate_reasoning(result):

    reasons = []

    if result["suspicious_words"]:
        reasons.append(
            "Suspicious keywords were detected."
        )

    if result["claim_indicators"]:
        reasons.append(
            "The headline contains a claim that needs verification."
        )

    if result["source_score"] > 0:
        reasons.append(
            "There is limited or negative source evidence."
        )

    if not reasons:
        reasons.append(
            "No major suspicious signals were detected."
        )

    return reasons

# ==========================================
# DASHBOARD
# ==========================================

st.title("📰 Fake News Detection System")

st.write(
    "A rule-based fake news detection system using "
    "headline analysis, claim checking, and source checking."
)

st.divider()

# ==========================================
# USER INPUT
# ==========================================

st.subheader("Enter News Details")

headline = st.text_area(
    "News Headline",
    placeholder="Enter the news headline here...",
    height=120
)

domain = st.text_input(
    "News Source / URL",
    placeholder="Example: bbc.com or https://www.bbc.com"
)

check_button = st.button(
    "🔍 Check News",
    use_container_width=True
)

# ==========================================
# DETECTION
# ==========================================

if check_button:

    if headline.strip() == "":

        st.warning(
            "Please enter a news headline."
        )

    else:

        result = detect_fake_news(
            headline,
            domain
        )

        st.divider()

        st.subheader("Detection Result")

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Risk Score",
                f'{result["risk_score"]} / 100'
            )

        with col2:

            st.metric(
                "Prediction",
                result["prediction"]
            )

        st.divider()

        # ==================================
        # HEADLINE ANALYSIS
        # ==================================

        st.subheader("🔤 Headline Analysis")

        col1, col2 = st.columns(2)

        with col1:

            st.write(
                f'**Headline Score:** '
                f'{result["headline_score"]} / 40'
            )

        with col2:

            if result["suspicious_words"]:

                st.write(
                    "**Suspicious Words:** "
                    + ", ".join(
                        result["suspicious_words"]
                    )
                )

            else:

                st.write(
                    "**Suspicious Words:** None"
                )

        # ==================================
        # CLAIM CHECK
        # ==================================

        st.subheader("📋 Claim Check")

        col1, col2 = st.columns(2)

        with col1:

            st.write(
                f'**Claim Score:** '
                f'{result["claim_score"]} / 20'
            )

        with col2:

            st.write(
                f'**Claim Status:** '
                f'{result["claim_status"]}'
            )

        if result["claim_indicators"]:

            st.write(
                "**Claim Indicators:** "
                + ", ".join(
                    result["claim_indicators"]
                )
            )

        else:

            st.write(
                "**Claim Indicators:** None"
            )

        # ==================================
        # SOURCE CHECK
        # ==================================

        st.subheader("🌐 Source Check")

        col1, col2 = st.columns(2)

        with col1:

            st.write(
                f'**Source Score:** '
                f'{result["source_score"]} / 20'
            )

        with col2:

            st.write(
                f'**Source Status:** '
                f'{result["source_status"]}'
            )

        # ==================================
        # REASONING
        # ==================================

        st.subheader("💡 System Reasoning")

        reasons = generate_reasoning(result)

        for reason in reasons:

            st.write("• " + reason)

        st.divider()

        st.caption(
            "This system provides a rule-based assessment "
            "and should not be treated as definitive proof "
            "that a news story is fake or genuine."
        )