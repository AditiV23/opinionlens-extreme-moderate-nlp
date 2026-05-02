"""
Utility functions for the Extreme vs Moderate Opinion Identifier project.

Keep this file with app.py and train_model.py.
The trained scikit-learn pipeline depends on clean_text() from this module.
"""

import re
import numpy as np
import pandas as pd

STAR_LABELS = {
    0: "Extreme Negative",
    1: "Moderate Negative",
    2: "Neutral / Mixed",
    3: "Moderate Positive",
    4: "Extreme Positive",
}

BUSINESS_ACTIONS = {
    "Extreme Negative": "High priority: investigate complaint, service recovery, escalation",
    "Moderate Negative": "Monitor: identify issue pattern and improve customer support",
    "Neutral / Mixed": "Observe: collect more feedback and clarify expectations",
    "Moderate Positive": "Nurture: reinforce strengths and request review/testimonial",
    "Extreme Positive": "Advocate: use for testimonials, referrals, loyalty campaigns",
}


def clean_text(text: str) -> str:
    """Lightweight preprocessing suitable for customer reviews/opinions."""
    if text is None:
        return ""
    text = str(text)
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " URL ", text)
    text = re.sub(r"@\w+", " USER ", text)
    text = re.sub(r"#", " ", text)
    text = re.sub(r"[^a-z0-9\s!?']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def star_to_bucket(star_label: int) -> str:
    """Map Yelp labels 0-4 to broader project category."""
    return "Extreme" if int(star_label) in (0, 4) else "Moderate"


def star_to_detail(star_label: int) -> str:
    return STAR_LABELS.get(int(star_label), "Unknown")


def business_action(detail_label: str) -> str:
    return BUSINESS_ACTIONS.get(detail_label, "Review manually")


def prediction_frame(pipeline, texts):
    """
    Convert raw texts into project-ready predictions.
    Expected model: scikit-learn Pipeline ending with classifier supporting predict_proba.
    """
    texts = ["" if t is None else str(t) for t in texts]
    proba = pipeline.predict_proba(texts)
    classes = [int(c) for c in pipeline.classes_]
    class_to_col = {c: i for i, c in enumerate(classes)}

    # Defensive handling in case a class is absent in a small demo model
    def p_for(c):
        if c in class_to_col:
            return proba[:, class_to_col[c]]
        return np.zeros(len(texts))

    p_star_1 = p_for(0)
    p_star_2 = p_for(1)
    p_star_3 = p_for(2)
    p_star_4 = p_for(3)
    p_star_5 = p_for(4)

    p_extreme = p_star_1 + p_star_5
    p_moderate = p_star_2 + p_star_3 + p_star_4

    broad = np.where(p_extreme >= p_moderate, "Extreme", "Moderate")
    broad_conf = np.maximum(p_extreme, p_moderate)

    detail = []
    predicted_star = []
    for i in range(len(texts)):
        if broad[i] == "Extreme":
            if p_star_1[i] >= p_star_5[i]:
                label = 0
            else:
                label = 4
        else:
            moderate_probs = {1: p_star_2[i], 2: p_star_3[i], 3: p_star_4[i]}
            label = max(moderate_probs, key=moderate_probs.get)
        predicted_star.append(label + 1)
        detail.append(star_to_detail(label))

    df = pd.DataFrame({
        "text": texts,
        "category": broad,
        "intensity_type": detail,
        "confidence": np.round(broad_conf, 4),
        "extreme_probability": np.round(p_extreme, 4),
        "moderate_probability": np.round(p_moderate, 4),
        "predicted_star_level": predicted_star,
        "recommended_business_action": [business_action(d) for d in detail],
    })

    df["priority"] = np.select(
        [
            (df["intensity_type"] == "Extreme Negative") & (df["confidence"] >= 0.65),
            (df["category"] == "Extreme") | (df["confidence"] >= 0.80),
            (df["category"] == "Moderate"),
        ],
        ["Critical", "High", "Normal"],
        default="Normal",
    )
    return df
