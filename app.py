import pickle
from pathlib import Path

import streamlit as st
from rapidfuzz import fuzz

# ---------------------------------------------------------------
# Config
# ---------------------------------------------------------------
MODEL_PATH = Path(__file__).parent / "entity_match_model.pkl"
THRESHOLD = 0.7

# Training me jo 10th feature tha uski value yahan daalo.
LAST_FEATURE_DEFAULT = 1

st.set_page_config(page_title="Business Entity Resolution", page_icon="🔍")


# ---------------------------------------------------------------
# Model load (cached)
# ---------------------------------------------------------------
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


model = load_model()


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------
def norm(s) -> str:
    """Training wali normalization hi use karo."""
    return " ".join(str(s or "").lower().split())


def build_features(name1, addr1, name2, addr2):
    return [[
        fuzz.ratio(name1, name2),
        fuzz.token_sort_ratio(name1, name2),
        fuzz.partial_ratio(name1, name2),
        fuzz.token_set_ratio(name1, name2),
        fuzz.ratio(addr1, addr2),
        fuzz.token_sort_ratio(addr1, addr2),
        fuzz.partial_ratio(addr1, addr2),
        abs(len(name1) - len(name2)),
        abs(len(addr1) - len(addr2)),
        LAST_FEATURE_DEFAULT,
    ]]


def predict_prob(features) -> float:
    if hasattr(model, "predict_proba"):
        return float(model.predict_proba(features)[0][1])
    return float(model.predict(features)[0])  # raw lgb.Booster


# ---------------------------------------------------------------
# UI
# ---------------------------------------------------------------
st.title("🔍 Business Entity Resolution")
st.caption(
    "Check if two business records refer to the same real-world business — "
    "built for Amazon ML Challenge 2026."
)

col1, col2 = st.columns(2)
with col1:
    name1 = st.text_input("Business Name 1", "Sharma Sweets Pvt Ltd")
    addr1 = st.text_input("Address 1", "123 Main St, Mumbai")
with col2:
    name2 = st.text_input("Business Name 2", "Sharma Sweets Private Limited")
    addr2 = st.text_input("Address 2", "123 Main Street, Mumbai")

if st.button("Check match", type="primary"):
    if not name1.strip() or not name2.strip():
        st.warning("Please enter both business names.")
    else:
        try:
            n1, a1, n2, a2 = map(norm, (name1, addr1, name2, addr2))
            prob = predict_prob(build_features(n1, a1, n2, a2))
        except Exception as e:
            st.error(f"Prediction failed: {type(e).__name__}: {e}")
        else:
            if prob >= THRESHOLD:
                st.success(f"✅ Likely MATCH — Match probability: {prob:.1%}")
            else:
                st.error(f"❌ Likely NOT a match — Match probability: {prob:.1%}")
            st.progress(min(max(prob, 0.0), 1.0))
