from pathlib import Path

import lightgbm as lgb
import streamlit as st
from rapidfuzz import fuzz

# ---------------------------------------------------------------
# Config
# ---------------------------------------------------------------
# Repo me jo txt model hai (app.py ke saath same folder)
MODEL_PATH = Path(__file__).parent / "entity_match_lightgbm.txt"
THRESHOLD = 0.7

st.set_page_config(page_title="Business Entity Resolution", page_icon="🔍")


# ---------------------------------------------------------------
# Model load (cached)
# ---------------------------------------------------------------
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    return lgb.Booster(model_file=str(MODEL_PATH))


model = load_model()


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------
def norm(s) -> str:
    """Training wali normalization hi use karo."""
    return " ".join(str(s or "").lower().split())


def build_features(name1, addr1, name2, addr2):
    # NOTE: ye list training ke features se EXACT match honi chahiye
    # (same order, same count). Model me kitne features hain neeche
    # app me dikhaya jata hai.
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
    ]]


# ---------------------------------------------------------------
# UI
# ---------------------------------------------------------------
st.title("🔍 Business Entity Resolution")
st.caption(
    "Check if two business records refer to the same real-world business — "
    "built for Amazon ML Challenge 2026."
)

with st.expander("Model info (debug)"):
    st.write("Features expected by model:", model.num_feature())
    st.write(model.feature_name())

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
        n1, a1, n2, a2 = map(norm, (name1, addr1, name2, addr2))
        features = build_features(n1, a1, n2, a2)

        if len(features[0]) != model.num_feature():
            st.error(
                f"Feature mismatch: app gives {len(features[0])} features, "
                f"model expects {model.num_feature()}. "
                "Model info (debug) me feature names dekho aur build_features() update karo."
            )
        else:
            prob = float(model.predict(features)[0])
            if prob >= THRESHOLD:
                st.success(f"✅ Likely MATCH — Match probability: {prob:.1%}")
            else:
                st.error(f"❌ Likely NOT a match — Match probability: {prob:.1%}")
            st.progress(min(max(prob, 0.0), 1.0))
            else:
                st.error(f"❌ Likely NOT a match — Match probability: {prob:.1%}")
            st.progress(min(max(prob, 0.0), 1.0))
