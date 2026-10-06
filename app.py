import pickle
from pathlib import Path

import gradio as gr
from rapidfuzz import fuzz

# ---------------------------------------------------------------
# Config
# ---------------------------------------------------------------
MODEL_PATH = Path(__file__).parent / "entity_match_model.pkl"
THRESHOLD = 0.7

# Training me jo 10th feature tha uski value yahan daalo.
# Agar wo real feature tha (e.g. same-city flag), use properly compute karo.
LAST_FEATURE_DEFAULT = 1

# ---------------------------------------------------------------
# Model load (ek hi baar, startup pe)
# ---------------------------------------------------------------
if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model file not found at {MODEL_PATH}. "
        "Check ki .pkl repo me commit hui hai aur .gitignore me nahi hai."
    )

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------
def norm(s) -> str:
    """Lowercase + extra spaces hatao.
    IMPORTANT: training me jo normalization kiya tha, bilkul wahi yahan rakho.
    Agar training me lowercase nahi kiya tha, to .lower() hata do.
    """
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
    # sklearn wrapper (LGBMClassifier etc.)
    if hasattr(model, "predict_proba"):
        return float(model.predict_proba(features)[0][1])
    # raw lgb.Booster: predict() binary me seedha probability deta hai
    return float(model.predict(features)[0])


# ---------------------------------------------------------------
# Main function
# ---------------------------------------------------------------
def check_match(name1, addr1, name2, addr2):
    if not str(name1 or "").strip() or not str(name2 or "").strip():
        return "⚠️ Please enter both business names."

    name1, addr1, name2, addr2 = map(norm, (name1, addr1, name2, addr2))

    try:
        features = build_features(name1, addr1, name2, addr2)
        prob = predict_prob(features)
    except Exception as e:
        return f"⚠️ Prediction failed: {type(e).__name__}: {e}"

    if prob >= THRESHOLD:
        return f"✅ Likely MATCH — Match probability: {prob:.1%}"
    return f"❌ Likely NOT a match — Match probability: {prob:.1%}"


# ---------------------------------------------------------------
# UI
# ---------------------------------------------------------------
demo = gr.Interface(
    fn=check_match,
    inputs=[
        gr.Textbox(label="Business Name 1", value="Sharma Sweets Pvt Ltd"),
        gr.Textbox(label="Address 1", value="123 Main St, Mumbai"),
        gr.Textbox(label="Business Name 2", value="Sharma Sweets Private Limited"),
        gr.Textbox(label="Address 2", value="123 Main Street, Mumbai"),
    ],
    outputs=gr.Textbox(label="Result"),
    title="🔍 Business Entity Resolution",
    description=(
        "Check if two business records refer to the same real-world business — "
        "built for Amazon ML Challenge 2026."
    ),
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
