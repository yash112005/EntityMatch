from pathlib import Path

import lightgbm as lgb
import streamlit as st
from rapidfuzz import fuzz


# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = Path(__file__).parent / "entity_match_lightgbm.txt"

# Match probability threshold
THRESHOLD = 0.7


st.set_page_config(
    page_title="Business Entity Resolution",
    page_icon="🔍",
    layout="wide"
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    return lgb.Booster(
        model_file=str(MODEL_PATH)
    )


model = load_model()


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def norm(s) -> str:
    """
    Normalize text:
    - Convert to string
    - Lowercase
    - Remove extra spaces
    """

    return " ".join(
        str(s or "").lower().split()
    )


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def build_features(
    name1,
    addr1,
    country1,
    name2,
    addr2,
    country2
):
    """
    Build EXACTLY 11 features expected by the LightGBM model.

    Feature order MUST remain exactly:

    0  name_ratio
    1  name_partial_ratio
    2  name_token_ratio
    3  name_length_diff
    4  address_ratio
    5  address_partial_ratio
    6  address_token_ratio
    7  address_length_diff
    8  country_match
    9  name_exact
    10 address_exact
    """

    features = [
        # ----------------------------------------------------
        # NAME FEATURES
        # ----------------------------------------------------

        # 0: name_ratio
        fuzz.ratio(
            name1,
            name2
        ),

        # 1: name_partial_ratio
        fuzz.partial_ratio(
            name1,
            name2
        ),

        # 2: name_token_ratio
        fuzz.token_sort_ratio(
            name1,
            name2
        ),

        # 3: name_length_diff
        abs(
            len(name1) -
            len(name2)
        ),

        # ----------------------------------------------------
        # ADDRESS FEATURES
        # ----------------------------------------------------

        # 4: address_ratio
        fuzz.ratio(
            addr1,
            addr2
        ),

        # 5: address_partial_ratio
        fuzz.partial_ratio(
            addr1,
            addr2
        ),

        # 6: address_token_ratio
        fuzz.token_sort_ratio(
            addr1,
            addr2
        ),

        # 7: address_length_diff
        abs(
            len(addr1) -
            len(addr2)
        ),

        # ----------------------------------------------------
        # COUNTRY
        # ----------------------------------------------------

        # 8: country_match
        int(
            country1 == country2
        ),

        # ----------------------------------------------------
        # EXACT MATCH
        # ----------------------------------------------------

        # 9: name_exact
        int(
            name1 == name2
        ),

        # 10: address_exact
        int(
            addr1 == addr2
        ),
    ]

    return [features]


# ============================================================
# HEADER
# ============================================================

st.title("🔍 Business Entity Resolution")

st.caption(
    "Check if two business records refer to the same "
    "real-world business — built for Amazon ML Challenge 2026."
)


# ============================================================
# MODEL DEBUG INFORMATION
# ============================================================

with st.expander("Model info (debug)"):

    st.write(
        "Features expected by model:",
        model.num_feature()
    )

    st.write(
        "Feature names:"
    )

    st.write(
        model.feature_name()
    )


# ============================================================
# CHECK MODEL FEATURES
# ============================================================

EXPECTED_FEATURES = [
    "name_ratio",
    "name_partial_ratio",
    "name_token_ratio",
    "name_length_diff",
    "address_ratio",
    "address_partial_ratio",
    "address_token_ratio",
    "address_length_diff",
    "country_match",
    "name_exact",
    "address_exact",
]


# ============================================================
# INPUT SECTION
# ============================================================

st.subheader("Business Record Comparison")

col1, col2 = st.columns(2)


# ------------------------------------------------------------
# RECORD 1
# ------------------------------------------------------------

with col1:

    st.markdown("### 📄 Business Record 1")

    name1 = st.text_input(
        "Business Name 1",
        "Sharma Sweets Pvt Ltd"
    )

    addr1 = st.text_input(
        "Address 1",
        "123 Main St, Mumbai"
    )

    country1 = st.text_input(
        "Country 1",
        "India"
    )


# ------------------------------------------------------------
# RECORD 2
# ------------------------------------------------------------

with col2:

    st.markdown("### 📄 Business Record 2")

    name2 = st.text_input(
        "Business Name 2",
        "Sharma Sweets Private Limited"
    )

    addr2 = st.text_input(
        "Address 2",
        "123 Main Street, Mumbai"
    )

    country2 = st.text_input(
        "Country 2",
        "India"
    )


# ============================================================
# MATCH BUTTON
# ============================================================

if st.button(
    "🔍 Check Match",
    type="primary",
    use_container_width=True
):

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    if not name1.strip() or not name2.strip():

        st.warning(
            "Please enter both business names."
        )

        st.stop()


    # --------------------------------------------------------
    # NORMALIZE INPUT
    # --------------------------------------------------------

    n1 = norm(name1)
    a1 = norm(addr1)
    c1 = norm(country1)

    n2 = norm(name2)
    a2 = norm(addr2)
    c2 = norm(country2)


    # --------------------------------------------------------
    # BUILD FEATURES
    # --------------------------------------------------------

    features = build_features(
        n1,
        a1,
        c1,
        n2,
        a2,
        c2
    )


    # --------------------------------------------------------
    # FEATURE COUNT CHECK
    # --------------------------------------------------------

    if len(features[0]) != model.num_feature():

        st.error(
            f"Feature mismatch: app gives "
            f"{len(features[0])} features, "
            f"but model expects "
            f"{model.num_feature()} features."
        )

        st.write(
            "Model features:",
            model.feature_name()
        )

        st.stop()


    # --------------------------------------------------------
    # FEATURE NAME CHECK
    # --------------------------------------------------------

    model_features = model.feature_name()

    if model_features != EXPECTED_FEATURES:

        st.warning(
            "⚠️ Model feature names/order differ "
            "from the expected feature list."
        )

        st.write(
            "Expected:",
            EXPECTED_FEATURES
        )

        st.write(
            "Model:",
            model_features
        )


    # --------------------------------------------------------
    # MODEL PREDICTION
    # --------------------------------------------------------

    try:

        prediction = model.predict(
            features
        )

        prob = float(
            prediction[0]
        )

    except Exception as e:

        st.error(
            f"Model prediction failed: {e}"
        )

        st.stop()


    # --------------------------------------------------------
    # KEEP PROBABILITY BETWEEN 0 AND 1
    # --------------------------------------------------------

    prob = max(
        0.0,
        min(
            1.0,
            prob
        )
    )


    # ========================================================
    # RESULT
    # ========================================================

    st.divider()

    if prob >= THRESHOLD:

        st.success(
            f"✅ Likely MATCH — "
            f"Match probability: {prob:.1%}"
        )

    else:

        st.error(
            f"❌ Likely NOT a match — "
            f"Match probability: {prob:.1%}"
        )


    # --------------------------------------------------------
    # PROGRESS BAR
    # --------------------------------------------------------

    st.progress(
        prob
    )


    # ========================================================
    # FEATURE VALUES
    # ========================================================

    with st.expander("🔬 View calculated features"):

        feature_values = features[0]

        for name, value in zip(
            EXPECTED_FEATURES,
            feature_values
        ):

            st.write(
                f"**{name}:** {value}"
            )
