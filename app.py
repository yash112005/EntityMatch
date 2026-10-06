from pathlib import Path

import lightgbm as lgb
import streamlit as st
from rapidfuzz import fuzz


# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = Path(__file__).parent / "entity_match_lightgbm.txt"

# Model probability threshold
THRESHOLD = 0.70


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Business Match Checker",
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
# NORMALIZE TEXT
# ============================================================

def norm(value):

    return " ".join(
        str(value or "").lower().split()
    )


# ============================================================
# BUILD MODEL FEATURES
# ============================================================

def build_features(
    name1,
    addr1,
    country1,
    name2,
    addr2,
    country2
):

    return [[

        # 0 - Name similarity
        fuzz.ratio(
            name1,
            name2
        ),

        # 1 - Name partial similarity
        fuzz.partial_ratio(
            name1,
            name2
        ),

        # 2 - Name token similarity
        fuzz.token_sort_ratio(
            name1,
            name2
        ),

        # 3 - Name length difference
        abs(
            len(name1) -
            len(name2)
        ),

        # 4 - Address similarity
        fuzz.ratio(
            addr1,
            addr2
        ),

        # 5 - Address partial similarity
        fuzz.partial_ratio(
            addr1,
            addr2
        ),

        # 6 - Address token similarity
        fuzz.token_sort_ratio(
            addr1,
            addr2
        ),

        # 7 - Address length difference
        abs(
            len(addr1) -
            len(addr2)
        ),

        # 8 - Country match
        int(
            country1 == country2
        ),

        # 9 - Exact name match
        int(
            name1 == name2
        ),

        # 10 - Exact address match
        int(
            addr1 == addr2
        )
    ]]


# ============================================================
# FRIENDLY RESULT
# ============================================================

def get_result(probability):

    if probability >= 0.85:

        return (
            "strong",
            "Strong Match",
            "These two records are very likely to represent the same business."
        )

    elif probability >= THRESHOLD:

        return (
            "possible",
            "Possible Match",
            "These two records appear to be the same business, but you may want to verify the details."
        )

    else:

        return (
            "different",
            "Likely Different Businesses",
            "The business details appear different, so these records are unlikely to represent the same business."
        )


# ============================================================
# HEADER
# ============================================================

st.title("🔍 Business Match Checker")

st.write(
    "Compare two business listings and check whether "
    "they are likely to belong to the same real-world business."
)

st.info(
    "💡 Enter the business name, address and country for both businesses."
)


# ============================================================
# BUSINESS INPUTS
# ============================================================

col1, col2 = st.columns(2)


# ------------------------------------------------------------
# BUSINESS A
# ------------------------------------------------------------

with col1:

    st.subheader("🏢 Business A")

    name1 = st.text_input(
        "Business Name",
        placeholder="e.g. Sharma Sweets Pvt Ltd",
        key="name1"
    )

    addr1 = st.text_area(
        "Business Address",
        placeholder="e.g. 123 Main Street, Mumbai",
        key="addr1"
    )

    country1 = st.text_input(
        "Country",
        placeholder="e.g. India",
        key="country1"
    )


# ------------------------------------------------------------
# BUSINESS B
# ------------------------------------------------------------

with col2:

    st.subheader("🏢 Business B")

    name2 = st.text_input(
        "Business Name",
        placeholder="e.g. Sharma Sweets Private Limited",
        key="name2"
    )

    addr2 = st.text_area(
        "Business Address",
        placeholder="e.g. 123 Main St, Mumbai",
        key="addr2"
    )

    country2 = st.text_input(
        "Country",
        placeholder="e.g. India",
        key="country2"
    )


# ============================================================
# EXAMPLE BUTTONS
# ============================================================

st.markdown("---")

st.subheader("✨ Need an example?")

example_col1, example_col2 = st.columns(2)

with example_col1:

    if st.button(
        "✅ Try a Match Example",
        use_container_width=True
    ):

        st.session_state.name1 = "Sharma Sweets Pvt Ltd"
        st.session_state.addr1 = "123 Main Street, Mumbai"
        st.session_state.country1 = "India"

        st.session_state.name2 = "Sharma Sweets Private Limited"
        st.session_state.addr2 = "123 Main St, Mumbai"
        st.session_state.country2 = "India"

        st.rerun()


with example_col2:

    if st.button(
        "❌ Try a Different Business Example",
        use_container_width=True
    ):

        st.session_state.name1 = "Sharma Sweets Pvt Ltd"
        st.session_state.addr1 = "123 Main Street, Mumbai"
        st.session_state.country1 = "India"

        st.session_state.name2 = "Rajesh Electronics Store"
        st.session_state.addr2 = "45 MG Road, Delhi"
        st.session_state.country2 = "India"

        st.rerun()


# ============================================================
# CHECK BUTTON
# ============================================================

st.markdown("")

if st.button(
    "🔍 Check Businesses",
    type="primary",
    use_container_width=True
):

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    if not name1.strip():

        st.warning(
            "Please enter the name of Business A."
        )

        st.stop()

    if not name2.strip():

        st.warning(
            "Please enter the name of Business B."
        )

        st.stop()


    # --------------------------------------------------------
    # NORMALIZE
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
    # SAFETY CHECK
    # --------------------------------------------------------

    if len(features[0]) != model.num_feature():

        st.error(
            "Something went wrong while preparing the "
            "business information for the model."
        )

        st.stop()


    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    try:

        prediction = model.predict(
            features
        )

        probability = float(
            prediction[0]
        )

    except Exception as error:

        st.error(
            "Unable to check the businesses right now."
        )

        st.stop()


    # --------------------------------------------------------
    # KEEP VALUE BETWEEN 0 AND 1
    # --------------------------------------------------------

    probability = max(
        0.0,
        min(
            1.0,
            probability
        )
    )


    # ========================================================
    # RESULT
    # ========================================================

    result_type, result_title, result_description = (
        get_result(probability)
    )


    st.markdown("---")

    st.subheader("📊 Result")


    # --------------------------------------------------------
    # STRONG MATCH
    # --------------------------------------------------------

    if result_type == "strong":

        st.success(
            f"🟢 {result_title}"
        )


    # --------------------------------------------------------
    # POSSIBLE MATCH
    # --------------------------------------------------------

    elif result_type == "possible":

        st.warning(
            f"🟡 {result_title}"
        )


    # --------------------------------------------------------
    # DIFFERENT
    # --------------------------------------------------------

    else:

        st.error(
            f"🔴 {result_title}"
        )


    # --------------------------------------------------------
    # DESCRIPTION
    # --------------------------------------------------------

    st.write(
        result_description
    )


    # --------------------------------------------------------
    # CONFIDENCE
    # --------------------------------------------------------

    st.metric(
        "Match Confidence",
        f"{probability:.1%}"
    )

    st.progress(
        probability
    )


    # ========================================================
    # SIMPLE EXPLANATION
    # ========================================================

    st.subheader("💡 Why this result?")


    name_similarity = fuzz.token_sort_ratio(
        n1,
        n2
    )

    address_similarity = fuzz.token_sort_ratio(
        a1,
        a2
    )


    explanation = []


    if name_similarity >= 85:

        explanation.append(
            "✅ The business names are very similar."
        )

    elif name_similarity >= 60:

        explanation.append(
            "🟡 The business names have some similarity."
        )

    else:

        explanation.append(
            "❌ The business names are quite different."
        )


    if address_similarity >= 85:

        explanation.append(
            "✅ The addresses are very similar."
        )

    elif address_similarity >= 60:

        explanation.append(
            "🟡 The addresses have some similarity."
        )

    else:

        explanation.append(
            "❌ The addresses are quite different."
        )


    if c1 and c2:

        if c1 == c2:

            explanation.append(
                "✅ Both businesses are listed in the same country."
            )

        else:

            explanation.append(
                "❌ The businesses are listed in different countries."
            )


    for item in explanation:

        st.write(item)


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "AI-powered Business Entity Resolution • "
    "Use the result as a decision-support signal, not as a guarantee."
)
