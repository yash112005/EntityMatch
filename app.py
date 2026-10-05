import streamlit as st
import lightgbm as lgb
from rapidfuzz import fuzz
import re

# -----------------------------
# Load trained LightGBM model
# -----------------------------
@st.cache_resource
def load_model():
    return lgb.Booster(model_file="entity_match_lightgbm.txt")


model = load_model()

THRESHOLD = 0.82


# -----------------------------
# Text cleaning
# -----------------------------
def clean_text(text):
    if not text:
        return ""

    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


# -----------------------------
# Feature generation
# -----------------------------
def create_features(name1, addr1, country1, name2, addr2, country2):

    name1 = clean_text(name1)
    name2 = clean_text(name2)

    addr1 = clean_text(addr1)
    addr2 = clean_text(addr2)

    name_ratio = fuzz.ratio(name1, name2) / 100
    name_partial_ratio = fuzz.partial_ratio(name1, name2) / 100
    name_token_ratio = fuzz.token_sort_ratio(name1, name2) / 100

    name_length_diff = abs(len(name1) - len(name2))

    address_ratio = fuzz.ratio(addr1, addr2) / 100
    address_partial_ratio = fuzz.partial_ratio(addr1, addr2) / 100
    address_token_ratio = fuzz.token_sort_ratio(addr1, addr2) / 100

    address_length_diff = abs(len(addr1) - len(addr2))

    country_match = int(
        clean_text(country1) == clean_text(country2)
        and clean_text(country1) != ""
    )

    name_exact = int(name1 == name2 and name1 != "")
    address_exact = int(addr1 == addr2 and addr1 != "")

    return [[
        name_ratio,
        name_partial_ratio,
        name_token_ratio,
        name_length_diff,
        address_ratio,
        address_partial_ratio,
        address_token_ratio,
        address_length_diff,
        country_match,
        name_exact,
        address_exact
    ]]


# -----------------------------
# Page
# -----------------------------
st.set_page_config(
    page_title="Business Entity Resolution AI",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Business Entity Resolution AI")

st.write(
    "Check whether two business records refer to the same real-world business entity."
)

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Business Record 1")

    name1 = st.text_input(
        "Business Name",
        "Sharma Sweets Pvt Ltd"
    )

    addr1 = st.text_area(
        "Address",
        "123 Main Street, Mumbai"
    )

    country1 = st.text_input(
        "Country",
        "India"
    )


with col2:
    st.subheader("Business Record 2")

    name2 = st.text_input(
        "Business Name",
        "Sharma Sweets Private Limited"
    )

    addr2 = st.text_area(
        "Address",
        "123 Main Street, Mumbai"
    )

    country2 = st.text_input(
        "Country",
        "India"
    )


st.divider()

if st.button("🔎 Check Entity Match", type="primary"):

    features = create_features(
        name1,
        addr1,
        country1,
        name2,
        addr2,
        country2
    )

    probability = float(model.predict(features)[0])

    st.subheader("Result")

    if probability >= THRESHOLD:

        st.success(
            f"✅ LIKELY MATCH\n\n"
            f"Confidence: **{probability:.2%}**"
        )

    else:

        st.error(
            f"❌ LIKELY NOT A MATCH\n\n"
            f"Confidence: **{probability:.2%}**"
        )

    with st.expander("View Model Features"):

        feature_names = [
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
            "address_exact"
        ]

        for name, value in zip(feature_names, features[0]):
            st.write(f"**{name}:** {value}")
