import streamlit as st
import time
from pathlib import Path
import sys

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from scripts.predict import predict

st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6;
    }
    .header {
        color: #1f77b4;
    }
    .stProgress > div > div > div > div {
        background-color: #1f77b4;
    }
    .result-real {
        color: #2ecc71;
        font-weight: bold;
        font-size: 1.5em;
    }
    .result-fake {
        color: #e74c3c;
        font-weight: bold;
        font-size: 1.5em;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

def main():
    st.title("📰 Fake News Detector")
    st.markdown("Enter the title and content of a news article to verify its authenticity.")

    # Title + Text inputs
    title_input = st.text_input("News Title", placeholder="e.g. NASA confirms moon base construction")
    text_input = st.text_area(
        "News Text",
        height=250,
        placeholder="Paste full news article here...",
        help="Minimum 50 characters total for accurate analysis"
    )

    # Trigger prediction
    if st.button("🔍 Analyze News", use_container_width=True):
        if len(title_input.strip() + text_input.strip()) < 50:
            st.warning("⚠️ Please enter at least 50 characters total")
        else:
            with st.spinner("Analyzing..."):
                time.sleep(0.5)
                result = predict(title_input, text_input)

                st.subheader("Analysis Results")
                if result["label"] == "Real":
                    st.markdown('<p class="result-real">✅ This news appears to be REAL</p>', unsafe_allow_html=True)
                elif result["label"] == "Fake":
                    st.markdown('<p class="result-fake">⚠️ This news appears to be FAKE</p>', unsafe_allow_html=True)
                else:
                    st.error(result.get("error", "Something went wrong."))

                if "confidence" in result:
                    st.metric("Confidence Score", result["confidence"])

                if "probabilities" in result:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Real Probability", f"{result['probabilities']['Real']*100:.1f}%")
                    with col2:
                        st.metric("Fake Probability", f"{result['probabilities']['Fake']*100:.1f}%")

                    st.progress(result["probabilities"]["Real"] if result["label"] == "Real" else result["probabilities"]["Fake"])

                with st.expander("View Detailed Analysis"):
                    st.write("**Processed Text Preview:**")
                    st.caption(result.get("processed_text", "Not available")[:500] + "...")
                    st.write("**Probability Breakdown:**")
                    st.json(result.get("probabilities", {}))

if __name__ == "__main__":
    main()
