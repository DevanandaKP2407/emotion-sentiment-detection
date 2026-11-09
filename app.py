import os
import streamlit as st
from model_utils import load_models, predict_text

st.set_page_config(page_title="Beyond Sentiment", page_icon="🧠", layout="centered")

st.markdown("""
<style>
.main {max-width: 900px; padding-top: 2rem;}
.hero {padding: 1.5rem; border-radius: 18px; background: linear-gradient(120deg,#18233d,#38265f); color: white; margin-bottom: 1.2rem;}
.hero h1 {margin: 0; font-size: 2.2rem;}
.hero p {margin: .5rem 0 0; color: #d9d8ee;}
.metric-card {padding: 1rem; border: 1px solid rgba(128,128,128,.3); border-radius: 14px;}
.small-note {font-size: .85rem; opacity: .8;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1>🧠 Beyond Sentiment</h1>
  <p>Understand the emotion behind the words — not just whether they are positive or negative.</p>
</div>
""", unsafe_allow_html=True)

st.write("Enter a sentence below to predict its sentiment and likely emotion.")

model_dir = os.path.join(os.path.dirname(__file__), "models")
emotion_model, sentiment_model, error = load_models(model_dir)

if error:
    st.warning("The trained models are not available yet. Follow the setup instructions in README.md, then run `python train_models.py`.")
    st.code("pip install -r requirements.txt\npython train_models.py\nstreamlit run app.py")
    st.stop()

user_text = st.text_area(
    "Your text",
    placeholder="Example: I worked so hard for this, and I'm finally happy with the result!",
    height=130,
    max_chars=1000
)

if st.button("Analyze text", type="primary", use_container_width=True):
    if not user_text.strip():
        st.info("Please enter a sentence first.")
    else:
        result = predict_text(user_text, emotion_model, sentiment_model)
        st.divider()
        st.subheader("Analysis result")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Sentiment")
            st.metric(result["sentiment"].title(), f'{result["sentiment_confidence"]:.1%} confidence')
        with c2:
            st.markdown("#### Predicted emotion")
            st.metric(result["emotion"].title(), f'{result["emotion_confidence"]:.1%} confidence')

        st.markdown("#### Emotion confidence")
        st.progress(float(result["emotion_confidence"]))
        st.caption(f'The model is {result["emotion_confidence"]:.1%} confident in the emotion prediction.')

        with st.expander("See all emotion scores"):
            for label, score in sorted(result["emotion_scores"].items(), key=lambda x: x[1], reverse=True):
                st.write(f"**{label.title()}** — {score:.1%}")
                st.progress(float(score))

        st.info("These are model predictions, not certain interpretations of a person's true feelings. Context, sarcasm and mixed emotions can change the meaning.")

st.divider()
with st.expander("About this project"):
    st.write("""
    This project uses TF-IDF to convert text into numerical features and Logistic Regression
    to classify emotion and sentiment. It is trained using the public `dair-ai/emotion`
    dataset from Hugging Face.
    """)
    st.caption("Sentiment labels are derived from the dataset's emotion labels. See README for the mapping and limitations.")
