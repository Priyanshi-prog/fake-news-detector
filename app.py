import streamlit as st
from transformers import pipeline
import torch

# ============================================
# PAGE CONFIG
# ============================================
st.set_page_config(
    page_title="🚨 Fake News Detector",
    page_icon="🔍",
    layout="wide"
)

st.title("🚨 Fake News Detection AI")
st.markdown("---")
st.write("Paste news text to check if it's authentic or misinformation using AI.")

# ============================================
# LOAD MODEL (ONCE - CACHED)
# ============================================
@st.cache_resource
def load_fake_news_model():
    """Load fake news model locally - downloads once, runs offline forever."""
    model_name = "vikram71198/distilroberta-base-finetuned-fake-news-detection"
    
    classifier = pipeline(
        "text-classification",
        model=model_name,
        tokenizer=model_name,
        device=-1  # CPU (use 0 for GPU if available)

    )
    return classifier

# ============================================
# HELPER FUNCTIONS
# ============================================
def call_hf_model(text: str) -> dict:
    try:
        model = load_fake_news_model()

        # Important: request all scores at call time (consistent parsing)
        result = model(text[:512], top_k=None)

        # result can be:
        # A) [[{...}, {...}]]  (nested)
        # B) [{...}, {...}]    (flat)
        preds = []
        if isinstance(result, list) and len(result) > 0:
            if isinstance(result[0], list):
                preds = result[0]
            elif isinstance(result[0], dict):
                preds = result

        real = 0.0
        fake = 0.0
        for p in preds:
            lab = str(p.get("label", "")).upper()
            sc = float(p.get("score", 0.0))

            if any(x in lab for x in ["LABEL_1", "REAL", "TRUE"]):
                real = max(real, sc)
            elif any(x in lab for x in ["LABEL_0", "FAKE", "FALSE"]):
                fake = max(fake, sc)

        return {
            "ok": True,
            "status": 200,
            "real": real,
            "fake": fake,
            "raw": result,
            "msg": "✅ Analysis complete",
        }

    except Exception as e:
        return {
            "ok": False,
            "status": 500,
            "msg": f"❌ Model error: {str(e)}",
            "raw": None,
            "real": 0.0,
            "fake": 0.0,
        }



def get_verdict(real: float, fake: float) -> tuple:
    """
    Determine verdict based on scores.
    Returns: (verdict_text, css_class)
    """
    if real >= 0.65:
        return (
            "✅ <strong>LIKELY AUTHENTIC</strong><br>This news appears to be genuine based on AI analysis.",
            "authentic"
        )
    elif fake >= 0.65:
        return (
            "🚨 <strong>LIKELY FAKE</strong><br>This news shows characteristics of misinformation.",
            "fake"
        )
    else:
        return (
            "⚠️ <strong>UNCERTAIN</strong><br>Analysis inconclusive. Verify with multiple sources.",
            "uncertain"
        )


# ============================================
# CUSTOM CSS
# ============================================
st.markdown("""
<style>
    .authentic {
        background-color: #d4edda;
        color: #155724;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #28a745;
        font-size: 18px;
        margin: 20px 0;
    }
    .fake {
        background-color: #f8d7da;
        color: #721c24;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #dc3545;
        font-size: 18px;
        margin: 20px 0;
    }
    .uncertain {
        background-color: #fff3cd;
        color: #856404;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #ffc107;
        font-size: 18px;
        margin: 20px 0;
    }
    .score-bar {
        width: 100%;
        height: 40px;
        background-color: #f0f0f0;
        border-radius: 10px;
        overflow: hidden;
        margin: 10px 0;
        display: flex;
    }
    .score-real {
        background-color: #28a745;
        height: 100%;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: bold;
    }
    .score-fake {
        background-color: #dc3545;
        height: 100%;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# MAIN UI
# ============================================
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📝 Paste News Text")
    news_text = st.text_area(
        "Enter news text (or headline):",
        placeholder="e.g., 'Scientists discover new cure for...'",
        height=150
    )

with col2:
    st.subheader("📊 Instructions")
    st.info("""
    1️⃣ Paste news text
    2️⃣ Click **Analyze**
    3️⃣ View results
    
    ⚠️ **Note**: Always verify with multiple sources!
    """)

st.markdown("---")

# ============================================
# ANALYZE BUTTON
# ============================================
if st.button("🔍 Analyze News", use_container_width=True, type="primary"):
    if not news_text.strip():
        st.error("❌ Please enter news text to analyze.")
    else:
        with st.spinner("🤖 AI is analyzing... (first run may take 1-2 min)"):
            result = call_hf_model(news_text)
        
        # Display debug info
        with st.expander("🔧 Debug Info"):
            st.write("Raw API Response:", result['raw'])
        
        if result['ok']:
            real_score = result['real']
            fake_score = result['fake']
            
            # Display verdict
            verdict_text, verdict_class = get_verdict(real_score, fake_score)
            st.markdown(f'<div class="{verdict_class}">{verdict_text}</div>', unsafe_allow_html=True)
            
            # Display scores
            st.subheader("📈 Confidence Scores")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("✅ Authentic Score", f"{real_score*100:.1f}%")
            with col2:
                st.metric("🚨 Fake Score", f"{fake_score*100:.1f}%")
            
            # Score visualization
            st.markdown("**Score Breakdown:**")
            st.markdown(f"""
            <div class="score-bar">
                <div class="score-real" style="width: {real_score*100}%">{real_score*100:.0f}% Real</div>
                <div class="score-fake" style="width: {fake_score*100}%">{fake_score*100:.0f}% Fake</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Recommendations
            st.markdown("---")
            st.subheader("💡 Recommendations")
            st.warning("""
            - ✅ Cross-check with reliable news sources
            - ✅ Check author credentials
            - ✅ Look for fact-checking sites (Snopes, AFP Fact Check, etc.)
            - ✅ Analyze source bias and agenda
            - ✅ Check publication date and context
            """)
        else:
            st.error(f"❌ {result['msg']}")

# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888; font-size: 12px;">
    Powered by vikram71198/distilroberta-base-finetuned-fake-news-detection | 
    Built with Streamlit | 
    AI Detection: ~99% Accuracy
</div>
""", unsafe_allow_html=True)
