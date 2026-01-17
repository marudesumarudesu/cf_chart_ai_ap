import streamlit as st


def inject_css() -> None:
    """Add lightweight CSS to improve aesthetics without breaking Streamlit."""
    st.markdown(
        """
<style>
  #MainMenu {visibility: hidden;}
  footer {visibility: hidden;}
  header {visibility: hidden;}

  .stApp {
    background: linear-gradient(135deg, rgba(15, 23, 42, 1) 0%, rgba(2, 6, 23, 1) 100%);
  }

  .block-container {
    padding-top: 1.2rem;
    padding-bottom: 3rem;
    max-width: 1400px;
  }

  section[data-testid="stSidebar"] > div {
    background: rgba(2, 6, 23, 0.55);
    border-right: 1px solid rgba(148, 163, 184, 0.12);
  }

  div[data-testid="stMetric"] {
    background: rgba(2, 6, 23, 0.35);
    border: 1px solid rgba(148, 163, 184, 0.12);
    padding: 12px 14px;
    border-radius: 16px;
  }

  div[data-testid="stPlotlyChart"] {
    background: rgba(2, 6, 23, 0.25);
    border: 1px solid rgba(148, 163, 184, 0.12);
    border-radius: 16px;
    padding: 8px;
  }

  .stButton > button {
    border-radius: 14px;
    padding: 0.6rem 0.9rem;
    border: 1px solid rgba(148, 163, 184, 0.18);
  }

  .stTextInput input,
  .stTextArea textarea,
  .stSelectbox div[data-baseweb="select"] > div,
  .stMultiSelect div[data-baseweb="select"] > div {
    border-radius: 14px;
  }
</style>
        """,
        unsafe_allow_html=True,
    )
