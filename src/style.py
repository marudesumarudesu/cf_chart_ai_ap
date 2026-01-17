import streamlit as st


def inject_css() -> None:
    """Add lightweight CSS to improve aesthetics without breaking Streamlit."""
    st.markdown(
        """
<style>
  #MainMenu {visibility: hidden;}
  footer {visibility: hidden;}
  /* Keep header visible: it contains the sidebar toggle button.
     We only make it visually unobtrusive. */
  header[data-testid="stHeader"] {background: rgba(0,0,0,0);}
  div[data-testid="stToolbar"] {right: 1rem;}

  /* Make sure the collapsed sidebar control stays visible */
  button[data-testid="collapsedControl"] {
    color: rgba(248, 250, 252, 0.92) !important;
    background: rgba(2, 6, 23, 0.35) !important;
    border: 1px solid rgba(148, 163, 184, 0.25) !important;
    border-radius: 12px !important;
  }

  :root {
    --bg0: #050A18;
    --bg1: #0F172A;
    --card: rgba(15, 23, 42, 0.55);
    --card2: rgba(15, 23, 42, 0.35);
    --border: rgba(148, 163, 184, 0.16);
    --text: #F8FAFC;
    --muted: rgba(226, 232, 240, 0.86);
  }

  html, body {
    font-size: 16px;
  }

  .stApp {
    background: linear-gradient(135deg, var(--bg1) 0%, var(--bg0) 100%);
    color: var(--text);
  }

  .block-container {
    padding-top: 1.2rem;
    padding-bottom: 3rem;
    max-width: 1400px;
  }

  section[data-testid="stSidebar"] > div {
    background: rgba(5, 10, 24, 0.55);
    border-right: 1px solid var(--border);
    backdrop-filter: blur(10px);
  }

  /* Sidebar readability */
  section[data-testid="stSidebar"] label,
  section[data-testid="stSidebar"] p,
  section[data-testid="stSidebar"] span,
  section[data-testid="stSidebar"] div {
    color: var(--text) !important;
  }

  div[data-testid="stMetric"] {
    background: var(--card);
    border: 1px solid var(--border);
    padding: 12px 14px;
    border-radius: 16px;
  }

  div[data-testid="stMetric"] label, div[data-testid="stMetric"] div {
    color: var(--text);
  }

  div[data-testid="stPlotlyChart"] {
    background: var(--card2);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 8px;
  }

  .stButton > button {
    border-radius: 14px;
    padding: 0.6rem 0.9rem;
    border: 1px solid rgba(148, 163, 184, 0.22);
  }

  .stTextInput input,
  .stTextArea textarea,
  .stSelectbox div[data-baseweb="select"] > div,
  .stMultiSelect div[data-baseweb="select"] > div {
    border-radius: 14px;
    background: rgba(2, 6, 23, 0.30);
    color: var(--text);
    border: 1px solid rgba(148, 163, 184, 0.22);
  }

  /* Dropdown menu */
  div[data-baseweb="popover"] * {
    color: var(--text) !important;
  }

  /* Selected tickers bar */
  .ticker-bar {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 10px 12px;
    margin: 10px 0 6px 0;
    border: 1px solid var(--border);
    border-radius: 16px;
    background: rgba(15, 23, 42, 0.45);
  }

  .ticker-bar-title {
    font-weight: 700;
    color: var(--text);
    white-space: nowrap;
    opacity: 0.95;
  }

  .ticker-chips {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
  }

  .ticker-chip {
    display: inline-block;
    padding: 6px 10px;
    border-radius: 999px;
    border: 1px solid rgba(148, 163, 184, 0.22);
    background: rgba(2, 6, 23, 0.25);
    color: var(--text);
    font-size: 0.92rem;
    line-height: 1.1;
  }

  .stCaption, .stMarkdown small {
    color: var(--muted) !important;
  }
</style>
        """,
        unsafe_allow_html=True,
    )
