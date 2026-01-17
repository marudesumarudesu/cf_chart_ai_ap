import streamlit as st


def inject_css() -> None:
    """High-contrast UI CSS for readability on dark background."""
    st.markdown(
        """
<style>
  /* Hide menu/footer, but keep header because it contains sidebar toggle */
  #MainMenu {visibility: hidden;}
  footer {visibility: hidden;}
  header[data-testid="stHeader"] {background: rgba(0,0,0,0);}

  :root {
    --bg0: #050A18;
    --bg1: #0B1224;
    --card: rgba(15, 23, 42, 0.62);
    --card2: rgba(15, 23, 42, 0.42);
    --border: rgba(148, 163, 184, 0.22);

    --text: #F8FAFC;
    --text2: rgba(226, 232, 240, 0.92);
    --muted: rgba(226, 232, 240, 0.78);

    --input-bg: rgba(2, 6, 23, 0.55);
    --input-border: rgba(148, 163, 184, 0.28);

    --accent: #E11D48;
  }

  html, body { font-size: 16px; }

  .stApp {
    background: linear-gradient(135deg, var(--bg1) 0%, var(--bg0) 100%);
    color: var(--text);
  }

  h1,h2,h3,h4,h5,h6, p, span, label, div {
    color: var(--text);
  }

  .block-container {
    padding-top: 1.2rem;
    padding-bottom: 3rem;
    max-width: 1450px;
  }

  /* ---- Sidebar ---- */
  section[data-testid="stSidebar"] > div {
    background: rgba(5, 10, 24, 0.72);
    border-right: 1px solid var(--border);
    backdrop-filter: blur(10px);
  }

  /* Ensure sidebar text is always readable */
  section[data-testid="stSidebar"] * {
    color: var(--text) !important;
  }

  /* Inputs look dark with light text */
  .stTextInput input,
  .stTextArea textarea,
  .stSelectbox div[data-baseweb="select"] > div,
  .stMultiSelect div[data-baseweb="select"] > div {
    border-radius: 14px !important;
    background: var(--input-bg) !important;
    color: var(--text) !important;
    border: 1px solid var(--input-border) !important;
  }

  /* Placeholder */
  .stTextInput input::placeholder,
  .stTextArea textarea::placeholder {
    color: rgba(226, 232, 240, 0.55) !important;
  }

  /* Dropdown menu background + text */
  div[data-baseweb="menu"] {
    background: rgba(2, 6, 23, 0.92) !important;
    border: 1px solid var(--border) !important;
  }
  div[data-baseweb="menu"] * {
    color: var(--text) !important;
  }

  /* Multiselect tag/chip readability */
  span[data-baseweb="tag"] {
    background: rgba(2, 6, 23, 0.65) !important;
    border: 1px solid rgba(148, 163, 184, 0.28) !important;
    color: var(--text) !important;
  }

  /* ---- Sidebar collapsed toggle button ----
     Streamlit puts it in the header; we make it stand out. */
  button[data-testid="collapsedControl"] {
    color: var(--text) !important;
    background: rgba(248, 250, 252, 0.14) !important;
    border: 1px solid rgba(248, 250, 252, 0.25) !important;
    border-radius: 14px !important;
    box-shadow: 0 6px 18px rgba(0,0,0,0.35) !important;
  }
  button[data-testid="collapsedControl"]:hover {
    background: rgba(248, 250, 252, 0.22) !important;
    border-color: rgba(248, 250, 252, 0.36) !important;
  }

  /* ---- Metric cards ---- */
  div[data-testid="stMetric"] {
    background: var(--card);
    border: 1px solid var(--border);
    padding: 12px 14px;
    border-radius: 16px;
  }
  div[data-testid="stMetric"] * {
    color: var(--text) !important;
  }

  /* ---- Plotly container ---- */
  div[data-testid="stPlotlyChart"] {
    background: var(--card2);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 8px;
  }

  /* Buttons */
  .stButton > button {
    border-radius: 14px;
    padding: 0.6rem 0.95rem;
    border: 1px solid rgba(148, 163, 184, 0.28);
    background: rgba(2, 6, 23, 0.35);
    color: var(--text);
  }
  .stButton > button:hover {
    border-color: rgba(226, 232, 240, 0.40);
    background: rgba(2, 6, 23, 0.50);
  }

  /* Captions */
  .stCaption, .stMarkdown small {
    color: var(--muted) !important;
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
    background: rgba(15, 23, 42, 0.55);
  }
  .ticker-bar-title {
    font-weight: 800;
    color: var(--text);
    white-space: nowrap;
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
    border: 1px solid rgba(148, 163, 184, 0.28);
    background: rgba(2, 6, 23, 0.55);
    color: var(--text);
    font-size: 0.93rem;
    line-height: 1.1;
  }
</style>
        """,
        unsafe_allow_html=True,
    )
