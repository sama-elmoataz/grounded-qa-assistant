import html
import json

import streamlit as st
import streamlit.components.v1 as components

from src.retrieval import (
    load_embeddings,
    load_vectorstore,
    retrieve_documents,
)
from src.reranking import (
    create_reranker,
    rerank_documents,
)
from src.generation import (
    create_llm,
    create_prompt,
    generate_answer,
    get_sources,
)


# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="NovaDesk Assistant",
    page_icon="N",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# Light Luxury Theme
# ============================================================

st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@600;700&display=swap');

    :root {
        --ivory: #f7f3ec;
        --ivory-2: #fcfaf6;
        --paper: #ffffff;
        --paper-warm: #fffdf9;
        --navy: #142238;
        --navy-2: #253953;
        --muted: #798492;
        --muted-2: #5f6d7d;
        --gold: #b99358;
        --gold-dark: #8f6e3d;
        --gold-soft: #f3e7d4;
        --sage: #839b92;
        --line: #e7ddd0;
        --line-soft: #f0e9df;
        --shadow-1: 0 8px 24px rgba(52, 43, 31, 0.055);
        --shadow-2: 0 18px 44px rgba(52, 43, 31, 0.085);
        --shadow-3: 0 28px 75px rgba(52, 43, 31, 0.11);
    }

    html, body, [class*="css"] {
        font-family: "DM Sans", system-ui, -apple-system, sans-serif;
    }

    .stApp {
        background:
            radial-gradient(circle at 6% 0%, rgba(185,147,88,0.10), transparent 26%),
            radial-gradient(circle at 97% 4%, rgba(131,155,146,0.09), transparent 23%),
            linear-gradient(180deg, var(--ivory-2) 0%, var(--ivory) 100%);
        color: var(--navy);
    }

    header[data-testid="stHeader"] {
        background: transparent;
    }

    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }

    .main .block-container {
        max-width: 1180px;
        padding-top: 1.15rem;
        padding-bottom: 7rem;
    }

    h1, h2, h3, h4, h5, h6 {
        color: var(--navy) !important;
        letter-spacing: -0.025em;
    }

    .stMarkdown p,
    .stMarkdown li,
    .stMarkdown blockquote,
    .stMarkdown strong,
    .stMarkdown em {
        color: var(--navy-2) !important;
    }

    /* ========================================================
       Sidebar
       ======================================================== */

    section[data-testid="stSidebar"] {
        background: rgba(252,250,246,0.98);
        border-right: 1px solid var(--line);
    }

    section[data-testid="stSidebar"] .block-container {
        padding-top: 1.2rem;
    }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] label {
        color: var(--navy) !important;
    }

    .brand-card {
        border: 1px solid var(--line);
        border-radius: 25px;
        background:
            radial-gradient(circle at 100% 0%, rgba(185,147,88,.16), transparent 32%),
            linear-gradient(145deg, #ffffff, #faf5ed);
        box-shadow: var(--shadow-2);
        padding: 1rem;
        margin-bottom: 1rem;
    }

    .brand-row {
        display: flex;
        align-items: center;
        gap: .85rem;
    }

    .brand-mark {
        width: 48px;
        height: 48px;
        border-radius: 16px;
        display: flex;
        align-items: center;
        justify-content: center;
        background: linear-gradient(145deg, #132239, #2d4564);
        color: #f8ead1;
        font-family: "Playfair Display", serif;
        font-size: 1.35rem;
        font-weight: 700;
        position: relative;
        overflow: hidden;
        box-shadow: 0 12px 28px rgba(20,34,56,.20);
        flex-shrink: 0;
    }

    .brand-mark::before {
        content: "";
        position: absolute;
        width: 20px;
        height: 20px;
        border-radius: 50%;
        border: 1.5px solid rgba(216,183,128,.95);
        top: -7px;
        right: -7px;
    }

    .brand-mark::after {
        content: "";
        position: absolute;
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background: #d6b778;
        left: 8px;
        bottom: 8px;
    }

    .brand-name {
        font-family: "Playfair Display", serif;
        color: var(--navy);
        font-size: 1.36rem;
        font-weight: 700;
        line-height: 1;
    }

    .brand-sub {
        color: var(--muted) !important;
        font-size: .77rem;
        line-height: 1.35;
        margin-top: .26rem;
    }

    .brand-note {
        margin-top: .85rem;
        padding-top: .75rem;
        border-top: 1px solid var(--line-soft);
        color: var(--muted) !important;
        font-size: .76rem;
        line-height: 1.55;
    }

    .side-area {
        border: 1px solid var(--line-soft);
        border-radius: 15px;
        background: rgba(255,255,255,.67);
        padding: .72rem .82rem;
        margin: .5rem 0;
    }

    .side-area-title {
        color: var(--navy);
        font-size: .83rem;
        font-weight: 700;
        margin-bottom: .2rem;
    }

    .side-area-text {
        color: var(--muted);
        font-size: .75rem;
        line-height: 1.45;
    }

    /* ========================================================
       Buttons
       ======================================================== */

    .stButton > button,
    .stDownloadButton > button {
        min-height: 2.85rem;
        border-radius: 13px;
        border: 1px solid var(--line);
        background: rgba(255,255,255,.88);
        color: var(--navy) !important;
        font-size: .87rem;
        font-weight: 600;
        box-shadow: var(--shadow-1);
        transition: all .18s ease;
    }

    .stButton > button:hover,
    .stDownloadButton > button:hover {
        transform: translateY(-1px);
        background: #ffffff;
        border-color: #cfd8e3;
        color: var(--navy) !important;
        box-shadow: 0 11px 26px rgba(20,34,56,.08);
    }

    /* Primary / selected state: restrained navy */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #18283f, #28435f) !important;
        border: 1px solid #18283f !important;
        color: #ffffff !important;
        box-shadow: 0 10px 24px rgba(20,34,56,.16) !important;
    }

    .stButton > button[kind="primary"] p,
    .stButton > button[kind="primary"] span,
    .stButton > button[kind="primary"] div {
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
        font-weight: 700 !important;
    }

    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #213550, #31506f) !important;
        border-color: #213550 !important;
        color: #ffffff !important;
        transform: translateY(-1px);
        box-shadow: 0 12px 28px rgba(20,34,56,.19) !important;
    }

    /* ========================================================
       Welcome Hero
       ======================================================== */

    .hero-card {
        position: relative;
        overflow: hidden;
        border: 1px solid var(--line);
        border-radius: 34px;
        background:
            radial-gradient(circle at 92% 12%, rgba(185,147,88,.22), transparent 24%),
            radial-gradient(circle at 10% 90%, rgba(131,155,146,.10), transparent 28%),
            linear-gradient(145deg, rgba(255,255,255,.99), rgba(249,244,236,.98));
        padding: 2.8rem 2.5rem 2.5rem;
        margin: .8rem 0 1.55rem;
        box-shadow: var(--shadow-3);
    }

    .hero-card::after {
        content: "N";
        position: absolute;
        right: 2.15rem;
        bottom: -2.5rem;
        font-family: "Playfair Display", serif;
        font-size: 12rem;
        font-weight: 700;
        color: rgba(185,147,88,.065);
        pointer-events: none;
        line-height: 1;
    }

    .eyebrow {
        display: inline-flex;
        align-items: center;
        border-radius: 999px;
        padding: .43rem .72rem;
        background: #f7eddd;
        border: 1px solid #ead7b8;
        color: var(--gold-dark);
        font-size: .72rem;
        font-weight: 700;
        letter-spacing: .08em;
    }

    .hero-title {
        margin-top: 1rem;
        font-family: "Playfair Display", serif;
        color: var(--navy);
        font-size: 3.3rem;
        line-height: 1.02;
        letter-spacing: -.045em;
        max-width: 760px;
    }

    .hero-copy {
        max-width: 760px;
        color: var(--muted-2);
        font-size: 1rem;
        line-height: 1.72;
        margin-top: 1rem;
    }

    .hero-pills,
    .chat-pills {
        display: flex;
        gap: .52rem;
        flex-wrap: wrap;
        margin-top: 1.15rem;
    }

    .hero-pill,
    .chat-pill {
        border: 1px solid var(--line);
        border-radius: 999px;
        background: rgba(255,255,255,.72);
        padding: .46rem .7rem;
        color: var(--navy-2);
        font-size: .75rem;
        font-weight: 600;
    }

    .section-label {
        margin: 1.25rem 0 .7rem;
        color: var(--gold-dark);
        font-size: .72rem;
        font-weight: 700;
        letter-spacing: .14em;
        text-transform: uppercase;
    }

    /* ========================================================
       Search Input
       ======================================================== */

    div[data-testid="stTextInput"] input {
        height: 4.05rem;
        border-radius: 18px;
        border: 1px solid #ded3c5;
        background: rgba(255,255,255,.96);
        color: var(--navy) !important;
        -webkit-text-fill-color: var(--navy) !important;
        font-size: .98rem;
        padding-left: 1rem;
        box-shadow: var(--shadow-1);
    }

    div[data-testid="stTextInput"] input::placeholder {
        color: #9aa1a8 !important;
        -webkit-text-fill-color: #9aa1a8 !important;
    }

    div[data-testid="stTextInput"] input:focus {
        border-color: #c2a16c;
        box-shadow: 0 0 0 1px rgba(185,147,88,.30), 0 14px 30px rgba(75,59,35,.07);
    }

    /* ========================================================
       Reliable Category Switcher (replaces st.tabs)
       ======================================================== */

    .category-shell {
        border: 1px solid var(--line);
        border-radius: 22px;
        background: rgba(255,255,255,.64);
        padding: .85rem .9rem .92rem;
        box-shadow: var(--shadow-1);
        margin-top: .25rem;
    }

    .category-detail {
        border: 1px solid var(--line-soft);
        border-radius: 17px;
        background: linear-gradient(145deg, #fff, #fbf7f0);
        padding: 1rem 1.05rem;
        margin-top: .55rem;
    }

    .category-detail-title {
        color: var(--navy);
        font-weight: 700;
        font-size: .92rem;
        margin-bottom: .22rem;
    }

    .category-detail-text {
        color: var(--muted);
        font-size: .83rem;
        line-height: 1.55;
    }

    /* ========================================================
       Premium Chat / Results Experience
       ======================================================== */

    .chat-hero {
        border: 1px solid var(--line);
        border-radius: 28px;
        background:
            radial-gradient(circle at 96% 0%, rgba(185,147,88,.15), transparent 28%),
            linear-gradient(145deg, rgba(255,255,255,.96), rgba(249,245,238,.94));
        padding: 1.35rem 1.45rem 1.25rem;
        margin: .55rem 0 1rem;
        box-shadow: var(--shadow-2);
    }

    .chat-hero-title {
        font-family: "Playfair Display", serif;
        color: var(--navy);
        font-size: 1.65rem;
        font-weight: 700;
    }

    .chat-hero-copy {
        color: var(--muted);
        font-size: .88rem;
        line-height: 1.55;
        margin-top: .32rem;
    }

    [data-testid="stChatMessage"] {
        border: 1px solid var(--line) !important;
        border-radius: 22px !important;
        background: rgba(255,255,255,.92) !important;
        box-shadow: var(--shadow-2) !important;
        margin-bottom: 1rem !important;
        overflow: hidden;
    }

    [data-testid="stChatMessage"] > div {
        color: var(--navy-2) !important;
    }

    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] li,
    [data-testid="stChatMessage"] span:not([data-testid="stChatMessageAvatar"]),
    [data-testid="stChatMessage"] strong,
    [data-testid="stChatMessage"] em,
    [data-testid="stChatMessage"] code {
        color: var(--navy-2) !important;
        -webkit-text-fill-color: var(--navy-2) !important;
    }

    [data-testid="stChatMessage"] a {
        color: #765a32 !important;
    }

    .answer-topline {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: .75rem;
        padding-bottom: .7rem;
        margin-bottom: .85rem;
        border-bottom: 1px solid var(--line-soft);
    }

    .answer-brand {
        color: var(--gold-dark);
        font-size: .72rem;
        font-weight: 700;
        letter-spacing: .12em;
        text-transform: uppercase;
    }

    .answer-meta {
        color: var(--muted);
        font-size: .73rem;
        font-weight: 600;
    }

    .source-label {
        color: var(--gold-dark);
        font-size: .71rem;
        font-weight: 700;
        letter-spacing: .13em;
        text-transform: uppercase;
        margin-bottom: .55rem;
    }

    .source-count-badge {
        display: inline-flex;
        border-radius: 999px;
        padding: .38rem .62rem;
        margin-bottom: .65rem;
        background: #f6ecdc;
        border: 1px solid #e7d4b6;
        color: #7f6338;
        font-size: .73rem;
        font-weight: 700;
    }

    .source-card {
        border: 1px solid var(--line);
        border-radius: 15px;
        background: linear-gradient(180deg, #fff, #fbf7f1);
        padding: .78rem .62rem;
        text-align: center;
        box-shadow: var(--shadow-1);
    }

    .source-number {
        font-family: "Playfair Display", serif;
        font-size: 1.18rem;
        font-weight: 700;
        color: var(--navy);
        line-height: 1;
    }

    .source-text {
        margin-top: .31rem;
        color: var(--muted);
        font-size: .66rem;
        letter-spacing: .08em;
        text-transform: uppercase;
    }

    /* Force generated markdown tables to remain readable in light mode */
    .stMarkdown table,
    [data-testid="stChatMessage"] table {
        width: 100% !important;
        border-collapse: separate !important;
        border-spacing: 0 !important;
        overflow: hidden;
        border: 1px solid var(--line) !important;
        border-radius: 16px !important;
        background: #ffffff !important;
        margin: .9rem 0 1.05rem !important;
        box-shadow: var(--shadow-1);
    }

    .stMarkdown thead tr,
    [data-testid="stChatMessage"] thead tr {
        background: #f5ead9 !important;
    }

    .stMarkdown th,
    [data-testid="stChatMessage"] th {
        background: #f5ead9 !important;
        color: var(--navy) !important;
        -webkit-text-fill-color: var(--navy) !important;
        font-weight: 700 !important;
        padding: .8rem .85rem !important;
        border-bottom: 1px solid #dfceb3 !important;
        border-right: 1px solid #eadfce !important;
    }

    .stMarkdown td,
    [data-testid="stChatMessage"] td {
        background: #ffffff !important;
        color: var(--navy-2) !important;
        -webkit-text-fill-color: var(--navy-2) !important;
        padding: .82rem .85rem !important;
        border-bottom: 1px solid var(--line-soft) !important;
        border-right: 1px solid var(--line-soft) !important;
        vertical-align: top !important;
    }

    .stMarkdown tr:last-child td,
    [data-testid="stChatMessage"] tr:last-child td {
        border-bottom: 0 !important;
    }

    /* Expanders: keep them inside the light-luxury palette even with a dark Streamlit theme */
    [data-testid="stExpander"] {
        border: 1px solid #dfd1bd !important;
        border-radius: 17px !important;
        background: #fffdf9 !important;
        overflow: hidden !important;
        box-shadow: 0 8px 22px rgba(70,55,34,.055) !important;
    }

    [data-testid="stExpander"] details,
    [data-testid="stExpander"] > details {
        background: #fffdf9 !important;
        color: var(--navy) !important;
    }

    [data-testid="stExpander"] summary {
        background: linear-gradient(135deg, #fbf5eb, #f5e9d7) !important;
        color: #5d4728 !important;
        -webkit-text-fill-color: #5d4728 !important;
        border-bottom: 1px solid transparent !important;
        padding-top: .85rem !important;
        padding-bottom: .85rem !important;
    }

    [data-testid="stExpander"] details[open] summary {
        border-bottom-color: #eadbc5 !important;
    }

    [data-testid="stExpander"] summary p,
    [data-testid="stExpander"] summary span,
    [data-testid="stExpander"] summary div {
        color: #5d4728 !important;
        -webkit-text-fill-color: #5d4728 !important;
        font-weight: 650 !important;
    }

    [data-testid="stExpander"] summary svg,
    [data-testid="stExpander"] summary svg path {
        color: #a47b43 !important;
        fill: #a47b43 !important;
        stroke: #a47b43 !important;
    }

    [data-testid="stExpander"] p,
    [data-testid="stExpander"] div,
    [data-testid="stExpander"] span,
    [data-testid="stExpander"] label {
        color: var(--navy-2) !important;
        -webkit-text-fill-color: var(--navy-2) !important;
    }

    /* Follow-up area */
    .followup-card {
        border: 1px solid var(--line);
        border-radius: 20px;
        background: rgba(255,255,255,.68);
        padding: .9rem 1rem .25rem;
        margin: .55rem 0 1rem;
        box-shadow: var(--shadow-1);
    }

    .followup-title {
        color: var(--navy);
        font-size: .86rem;
        font-weight: 700;
    }

    .followup-copy {
        color: var(--muted);
        font-size: .76rem;
        margin-top: .17rem;
        margin-bottom: .55rem;
    }

    /* Chat input dock: override Streamlit's dark bottom container as well as the input */
    [data-testid="stBottom"],
    [data-testid="stBottom"] > div,
    [data-testid="stBottom"] > div > div,
    [data-testid="stBottomBlockContainer"],
    .stBottom,
    .stBottom > div {
        background: linear-gradient(180deg, rgba(247,243,236,0) 0%, rgba(247,243,236,.96) 36%, #f7f3ec 100%) !important;
        background-color: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }

    [data-testid="stBottom"] {
        padding-top: .65rem !important;
    }

    [data-testid="stChatInput"] {
        background: #ffffff !important;
        border: 1px solid #d8dee7 !important;
        border-radius: 18px !important;
        box-shadow: 0 15px 36px rgba(20,34,56,.07) !important;
        transition: border-color .18s ease, box-shadow .18s ease !important;
        overflow: hidden !important;
    }

    [data-testid="stChatInput"]:focus-within {
        border-color: #9fb0c2 !important;
        box-shadow: 0 0 0 3px rgba(37,57,83,.08), 0 15px 36px rgba(20,34,56,.08) !important;
    }

    [data-testid="stChatInput"] > div,
    [data-testid="stChatInput"] > div > div,
    [data-testid="stChatInput"] div[data-baseweb="textarea"],
    [data-testid="stChatInput"] div[data-baseweb="base-input"],
    [data-testid="stChatInput"] div[data-baseweb="input"] {
        background: transparent !important;
        background-color: transparent !important;
    }

    [data-testid="stChatInput"] textarea {
        background: transparent !important;
        color: var(--navy-2) !important;
        -webkit-text-fill-color: var(--navy-2) !important;
        caret-color: var(--navy) !important;
    }

    [data-testid="stChatInput"] textarea::placeholder {
        color: #9aa3ad !important;
        -webkit-text-fill-color: #9aa3ad !important;
        opacity: 1 !important;
    }

    [data-testid="stChatInput"] button {
        background: linear-gradient(135deg, #18283f, #28435f) !important;
        color: #ffffff !important;
        border: 1px solid #18283f !important;
        border-radius: 11px !important;
        box-shadow: 0 8px 18px rgba(20,34,56,.16) !important;
        transition: all .18s ease !important;
    }

    [data-testid="stChatInput"] button:hover {
        background: linear-gradient(135deg, #213550, #31506f) !important;
        transform: translateY(-1px);
        box-shadow: 0 10px 22px rgba(20,34,56,.19) !important;
    }

    [data-testid="stChatInput"] button svg,
    [data-testid="stChatInput"] button svg path {
        color: #ffffff !important;
        fill: #ffffff !important;
        stroke: #ffffff !important;
    }

    hr {
        border-color: var(--line-soft) !important;
    }

    .fallback-note {
        border: 1px solid #e5d4b5;
        background: #fbf4e8;
        border-radius: 15px;
        padding: .75rem .85rem;
        color: #735d39;
        font-size: .8rem;
        line-height: 1.5;
        margin: .65rem 0;
    }

    .footer-note {
        text-align: center;
        color: var(--muted);
        font-size: .75rem;
        margin-top: 2rem;
        padding: 1rem;
    }

    /* ========================================================
       Premium conversation layout
       ======================================================== */

    .conversation-stack {
        margin-top: .6rem;
    }

    .user-query-card {
        max-width: 860px;
        margin: .65rem 0 1rem auto;
        border: 1px solid #ddcfbc;
        border-radius: 20px 20px 7px 20px;
        background: linear-gradient(145deg, #fffdf9, #f8f0e4);
        padding: .9rem 1.05rem;
        box-shadow: var(--shadow-1);
    }

    .user-query-label {
        color: var(--gold-dark);
        font-size: .66rem;
        font-weight: 700;
        letter-spacing: .12em;
        text-transform: uppercase;
        margin-bottom: .28rem;
    }

    .user-query-text {
        color: var(--navy) !important;
        font-size: .94rem;
        font-weight: 600;
        line-height: 1.55;
    }

    /* Only bordered st.container elements are used for assistant answer cards. */
    [data-testid="stVerticalBlockBorderWrapper"] {
        border: 1px solid #dfd3c3 !important;
        border-radius: 26px !important;
        background:
            radial-gradient(circle at 100% 0%, rgba(185,147,88,.09), transparent 23%),
            rgba(255,255,255,.96) !important;
        box-shadow: var(--shadow-2) !important;
        overflow: hidden !important;
        margin-bottom: 1.15rem !important;
    }

    [data-testid="stVerticalBlockBorderWrapper"] > div {
        padding: .25rem .35rem !important;
    }

    .answer-topline {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
        padding: .15rem 0 .85rem;
        margin-bottom: 1rem;
        border-bottom: 1px solid var(--line-soft);
    }

    .answer-brand-wrap {
        display: flex;
        align-items: center;
        gap: .65rem;
    }

    .answer-mini-mark {
        width: 30px;
        height: 30px;
        border-radius: 10px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        background: linear-gradient(145deg, #142238, #294564);
        color: #f4dfb9;
        font-family: "Playfair Display", serif;
        font-size: .9rem;
        font-weight: 700;
        box-shadow: 0 7px 18px rgba(20,34,56,.14);
    }

    .answer-brand {
        color: var(--gold-dark);
        font-size: .69rem;
        font-weight: 700;
        letter-spacing: .13em;
        text-transform: uppercase;
    }

    .answer-meta {
        color: var(--muted);
        font-size: .72rem;
        font-weight: 600;
        white-space: nowrap;
    }

    .question-separator {
        height: 1px;
        background: linear-gradient(90deg, transparent, #dfd2c0 12%, #dfd2c0 88%, transparent);
        margin: 1rem 0 .8rem;
    }

    .suggested-question-label {
        color: var(--gold-dark);
        font-size: .66rem;
        font-weight: 700;
        letter-spacing: .13em;
        text-transform: uppercase;
        margin: .1rem 0 .5rem;
    }

    /* Custom animated loading state */
    .thinking-card {
        border: 1px solid #dfd3c3;
        border-radius: 24px;
        background:
            radial-gradient(circle at 92% 5%, rgba(185,147,88,.11), transparent 25%),
            linear-gradient(145deg, rgba(255,255,255,.98), rgba(250,246,239,.98));
        padding: 1.1rem 1.2rem;
        margin: .45rem 0 1rem;
        box-shadow: var(--shadow-2);
        overflow: hidden;
        position: relative;
    }

    .thinking-card::after {
        content: "";
        position: absolute;
        top: 0;
        left: -45%;
        width: 42%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,.82), transparent);
        animation: novadeskShimmer 1.8s infinite;
        pointer-events: none;
    }

    .thinking-row {
        display: flex;
        align-items: center;
        gap: .85rem;
    }

    .thinking-orb {
        width: 38px;
        height: 38px;
        border-radius: 13px;
        background: linear-gradient(145deg, #142238, #304d6d);
        position: relative;
        flex: 0 0 auto;
        box-shadow: 0 9px 24px rgba(20,34,56,.18);
    }

    .thinking-orb::before,
    .thinking-orb::after {
        content: "";
        position: absolute;
        border-radius: 50%;
        background: #e2c58f;
        animation: novadeskPulse 1.2s infinite ease-in-out;
    }

    .thinking-orb::before { width: 7px; height: 7px; left: 9px; top: 15px; }
    .thinking-orb::after  { width: 7px; height: 7px; right: 9px; top: 15px; animation-delay: .25s; }

    .thinking-title {
        color: var(--navy);
        font-family: "Playfair Display", serif;
        font-size: 1rem;
        font-weight: 700;
    }

    .thinking-copy {
        color: var(--muted);
        font-size: .78rem;
        margin-top: .16rem;
    }

    .thinking-dots {
        display: inline-flex;
        gap: 4px;
        margin-left: 4px;
        vertical-align: middle;
    }

    .thinking-dots span {
        width: 4px;
        height: 4px;
        border-radius: 50%;
        background: var(--gold);
        animation: novadeskDot 1.2s infinite ease-in-out;
    }

    .thinking-dots span:nth-child(2) { animation-delay: .16s; }
    .thinking-dots span:nth-child(3) { animation-delay: .32s; }

    @keyframes novadeskShimmer {
        0% { left: -45%; }
        100% { left: 115%; }
    }

    @keyframes novadeskPulse {
        0%, 100% { transform: scale(.75); opacity: .55; }
        50% { transform: scale(1.12); opacity: 1; }
    }

    @keyframes novadeskDot {
        0%, 80%, 100% { transform: translateY(0); opacity: .4; }
        40% { transform: translateY(-4px); opacity: 1; }
    }

    /* Supporting excerpts: force true plain-text rendering. */
    .support-page-card {
        border: 1px solid var(--line-soft);
        border-radius: 16px;
        background: #fffdf9;
        padding: .9rem 1rem;
        margin: .65rem 0;
    }

    .support-page-head {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: .7rem;
        margin-bottom: .58rem;
    }

    .support-page-title {
        color: var(--navy);
        font-size: .82rem;
        font-weight: 700;
    }

    .support-page-tag {
        color: var(--gold-dark);
        background: #f7eddd;
        border: 1px solid #ead8bc;
        border-radius: 999px;
        padding: .25rem .5rem;
        font-size: .63rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: .08em;
    }

    .support-page-text {
        color: var(--navy-2) !important;
        -webkit-text-fill-color: var(--navy-2) !important;
        font-family: "DM Sans", system-ui, sans-serif !important;
        font-size: .78rem !important;
        line-height: 1.65 !important;
        white-space: pre-wrap;
        word-break: break-word;
    }

    .support-page-text code,
    .support-page-text pre {
        all: unset !important;
        color: var(--navy-2) !important;
        -webkit-text-fill-color: var(--navy-2) !important;
        font-family: "DM Sans", system-ui, sans-serif !important;
        white-space: pre-wrap !important;
    }

    @media (max-width: 800px) {
        .hero-title { font-size: 2.45rem; }
        .hero-card { padding: 2rem 1.35rem; }
        .main .block-container { padding-left: 1rem; padding-right: 1rem; }
    }
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# RAG Resources
# ============================================================

# show_spinner=False removes Streamlit's "Running load_resources()" cache banner.
@st.cache_resource(show_spinner=False)
def load_resources():
    embeddings = load_embeddings()
    vectorstore = load_vectorstore(embeddings)
    ranker = create_reranker()
    llm = create_llm()
    prompt = create_prompt()
    return vectorstore, ranker, llm, prompt


# ============================================================
# Session State
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "selected_question" not in st.session_state:
    st.session_state.selected_question = None

if "selected_category" not in st.session_state:
    st.session_state.selected_category = "Plans & Pricing"


suggestions = [
    "Compare NovaDesk Starter, Growth, and Enterprise plans, including price, minimum seats, and key inclusions.",
    "What is the NovaDesk failed card payment timeline from grace period to possible suspension?",
    "What are NovaDesk API request-per-minute limits for Starter, Growth, and Enterprise?",
    "What are NovaDesk audit log retention periods for Growth and Enterprise?",
]

category_content = {
    "Plans & Pricing": {
        "short": "Plans & Pricing",
        "description": "Compare plan tiers, pricing, included capabilities, seat requirements, and commercial rules.",
        "question": suggestions[0],
    },
    "Billing & Support": {
        "short": "Billing & Support",
        "description": "Explore invoices, payment rules, grace periods, refunds, support levels, and response targets.",
        "question": suggestions[1],
    },
    "Security & Data": {
        "short": "Security & Data",
        "description": "Review audit logs, retention periods, access controls, encryption, backups, and recovery objectives.",
        "question": suggestions[3],
    },
    "Integrations & API": {
        "short": "Integrations & API",
        "description": "Check native integrations, API rate limits, webhook behavior, and technical onboarding information.",
        "question": suggestions[2],
    },
}

sidebar_areas = [
    ("Plans & Pricing", "Plan tiers, commercial rules, seats, and feature inclusions."),
    ("Billing & Support", "Payments, refunds, support coverage, and response targets."),
    ("Security & Data", "Retention, logs, encryption, backups, and access controls."),
    ("Integrations & API", "Native integrations, API limits, webhooks, and setup guidance."),
]


# ============================================================
# Helpers
# ============================================================

def build_chat_export():
    lines = [
        "# NovaDesk Assistant — Conversation Export",
        "",
        "Generated from the NovaDesk Company Information Assistant.",
        "",
    ]

    for message in st.session_state.messages:
        role = "You" if message["role"] == "user" else "NovaDesk"
        lines.extend([f"## {role}", "", message["content"], ""])

        if message["role"] == "assistant" and message.get("sources"):
            pages = ", ".join(str(page) for page in sorted(set(message["sources"])))
            lines.extend([f"Sources: page(s) {pages}", ""])

    return "\n".join(lines)


def render_copy_button(answer, key_suffix):
    safe_answer = json.dumps(answer).replace("</", "<\\/")
    button_id = f"copy_btn_{key_suffix}"
    label_id = f"copy_label_{key_suffix}"

    components.html(
        f"""
        <style>
          html, body {{
            margin: 0;
            padding: 0;
            background: transparent;
            overflow: visible;
          }}
          .copy-wrap {{
            display: flex;
            justify-content: flex-end;
            align-items: center;
            min-height: 44px;
            padding: 4px 2px 7px 2px;
            box-sizing: border-box;
            overflow: visible;
          }}
          .copy-button {{
            min-height: 34px;
            padding: 0 14px;
            border-radius: 11px;
            border: 1px solid #d8dee7;
            background: #ffffff;
            color: #253953;
            font-family: Arial, sans-serif;
            font-size: 11px;
            font-weight: 700;
            cursor: pointer;
            box-shadow: 0 5px 14px rgba(73,56,35,.07);
            box-sizing: border-box;
            line-height: 1;
          }}
          .copy-button:hover {{
            border-color: #bcc8d5;
            background: #f7f9fb;
          }}
        </style>
        <div class="copy-wrap">
          <button class="copy-button" id="{button_id}" onclick="copyAnswer()">
            <span id="{label_id}">Copy answer</span>
          </button>
        </div>
        <script>
          const answerText = {safe_answer};
          async function copyAnswer() {{
            const label = document.getElementById("{label_id}");
            try {{
              await navigator.clipboard.writeText(answerText);
            }} catch (err) {{
              const textarea = document.createElement("textarea");
              textarea.value = answerText;
              textarea.style.position = "fixed";
              textarea.style.opacity = "0";
              document.body.appendChild(textarea);
              textarea.select();
              document.execCommand("copy");
              document.body.removeChild(textarea);
            }}
            label.innerText = "Copied";
            setTimeout(() => label.innerText = "Copy answer", 1500);
          }}
        </script>
        """,
        height=52,
        scrolling=False,
    )


def is_not_found_answer(answer):
    normalized = answer.strip().lower()
    phrases = (
        "information is not stated",
        "not stated in the novadesk company information guide",
        "not provided in the novadesk company information guide",
        "cannot be found in the provided context",
    )
    return any(phrase in normalized for phrase in phrases)


def diagnose_retrieved_context(documents):
    if not documents:
        return "empty"
    context = " ".join(document.page_content for document in documents).lower()
    return "novadesk_context" if "novadesk" in context else "wrong_corpus"


def render_source_cards(sources):
    unique_sources = sorted(set(sources)) if sources else []

    if not unique_sources:
        st.markdown('<div class="source-count-badge">0 reference pages</div>', unsafe_allow_html=True)
        return

    word = "page" if len(unique_sources) == 1 else "pages"
    st.markdown(
        f'<div class="source-count-badge">{len(unique_sources)} reference {word}</div>',
        unsafe_allow_html=True,
    )

    cols = st.columns(min(len(unique_sources), 4))
    for index, page in enumerate(unique_sources):
        with cols[index % len(cols)]:
            st.markdown(
                f"""
                <div class="source-card">
                    <div class="source-number">{page}</div>
                    <div class="source-text">Reference page</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_supporting_documents(documents):
    if not documents:
        return

    # Group excerpts by source page and render as escaped HTML so PDF text is
    # never interpreted as Markdown/code (which caused the black highlights).
    grouped = {}
    for document in documents:
        page = document.metadata.get("page", 0) + 1
        text = " ".join(document.page_content.split())
        if text and text not in grouped.setdefault(page, []):
            grouped[page].append(text)

    with st.expander("View supporting information"):
        for page in sorted(grouped):
            excerpts = grouped[page]
            combined = "\n\n".join(excerpts)
            safe_text = html.escape(combined)
            excerpt_label = "1 excerpt" if len(excerpts) == 1 else f"{len(excerpts)} excerpts"
            st.markdown(
                f"""
                <div class="support-page-card">
                    <div class="support-page-head">
                        <div class="support-page-title">Page {page}</div>
                        <div class="support-page-tag">{excerpt_label}</div>
                    </div>
                    <div class="support-page-text">{safe_text}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_user_question(question):
    st.markdown(
        f"""
        <div class="user-query-card">
            <div class="user-query-label">Your question</div>
            <div class="user-query-text">{html.escape(question)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_assistant_message(message, key_suffix):
    sources = message.get("sources", [])
    source_count = len(set(sources)) if sources else 0

    with st.container(border=True):
        st.markdown(
            f"""
            <div class="answer-topline">
                <div class="answer-brand-wrap">
                    <div class="answer-mini-mark">N</div>
                    <div class="answer-brand">NovaDesk answer</div>
                </div>
                <div class="answer-meta">{source_count} reference page{'s' if source_count != 1 else ''}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(message["content"])
        render_copy_button(message["content"], key_suffix)

        if "sources" in message:
            st.divider()
            heading = "Retrieved context" if is_not_found_answer(message["content"]) else "References"
            st.markdown(f'<div class="source-label">{heading}</div>', unsafe_allow_html=True)
            render_source_cards(sources)
            render_supporting_documents(message.get("documents"))


def render_loading_state():
    return st.markdown(
        """
        <div class="thinking-card">
            <div class="thinking-row">
                <div class="thinking-orb"></div>
                <div>
                    <div class="thinking-title">NovaDesk is preparing your answer
                        <span class="thinking-dots"><span></span><span></span><span></span></span>
                    </div>
                    <div class="thinking-copy">Searching the guide, ranking the strongest references, and composing a grounded response.</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_category_switcher(compact=False):
    if not compact:
        st.markdown('<div class="section-label">Browse by category</div>', unsafe_allow_html=True)

    names = list(category_content.keys())
    cols = st.columns(4)

    for idx, name in enumerate(names):
        with cols[idx]:
            active = st.session_state.selected_category == name
            if st.button(
                category_content[name]["short"],
                key=f"category_tab_{'compact' if compact else 'home'}_{idx}",
                use_container_width=True,
                type="primary" if active else "secondary",
            ):
                st.session_state.selected_category = name
                st.rerun()

    selected = category_content[st.session_state.selected_category]
    st.markdown(
        f"""
        <div class="category-detail">
            <div class="category-detail-title">{html.escape(st.session_state.selected_category)}</div>
            <div class="category-detail-text">{html.escape(selected['description'])}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="question-separator"></div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="suggested-question-label">Suggested question · click to ask</div>',
        unsafe_allow_html=True,
    )
    if st.button(
        selected["question"],
        key=f"category_suggested_question_{'compact' if compact else 'home'}",
        use_container_width=True,
    ):
        st.session_state.selected_question = selected["question"]
        st.rerun()



def render_followups(current_question):
    followups = [q for q in suggestions if q != current_question][:3]
    if not followups:
        return

    st.markdown(
        """
        <div class="followup-card">
            <div class="followup-title">Continue exploring</div>
            <div class="followup-copy">Jump to another common NovaDesk question.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cols = st.columns(len(followups))
    for idx, question in enumerate(followups):
        with cols[idx]:
            if st.button(
                question,
                key=f"followup_{len(st.session_state.messages)}_{idx}",
                use_container_width=True,
            ):
                st.session_state.selected_question = question
                st.rerun()


def render_sidebar():
    with st.sidebar:
        st.markdown(
            """
            <div class="brand-card">
                <div class="brand-row">
                    <div class="brand-mark">N</div>
                    <div>
                        <div class="brand-name">NovaDesk</div>
                        <div class="brand-sub">Company Information Assistant</div>
                    </div>
                </div>
                <div class="brand-note">
                    A refined knowledge experience grounded in the NovaDesk Company Information Guide.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button("＋  New conversation", use_container_width=True):
            st.session_state.messages = []
            st.session_state.selected_question = None
            st.rerun()

        if st.session_state.messages:
            st.download_button(
                "⇩  Export chat",
                data=build_chat_export(),
                file_name="novadesk_chat_export.md",
                mime="text/markdown",
                use_container_width=True,
            )

        st.divider()
        st.subheader("Quick Questions")
        for idx, suggestion in enumerate(suggestions):
            if st.button(suggestion, key=f"sidebar_sug_{idx}", use_container_width=True):
                st.session_state.selected_question = suggestion
                st.rerun()

        st.divider()
        st.subheader("Knowledge Areas")
        for title, text in sidebar_areas:
            st.markdown(
                f"""
                <div class="side-area">
                    <div class="side-area-title">{title}</div>
                    <div class="side-area-text">{text}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.divider()
        st.caption("Grounded answers • Page references • Supporting context")


def render_welcome():
    st.markdown(
        """
        <div class="hero-card">
            <div class="eyebrow">NOVA DESK • COMPANY KNOWLEDGE</div>
            <div class="hero-title">Company knowledge, with a quieter kind of intelligence.</div>
            <div class="hero-copy">
                Ask natural questions about NovaDesk and receive concise, source-backed answers from the company guide —
                designed to feel more like a premium knowledge desk than a search box.
            </div>
            <div class="hero-pills">
                <div class="hero-pill">Plans & Pricing</div>
                <div class="hero-pill">Billing & Support</div>
                <div class="hero-pill">Security & Data</div>
                <div class="hero-pill">Integrations & API</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_chat_header():
    user_count = sum(1 for message in st.session_state.messages if message["role"] == "user")
    st.markdown(
        f"""
        <div class="chat-hero">
            <div class="chat-hero-title">NovaDesk Assistant</div>
            <div class="chat-hero-copy">
                A focused company knowledge session. Answers are grounded in the NovaDesk Company Information Guide.
            </div>
            <div class="chat-pills">
                <div class="chat-pill">{user_count} question{'s' if user_count != 1 else ''}</div>
                <div class="chat-pill">Page-level references</div>
                <div class="chat-pill">Supporting context</div>
                <div class="chat-pill">Exportable conversation</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# Sidebar
# ============================================================

render_sidebar()


# ============================================================
# Selected Question
# ============================================================

user_question = None
if st.session_state.selected_question:
    user_question = st.session_state.selected_question
    st.session_state.selected_question = None


# ============================================================
# Welcome / Conversation Screen
# ============================================================

if not st.session_state.messages:
    render_welcome()

    st.markdown('<div class="section-label">Ask NovaDesk</div>', unsafe_allow_html=True)
    initial_question = st.text_input(
        "Ask a question",
        placeholder="Ask about plans, pricing, billing, support, security, integrations, or onboarding...",
        label_visibility="collapsed",
        key="initial_question",
    )

    c1, c2, c3 = st.columns([1, 1.45, 1])
    with c2:
        ask_clicked = st.button("Ask NovaDesk", use_container_width=True, type="primary")

    render_category_switcher(compact=False)

    if not user_question and ask_clicked and initial_question.strip():
        user_question = initial_question.strip()

else:
    render_chat_header()

    # Keep topic navigation available after search instead of disappearing.
    with st.expander("Explore another knowledge area"):
        render_category_switcher(compact=True)

    last_user_question = None

    st.markdown('<div class="conversation-stack">', unsafe_allow_html=True)
    for message_index, message in enumerate(st.session_state.messages):
        if message["role"] == "user":
            last_user_question = message["content"]
            render_user_question(message["content"])
        else:
            render_assistant_message(message, f"history_{message_index}")
    st.markdown('</div>', unsafe_allow_html=True)

    if last_user_question:
        render_followups(last_user_question)

    chat_input_val = st.chat_input("Ask another question...")
    if not user_question and chat_input_val:
        user_question = chat_input_val


# ============================================================
# Process New Question
# ============================================================

if user_question:
    user_question = user_question.strip()

    if user_question:
        st.session_state.messages.append({"role": "user", "content": user_question})
        render_user_question(user_question)

        loading_placeholder = st.empty()
        with loading_placeholder.container():
            render_loading_state()

        try:
            vectorstore, ranker, llm, prompt = load_resources()
            documents = retrieve_documents(user_question, vectorstore)
            reranked_documents = rerank_documents(user_question, documents, ranker)
            answer = generate_answer(user_question, reranked_documents, llm, prompt)
            sources = get_sources(reranked_documents)

            loading_placeholder.empty()

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                    "documents": reranked_documents,
                }
            )
            st.rerun()

        except Exception as error:
            loading_placeholder.empty()
            st.error("Sorry, I couldn't process that question.")
            st.caption(f"Technical details: {error}")


# ============================================================
# Footer
# ============================================================

if not st.session_state.messages:
    st.markdown(
        '<div class="footer-note">NovaDesk Company Information Assistant • Grounded retrieval with page-level references</div>',
        unsafe_allow_html=True,
    )
