import streamlit as st
import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import re

# ==============================================================================
# 0. CẤU HÌNH TRANG & CSS (PHẦN QUAN TRỌNG ĐỂ GIAO DIỆN ĐẸP)
# ==============================================================================
st.set_page_config(
    page_title="Cinematch - Gợi ý phim",
    page_icon="🍿",
    layout="wide",
    initial_sidebar_state="expanded"
)

def inject_custom_css():
    st.markdown("""
    <style>
    /* 1. NỀN CHUYỂN MÀU (GRADIENT BACKGROUND) */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        color: #ffffff;
    }

    /* 2. TÙY CHỈNH THANH SIDEBAR */
    section[data-testid="stSidebar"] {
        background-color: rgba(0, 0, 0, 0.2);
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }

    /* 3. HIỆU ỨNG CARD (THẺ PHIM) */
    .movie-card-container {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }

    /* 4. TÙY CHỈNH NÚT BẤM (BUTTONS) */
    .stButton > button {
        background: linear-gradient(90deg, #E50914 0%, #ff6b6b 100%);
        color: white;
        border: none;
        border-radius: 25px;
        font-weight: bold;
        padding: 0.5rem 1rem;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0 0 15px rgba(229, 9, 20, 0.6);
    }

    /* Nút phụ (Secondary) */
    button[kind="secondary"] {
        background: transparent !important;
        border: 1px solid rgba(255,255,255,0.5) !important;
    }

    /* 5. TIÊU ĐỀ & CHỮ */
    h1, h2, h3 {
        font-family: 'Helvetica Neue', sans-serif;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
    }
    h1 {
        background: -webkit-linear-gradient(#eee, #999);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* 6. INPUT FIELDS */
    .stTextInput input, .stSelectbox div[data-baseweb="select"] {
        background-color: rgba(255, 255, 255, 0.1) !important;
        color: white !important;
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }

    /* --- QUAN TRỌNG: CỐ ĐỊNH CHIỀU CAO ẢNH ĐỂ GRID ĐỀU NHAU --- */
    div[data-testid="stImage"] img {
        height: 400px !important; /* Chiều cao cố định cho poster */
        object-fit: cover;        /* Cắt ảnh vừa khung mà không bị méo */
        border-radius: 10px;
    }

    /* Ẩn Decoration mặc định của Streamlit */
    header {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)


# Gọi hàm CSS ngay đầu
inject_custom_css()

# ==============================================================================
# 1. CẤU HÌNH BIẾN TOÀN CỤC
# ==============================================================================

USER_DATA_FILE = "danh_sach_nguoi_dung_moi.csv"
MOVIE_DATA_FILE = "data_phim_full_images.csv"  # <--- ĐÃ CẬP NHẬT FILE MỚI
GUEST_USER = "Guest_ZeroClick"

if 'logged_in_user' not in st.session_state: st.session_state['logged_in_user'] = None
if 'auth_mode' not in st.session_state: st.session_state['auth_mode'] = 'login'
if 'last_profile_recommendations' not in st.session_state: st.session_state[
    'last_profile_recommendations'] = pd.DataFrame()
if 'show_profile_plot' not in st.session_state: st.session_state['
