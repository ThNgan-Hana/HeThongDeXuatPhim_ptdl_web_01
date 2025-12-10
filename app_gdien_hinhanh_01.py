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
    .movie-card-container:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 20px rgba(0, 0, 0, 0.5);
        border: 1px solid #E50914; /* Viền đỏ Netflix khi hover */
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

    /* Chỉnh ảnh trong card cho đều nhau */
    div[data-testid="stImage"] img {
        border-radius: 10px;
        object-fit: cover;
        width: 100%;
        height: 350px !important; /* Cố định chiều cao ảnh để grid đều */
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
MOVIE_DATA_FILE = "data_phim_full_images.csv"  # Đã cập nhật file mới
GUEST_USER = "Guest_ZeroClick"

if 'logged_in_user' not in st.session_state: st.session_state['logged_in_user'] = None
if 'auth_mode' not in st.session_state: st.session_state['auth_mode'] = 'login'
if 'last_profile_recommendations' not in st.session_state: st.session_state[
    'last_profile_recommendations'] = pd.DataFrame()
if 'show_profile_plot' not in st.session_state: st.session_state['show_profile_plot'] = False


# ==============================================================================
# 2. HÀM XỬ LÝ DỮ LIỆU
# ==============================================================================

@st.cache_data
def load_data(file_path):
    try:
        return pd.read_csv(file_path).fillna("")
    except FileNotFoundError:
        st.error(f"⚠️ LỖI: Không tìm thấy file '{file_path}'. Hãy đảm bảo bạn đã upload file này.")
        return pd.DataFrame()


def parse_genres(genre_string):
    if not isinstance(genre_string, str) or not genre_string: return set()
    genres = [g.strip().replace('"', '') for g in genre_string.split(',')]
    return set(genres)


@st.cache_resource
def load_and_preprocess_static_data():
    try:
        df_movies = load_data(MOVIE_DATA_FILE)
        if df_movies.empty: return pd.DataFrame(), np.array([[]]), []

        df_movies.columns = [col.strip() for col in df_movies.columns]

        # Kiểm tra các cột bắt buộc
        required_columns = ["Đạo diễn", "Diễn viên chính", "Thể loại phim", "Tên phim"]
        missing_cols = [col for col in required_columns if col not in df_movies.columns]
        if missing_cols:
            st.error(f"Dữ liệu thiếu các cột quan trọng: {missing_cols}")
            return pd.DataFrame(), np.array([[]]), []

        # Content-Based Features
        # Xử lý fillna cho chắc chắn chuỗi
        df_movies["combined_features"] = (
                df_movies["Đạo diễn"].astype(str) + " " +
                df_movies["Diễn viên chính"].astype(str) + " " +
                df_movies["Thể loại phim"].astype(str)
        )

        # XỬ LÝ NGÔN NGỮ
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(df_movies["combined_features"])
        cosine_sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

        # Popularity Normalization
        if 'Độ phổ biến' in df_movies.columns:
            df_movies['Độ phổ biến'] = pd.to_numeric(df_movies['Độ phổ biến'], errors='coerce')
            mean_popularity = df_movies['Độ phổ biến'].mean() if not df_movies['Độ phổ biến'].empty else 0
            df_movies['Độ phổ biến'] = df_movies['Độ phổ biến'].fillna(mean_popularity)
            scaler = MinMaxScaler()
            df_movies["popularity_norm"] = scaler.fit_transform(df_movies[["Độ phổ biến"]])
        else:
            df_movies["popularity_norm"] = 0.5  # Default

        # Genre & Recency
        df_movies['parsed_genres'] = df_movies['Thể loại phim'].apply(parse_genres)
        if 'Năm phát hành' in df_movies.columns:
            # Xử lý năm có thể lẫn ký tự
            df_movies['year_numeric'] = pd.to_numeric(df_movies['Năm phát hành'], errors='coerce').fillna(0).astype(int)
            current_year = 2025
            df_movies['recency_score'] = df_movies['year_numeric'].apply(
                lambda x: 1.0 if x >= current_year - 1 else (0.8 if x >= current_year - 5 else 0.5))
        else:
            df_movies['year_numeric'] = 0
            df_movies['recency_score'] = 0.5

        all_genres = set()
        for genres_str in df_movies['Thể loại phim']:
            if genres_str:
                parts = [g.strip() for g in str(genres_str).split(',')]
                all_genres.update(parts)
        sorted_genres = sorted(list(all_genres))

        return df_movies, cosine_sim_matrix, sorted_genres
    except Exception as e:
        st.error(f"LỖI XỬ LÝ DATA: {e}")
        return pd.DataFrame(), np.array([[]]), []


def initialize_user_data():
    if 'df_users' not in st.session_state:
        try:
            df_users = load_data(USER_DATA_FILE)
            if not df_users.empty:
                df_users.columns = [col.strip() for col in df_users.columns]
                if 'ID' in df_users.columns:
                    df_users['ID'] = pd.to_numeric(df_users['ID'], errors='coerce')
                    df_users = df_users.dropna(subset=['ID'])
                if 'Thể loại yêu thích' not in df_users.columns: df_users['Thể loại yêu thích'] = ""
            else:
                df_users = pd.DataFrame(columns=['ID', 'Tên người dùng', '5 phim coi gần nhất', 'Phim yêu thích nhất',
                                                 'Thể loại yêu thích'])
        except Exception:
            df_users = pd.DataFrame(
                columns=['ID', 'Tên người dùng', '5 phim coi gần nhất', 'Phim yêu thích nhất', 'Thể loại yêu thích'])
        st.session_state['df_users'] = df_users
    return st.session_state['df_users']


def get_unique_movie_titles(df_movies):
    if 'Tên phim' in df_movies.columns: return df_movies['Tên phim'].dropna().unique().tolist()
    return []


# ==============================================================================
# 3. HELPER HIỂN THỊ (QUAN TRỌNG: RENDER MOVIE CARD VỚI ẢNH)
# ==============================================================================

def display_movie_grid(df_result, title="Kết quả gợi ý"):
    """Hàm hiển thị danh sách phim dạng lưới (Grid) đẹp mắt với ảnh Poster"""
    st.markdown(f"### {title}")

    # Chia lưới 3 cột (tùy chỉnh responsive)
    cols = st.columns(3)

    for index, (i, row) in enumerate(df_result.iterrows()):
        col = cols[index % 3]  # Xoay vòng qua 3 cột
        with col:
            # Container tạo khung card
            with st.container(border=True):
                # --- PHẦN HIỂN THỊ ẢNH ---
                poster_url = row.get('Link Poster', '')

                # Kiểm tra link ảnh có hợp lệ không
                has_image = False
                if isinstance(poster_url, str) and poster_url.startswith('http'):
                    try:
                        st.image(poster_url, use_container_width=True)
                        has_image = True
                    except:
                        pass  # Nếu lỗi load ảnh thì fallback xuống dưới

                if not has_image:
                    # Nếu không có ảnh hoặc lỗi, hiển thị avatar màu
                    random_color = f"hsl({np.random.randint(0, 360)}, 60%, 25%)"
                    st.markdown(f"""
                    <div style="background-color: {random_color}; padding: 40px; border-radius: 10px; text-align: center; margin-bottom: 10px;">
                        <div style="font-size: 50px;">🎬</div>
                    </div>
                    """, unsafe_allow_html=True)

                # Tên phim
                st.markdown(f"#### {row['Tên phim']}")
                st.caption(f"📅 Năm: **{row.get('Năm phát hành', 'N/A')}**")

                # Thể loại dạng Tags
                genres_str = str(row.get('Thể loại phim', ''))
                genres = [g.strip() for g in genres_str.split(',')]
                genre_html = "".join([
                    f"<span style='background:rgba(255,255,255,0.1); padding:2px 8px; border-radius:12px; font-size:0.8em; margin-right:5px;'>{g}</span>"
                    for g in genres[:3]])
                st.markdown(f"<div style='margin-bottom:10px;'>{genre_html}</div>", unsafe_allow_html=True)

                # Điểm số
                score = row.get('final_score', row.get('Similarity_Score', row.get('weighted_score', 0)))
                # Normalize score để hiển thị progress bar (giả sử max 10 hoặc max theo logic)
                display_score = score
                if display_score > 10: display_score = 10  # Cap visual

                st.markdown(f"""
                <div style="display:flex; justify-content:space-between; align-items:center; font-size:0.9em; margin-top:5px;">
                    <span>