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
# 0. CẤU HÌNH TRANG & CSS (MÀU SẮC TỰ ĐỘNG + BỐ CỤC ĐẸP)
# ==============================================================================
st.set_page_config(
    page_title="Cinematch",
    page_icon="🍿",
    layout="wide",
    initial_sidebar_state="expanded"
)

def inject_custom_css():
    st.markdown("""
    <style>
    /* Tối ưu khoảng cách lề */
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 3rem !important;
    }

    /* Card Phim: Tự động đổi màu theo giao diện Sáng/Tối */
    .movie-card-container {
        background-color: var(--secondary-background-color); 
        border-radius: 12px;
        padding: 8px;
        border: 1px solid rgba(128, 128, 128, 0.2); 
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        height: 100%;
        color: var(--text-color);
    }
    
    .movie-card-container:hover {
        transform: translateY(-3px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        border-color: #E50914;
    }

    /* Ảnh Poster */
    div[data-testid="stImage"] img {
        height: 380px !important;
        object-fit: cover;        
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    
    /* Nút bấm đỏ Netflix */
    .stButton > button {
        background-color: #E50914 !important;
        color: white !important;
        border: none;
        border-radius: 6px;
        font-weight: 600;
    }
    .stButton > button:hover {
        opacity: 0.8;
    }

    /* Typography */
    h4 {
        font-family: 'Source Sans Pro', sans-serif;
        font-size: 1rem !important;
        margin-top: 8px;
        margin-bottom: 4px;
        white-space: nowrap; 
        overflow: hidden;
        text-overflow: ellipsis;
        font-weight: 700;
        color: var(--text-color);
    }
    
    .small-text {
        font-size: 0.85rem;
        opacity: 0.7;
        color: var(--text-color);
    }

    /* Input Fields trong suốt */
    .stSelectbox div[data-baseweb="select"] > div {
        background-color: var(--secondary-background-color) !important;
        color: var(--text-color) !important;
        border-color: rgba(128, 128, 128, 0.3) !important;
    }
    
    header {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

inject_custom_css()

# ==============================================================================
# 1. CẤU HÌNH DỮ LIỆU
# ==============================================================================

USER_DATA_FILE = "danh_sach_nguoi_dung_moi.csv"
MOVIE_DATA_FILE = "data_phim_full_images.csv"
GUEST_USER = "Guest_ZeroClick"

if 'logged_in_user' not in st.session_state: st.session_state['logged_in_user'] = None
if 'auth_mode' not in st.session_state: st.session_state['auth_mode'] = 'login'
if 'last_profile_recommendations' not in st.session_state: st.session_state['last_profile_recommendations'] = pd.DataFrame()
if 'show_profile_plot' not in st.session_state: st.session_state['show_profile_plot'] = False

# ==============================================================================
# 2. XỬ LÝ DỮ LIỆU (ĐÃ THÊM KHỬ TRÙNG LẶP)
# ==============================================================================

@st.cache_data
def load_data(file_path):
    try: return pd.read_csv(file_path).fillna("")
    except: return pd.DataFrame()

def parse_genres(genre_string):
    if not isinstance(genre_string, str) or not genre_string: return set()
    return set([g.strip().replace('"', '') for g in genre_string.split(',')])

@st.cache_resource
def load_and_preprocess_static_data():
    try:
        df = load_data(MOVIE_DATA_FILE)
        if df.empty: return pd.DataFrame(), np.array([[]]), []
        df.columns = [col.strip() for col in df.columns]

        # --- FIX QUAN TRỌNG: LOẠI BỎ CÁC PHIM TRÙNG TÊN NGAY TỪ ĐẦU ---
        # Giữ lại phim đầu tiên tìm thấy, bỏ các phim trùng tên phía sau
        if 'Tên phim' in df.columns:
            df = df.drop_duplicates(subset=['Tên phim'], keep='first').reset_index(drop=True)

        # Tạo đặc trưng để so sánh
        df["combined_features"] = (df["Đạo diễn"].astype(str) + " " + df["Diễn viên chính"].astype(str) + " " + df["Thể loại phim"].astype(str))
        tfidf_matrix = TfidfVectorizer(stop_words='english').fit_transform(df["combined_features"])
        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

        # Chuẩn hóa độ phổ biến
        if 'Độ phổ biến' in df.columns:
            df['Độ phổ biến'] = pd.to_numeric(df['Độ phổ biến'], errors='coerce').fillna(0)
            df["popularity_norm"] = MinMaxScaler().fit_transform(df[["Độ phổ biến"]])
        else: df["popularity_norm"] = 0.5

        # Xử lý thể loại và năm
        df['parsed_genres'] = df['Thể loại phim'].apply(parse_genres)
        df['year_numeric'] = pd.to_numeric(df.get('Năm phát hành', 0), errors='coerce').fillna(0).astype(int)
        df['recency_score'] = df['year_numeric'].apply(lambda x: 1.0 if x >= 2024 else (0.8 if x >= 2020 else 0.5))

        all_genres = set()
        for g in df['Thể loại phim']:
            if g: all_genres.update([x.strip() for x in str(g).split(',')])
            
        return df, cosine_sim, sorted(list(all_genres))
    except Exception as e:
        st.error(f"Lỗi xử lý dữ liệu: {e}")
        return pd.DataFrame(), np.array([[]]), []

def initialize_user_data():
    if 'df_users' not in st.session_state:
        st.session_state['df_users'] = load_data(USER_DATA_FILE)
        if st.session_state['df_users'].empty:
            st.session_state['df_users'] = pd.DataFrame(columns=['ID', 'Tên người dùng', '5 phim coi gần nhất', 'Phim yêu thích nhất', 'Thể loại yêu thích'])
    return st.session_state['df_users']

def get_unique_movie_titles(df):
    return df['Tên phim'].dropna().unique().tolist() if 'Tên phim' in df.columns else []

# ==============================================================================
# 3. HELPER HIỂN THỊ (GRID 5 CỘT)
# ==============================================================================

def display_movie_grid(df_result, title=None):
    if title: st.markdown(f"### {title}")
    
    if df_result.empty:
        st.info("Chưa có dữ liệu.")
        return

    # --- LỌC TRÙNG LẶP LẦN CUỐI TRƯỚC KHI HIỂN THỊ ---
    # Đảm bảo danh sách hiển thị không có phim trùng nhau
    if 'Tên phim' in df_result.columns:
        df_result = df_result.drop_duplicates(subset=['Tên phim'], keep='first')

    cols = st.columns(5)
    for index, (i, row) in enumerate(df_result.iterrows()):
        with cols[index % 5]:
            with st.container():
                st.markdown('<div class="movie-card-container">', unsafe_allow_html=True)
                
                # Poster
                poster = row.get('Link Poster', '')
                if isinstance(poster, str) and poster.startswith('http'):
                    st.image(poster, use_container_width=True)
                else:
                    st.markdown(f"""<div style="background:rgba(128,128,128,0.2);height:380px;display:flex;align-items:center;justify-content:center;border-radius:8px;color:var(--text-color);">No Image</div>""", unsafe_allow_html=True)

                # Info
                st.markdown(f"#### {row['Tên phim']}")
                st.markdown(f"<div class='small-text'>📅 {row.get('Năm phát hành', 'N/A')}</div>", unsafe_allow_html=True)

                # Score
                score = row.get('final_score', row.get('weighted_score', 0))
                st.progress(min(score / 10.0, 1.0))
                
                st.markdown('</div>', unsafe_allow_html=True)

# ==============================================================================
# 4. LOGIC ĐỀ XUẤT
# ==============================================================================

def get_recommendations_weighted_genres(selected_genres, df, num=10):
    pattern = '|'.join([re.escape(g) for g in selected_genres])
    filtered = df[df['Thể loại phim'].astype(str).str.contains(pattern, case=False, na=False)].copy()
    if filtered.empty: return pd.DataFrame()
    
    filtered['final_score'] = filtered.apply(lambda x: x['popularity_norm']*2 + sum(1 for g in selected_genres if g in str(x['Thể loại phim']))*1.5 + x['recency_score'], axis=1)
    
    # Sort và lấy top, sau đó drop duplicates lần nữa để chắc chắn
    return filtered.sort_values('final_score', ascending=False).drop_duplicates(subset=['Tên phim']).head(num)

def recommend_movies_smart(movie_name, df, cosine_sim):
    try:
        # Tìm index phim chính xác
        idx = df[df['Tên phim'].astype(str).str.lower() == movie_name.lower()].index[0]
        
        sim_scores = list(enumerate(cosine_sim[idx]))
        res = pd.merge(df, pd.DataFrame(sim_scores, columns=['index', 'similarity']), left_index=True, right_on='index')
        res['weighted_score'] = res['similarity']*0.7 + res['popularity_norm']*0.3
        
        # Loại bỏ chính phim đang tìm kiếm và các phim trùng tên
        return res.drop(idx).sort_values('weighted_score', ascending=False).drop_duplicates(subset=['Tên phim']).head(10)
    except: return pd.DataFrame()

# ==============================================================================
# 5. GIAO DIỆN CHÍNH
# ==============================================================================

def main_page(df_movies, cosine_sim, sorted_genres):
    username = st.session_state['logged_in_user']
    is_guest = username == GUEST_USER
    
    with st.sidebar:
        st.markdown(f"**👤 {username}**")
        menu = st.radio("Menu", ['Trang Chủ', 'Tìm kiếm', 'Thoát'], label_visibility="collapsed")
        if menu == 'Thoát':
            st.session_state['logged_in_user'] = None
            st.rerun()

    # --- KHÁCH ---
    if is_guest:
        st.caption("🔍 Chế độ khách: Chọn thể loại bên dưới")
        if hasattr(st, 'pills'):
            genres = st.pills("", sorted_genres, selection_mode="multi")
        else:
            genres = st.multiselect("", sorted_genres)
        
        st.markdown("---")
        if genres:
            display_movie_grid(get_recommendations_weighted_genres(genres, df_movies), "Kết quả")
        else:
            # Drop duplicates khi hiển thị danh sách mặc định
            display_movie_grid(df_movies.sort_values(['year_numeric', 'popularity_norm'], ascending=False).drop_duplicates(subset=['Tên phim']).head(10), "🔥 Phim Mới & Hot")
        return

    # --- TÌM KIẾM ---
    if menu == 'Tìm kiếm':
        c1, c2 = st.columns([6, 1], vertical_alignment="bottom") 
        with c1:
            selected_movie = st.selectbox("Chọn phim", get_unique_movie_titles(df_movies), label_visibility="collapsed", placeholder="Nhập tên phim...")
        with c2:
            search_btn = st.button("🔍", use_container_width=True, type="primary")

        st.markdown("---")
        
        if search_btn:
            res = recommend_movies_smart(selected_movie, df_movies, cosine_sim)
            display_movie_grid(res, f"Kết quả tương tự: {selected_movie}")
        else:
            display_movie_grid(df_movies.sort_values('popularity_norm', ascending=False).drop_duplicates(subset=['Tên phim']).head(10), "🎬 Đề xuất hôm nay")

    # --- TRANG CHỦ ---
    elif menu == 'Trang Chủ':
        c_title, c_btn = st.columns([6, 1])
        with c_title: st.markdown("### ✨ Gợi ý hôm nay")
        with c_btn: 
            if st.button("🔄", use_container_width=True):
                # Random 10 phim unique
                st.session_state['last_profile_recommendations'] = df_movies.drop_duplicates(subset=['Tên phim']).sample(10)
        
        recs = st.session_state.get('last_profile_recommendations', pd.DataFrame())
        if recs.empty: recs = df_movies.sort_values('popularity_norm', ascending=False).drop_duplicates(subset=['Tên phim']).head(10)
        
        display_movie_grid(recs)

# ==============================================================================
# AUTH PAGE
# ==============================================================================
def authentication_page(df_movies, sorted_genres):
    st.markdown("<h1 style='text-align:center;font-size:3rem;'>🎬 CINEMATCH</h1>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        tab1, tab2 = st.tabs(["Đăng Nhập", "Đăng Ký"])
        with tab1:
            with st.form("login"):
                u = st.text_input("Username")
                if st.form_submit_button("Đăng Nhập", use_container_width=True):
                    users = st.session_state['df_users']
                    if not users.empty and u in users['Tên người dùng'].values:
                        st.session_state['logged_in_user'] = u; st.rerun()
                    else: st.error("Sai username")
            if st.button("Chế độ Khách", use_container_width=True):
                st.session_state['logged_in_user'] = GUEST_USER; st.rerun()
        
        with tab2:
            with st.form("reg"):
                new_u = st.text_input("Username mới")
                fav = st.selectbox("Phim thích", [""] + get_unique_movie_titles(df_movies))
                g = st.multiselect("Thể loại", sorted_genres)
                if st.form_submit_button("Đăng Ký", type="primary", use_container_width=True):
                    users = st.session_state['df_users']
                    if new_u and (users.empty or new_u not in users['Tên người dùng'].values):
                        row = {'ID': len(users)+1, 'Tên người dùng': new_u, '5 phim coi gần nhất': "[]", 'Phim yêu thích nhất': fav, 'Thể loại yêu thích': ",".join(g)}
                        st.session_state['df_users'] = pd.concat([users, pd.DataFrame([row])], ignore_index=True)
                        st.success("OK!"); st.rerun()
                    else: st.error("Lỗi đăng ký")

if __name__ == '__main__':
    df, sim, genres = load_and_preprocess_static_data()
    initialize_user_data()
    if st.session_state['logged_in_user']: main_page(df, sim, genres)
    else: authentication_page(df, genres)
