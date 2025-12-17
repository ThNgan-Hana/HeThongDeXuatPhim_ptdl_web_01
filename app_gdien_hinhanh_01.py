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
# 0. CẤU HÌNH TRANG & CSS (TỐI ƯU KHÔNG GIAN)
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
    /* 1. ĐẨY NỘI DUNG LÊN SÁT TRÊN CÙNG (GIẢM LỀ) */
    .block-container {
        padding-top: 1rem !important; /* Quan trọng: Giảm khoảng trống trên cùng */
        padding-bottom: 2rem !important;
    }

    /* 2. NỀN ỨNG DỤNG */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        color: #ffffff;
    }

    /* 3. THANH TÌM KIẾM NHỎ GỌN (COMPACT SEARCH) */
    div[data-testid="stForm"] {
        border: none;
        padding: 0;
    }
    
    /* 4. TĂNG KÍCH THƯỚC ẢNH POSTER (QUAN TRỌNG) */
    div[data-testid="stImage"] img {
        height: 380px !important; /* Tăng chiều cao để chiếm không gian đẹp hơn */
        object-fit: cover;        
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }

    /* 5. CARD PHIM */
    .movie-card-container {
        padding: 5px;
    }
    
    /* Tên phim gọn gàng */
    h4 {
        font-family: 'Helvetica Neue', sans-serif;
        font-size: 1rem !important;
        margin-bottom: 0px;
        white-space: nowrap; 
        overflow: hidden;
        text-overflow: ellipsis;
    }
    
    /* Chữ nhỏ (năm, thể loại) */
    .small-text {
        font-size: 0.8rem;
        opacity: 0.8;
    }

    /* 6. INPUT FIELDS (Tối giản) */
    .stSelectbox div[data-baseweb="select"] {
        background-color: rgba(255, 255, 255, 0.1) !important;
        color: white !important;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }

    /* Ẩn header mặc định */
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
# 2. XỬ LÝ DỮ LIỆU (Giữ nguyên logic lõi)
# ==============================================================================

@st.cache_data
def load_data(file_path):
    try:
        return pd.read_csv(file_path).fillna("")
    except: return pd.DataFrame()

def parse_genres(genre_string):
    if not isinstance(genre_string, str) or not genre_string: return set()
    genres = [g.strip().replace('"', '') for g in genre_string.split(',')]
    return set(genres)

@st.cache_resource
def load_and_preprocess_static_data():
    try:
        df = load_data(MOVIE_DATA_FILE)
        if df.empty: return pd.DataFrame(), np.array([[]]), []
        df.columns = [col.strip() for col in df.columns]

        df["combined_features"] = (df["Đạo diễn"].astype(str) + " " + df["Diễn viên chính"].astype(str) + " " + df["Thể loại phim"].astype(str))
        tfidf_matrix = TfidfVectorizer(stop_words='english').fit_transform(df["combined_features"])
        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

        if 'Độ phổ biến' in df.columns:
            df['Độ phổ biến'] = pd.to_numeric(df['Độ phổ biến'], errors='coerce').fillna(0)
            df["popularity_norm"] = MinMaxScaler().fit_transform(df[["Độ phổ biến"]])
        else: df["popularity_norm"] = 0.5

        df['parsed_genres'] = df['Thể loại phim'].apply(parse_genres)
        df['year_numeric'] = pd.to_numeric(df.get('Năm phát hành', 0), errors='coerce').fillna(0).astype(int)
        df['recency_score'] = df['year_numeric'].apply(lambda x: 1.0 if x >= 2024 else (0.8 if x >= 2020 else 0.5))

        all_genres = set()
        for g in df['Thể loại phim']:
            if g: all_genres.update([x.strip() for x in str(g).split(',')])
        return df, cosine_sim, sorted(list(all_genres))
    except: return pd.DataFrame(), np.array([[]]), []

def initialize_user_data():
    if 'df_users' not in st.session_state:
        st.session_state['df_users'] = load_data(USER_DATA_FILE)
        if st.session_state['df_users'].empty:
            st.session_state['df_users'] = pd.DataFrame(columns=['ID', 'Tên người dùng', '5 phim coi gần nhất', 'Phim yêu thích nhất', 'Thể loại yêu thích'])
    return st.session_state['df_users']

def get_unique_movie_titles(df):
    return df['Tên phim'].dropna().unique().tolist() if 'Tên phim' in df.columns else []

# ==============================================================================
# 3. HELPER HIỂN THỊ (GRID 5 CỘT - ẢNH TO)
# ==============================================================================

def display_movie_grid(df_result, title=None):
    if title: st.markdown(f"### {title}")
    
    if df_result.empty:
        st.info("Chưa có dữ liệu.")
        return

    # Grid 5 cột
    cols = st.columns(5)
    for index, (i, row) in enumerate(df_result.iterrows()):
        with cols[index % 5]:
            with st.container(border=True):
                # 1. Poster
                poster = row.get('Link Poster', '')
                if isinstance(poster, str) and poster.startswith('http'):
                    st.image(poster, use_container_width=True)
                else:
                    st.markdown(f"""<div style="background:#444;height:380px;display:flex;align-items:center;justify-content:center;border-radius:8px;">🎬</div>""", unsafe_allow_html=True)

                # 2. Info Compact
                st.markdown(f"#### {row['Tên phim']}")
                st.markdown(f"<div class='small-text'>📅 {row.get('Năm phát hành', 'N/A')}</div>", unsafe_allow_html=True)

                # 3. Score bar
                score = row.get('final_score', row.get('weighted_score', 0))
                st.progress(min(score / 10.0, 1.0))

# ==============================================================================
# 4. LOGIC ĐỀ XUẤT
# ==============================================================================

def get_recommendations_weighted_genres(selected_genres, df, num=10):
    pattern = '|'.join([re.escape(g) for g in selected_genres])
    filtered = df[df['Thể loại phim'].astype(str).str.contains(pattern, case=False, na=False)].copy()
    if filtered.empty: return pd.DataFrame()
    
    filtered['final_score'] = filtered.apply(lambda x: x['popularity_norm']*2 + sum(1 for g in selected_genres if g in str(x['Thể loại phim']))*1.5 + x['recency_score'], axis=1)
    return filtered.sort_values('final_score', ascending=False).head(num)

def recommend_movies_smart(movie_name, df, cosine_sim):
    try:
        idx = df[df['Tên phim'].astype(str).str.lower() == movie_name.lower()].index[0]
        sim_scores = list(enumerate(cosine_sim[idx]))
        res = pd.merge(df, pd.DataFrame(sim_scores, columns=['index', 'similarity']), left_index=True, right_on='index')
        res['weighted_score'] = res['similarity']*0.7 + res['popularity_norm']*0.3
        return res.drop(idx).sort_values('weighted_score', ascending=False).head(10)
    except: return pd.DataFrame()

# ==============================================================================
# 5. GIAO DIỆN CHÍNH (MAIN PAGE)
# ==============================================================================

def main_page(df_movies, cosine_sim, sorted_genres):
    username = st.session_state['logged_in_user']
    is_guest = username == GUEST_USER
    
    # Sidebar tối giản
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
            display_movie_grid(df_movies.sort_values(['year_numeric', 'popularity_norm'], ascending=False).head(10), "🔥 Phim Mới & Hot")
        return

    # --- TÌM KIẾM (ĐÃ TỐI ƯU DIỆN TÍCH) ---
    if menu == 'Tìm kiếm':
        # THANH TÌM KIẾM SLIM (Chiếm ít diện tích nhất có thể)
        c1, c2 = st.columns([6, 1], vertical_alignment="bottom") 
        with c1:
            selected_movie = st.selectbox("Chọn phim", get_unique_movie_titles(df_movies), label_visibility="collapsed", placeholder="Nhập tên phim...")
        with c2:
            search_btn = st.button("🔍 Tìm", use_container_width=True, type="primary")

        st.markdown("---") # Đường kẻ mờ ngăn cách
        
        if search_btn:
            res = recommend_movies_smart(selected_movie, df_movies, cosine_sim)
            display_movie_grid(res, f"Kết quả tương tự: {selected_movie}")
        else:
            # Hiển thị mặc định phim hot để không trống màn hình
            display_movie_grid(df_movies.sort_values('popularity_norm', ascending=False).head(10), "🎬 Phim đề xuất hôm nay")

    # --- TRANG CHỦ ---
    elif menu == 'Trang Chủ':
        # Nút cập nhật nhỏ gọn, float phải
        c_title, c_btn = st.columns([6, 1])
        with c_title: st.markdown("### ✨ Gợi ý hôm nay")
        with c_btn: 
            if st.button("🔄 Làm mới", use_container_width=True):
                # Giả lập logic lấy user (ở đây dùng random cho demo nếu chưa có history)
                st.session_state['last_profile_recommendations'] = df_movies.sample(10) 
        
        recs = st.session_state.get('last_profile_recommendations', pd.DataFrame())
        if recs.empty: recs = df_movies.sort_values('popularity_norm', ascending=False).head(10)
        
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
