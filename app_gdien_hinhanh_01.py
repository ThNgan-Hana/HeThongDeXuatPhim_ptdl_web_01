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
# 0. CẤU HÌNH TRANG & CSS (GIAO DIỆN FULL MÀN HÌNH)
# ==============================================================================
st.set_page_config(
    page_title="Cinematch - Gợi ý phim",
    page_icon="🍿",
    layout="wide", # Quan trọng: Chế độ rộng toàn màn hình
    initial_sidebar_state="expanded"
)

def inject_custom_css():
    st.markdown("""
    <style>
    /* 1. NỀN CHUYỂN MÀU */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        color: #ffffff;
    }

    /* 2. CARD PHIM (GỌN HƠN) */
    .movie-card-container {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        padding: 10px; /* Giảm padding để ảnh to hơn */
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        height: 100%;
    }

    /* 3. NÚT BẤM */
    .stButton > button {
        background: linear-gradient(90deg, #E50914 0%, #ff6b6b 100%);
        color: white;
        border: none;
        border-radius: 20px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0 0 15px rgba(229, 9, 20, 0.6);
    }

    /* 4. TIÊU ĐỀ */
    h1, h2, h3, h4 {
        font-family: 'Helvetica Neue', sans-serif;
    }
    h1 {
        background: -webkit-linear-gradient(#eee, #999);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Tên phim trong Card */
    h4 {
        font-size: 1.1rem !important;
        white-space: nowrap; 
        overflow: hidden;
        text-overflow: ellipsis; /* Cắt tên phim dài quá */
        margin-top: 10px;
    }

    /* 5. INPUT FIELDS */
    .stTextInput input, .stSelectbox div[data-baseweb="select"] {
        background-color: rgba(255, 255, 255, 0.1) !important;
        color: white !important;
        border-radius: 8px;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }

    /* --- QUAN TRỌNG: CỐ ĐỊNH CHIỀU CAO ẢNH CHO GRID 5 CỘT --- */
    div[data-testid="stImage"] img {
        height: 320px !important; /* Chiều cao tối ưu cho 5 cột */
        object-fit: cover;        
        border-radius: 8px;
    }
    
    /* Ẩn Decoration */
    header {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Tối ưu khoảng cách giữa các cột */
    div[data-testid="column"] {
        padding: 0 5px;
    }
    </style>
    """, unsafe_allow_html=True)


inject_custom_css()

# ==============================================================================
# 1. CẤU HÌNH BIẾN TOÀN CỤC
# ==============================================================================

USER_DATA_FILE = "danh_sach_nguoi_dung_moi.csv"
MOVIE_DATA_FILE = "data_phim_full_images.csv"
GUEST_USER = "Guest_ZeroClick"

if 'logged_in_user' not in st.session_state: st.session_state['logged_in_user'] = None
if 'auth_mode' not in st.session_state: st.session_state['auth_mode'] = 'login'
if 'last_profile_recommendations' not in st.session_state: st.session_state['last_profile_recommendations'] = pd.DataFrame()
if 'show_profile_plot' not in st.session_state: st.session_state['show_profile_plot'] = False


# ==============================================================================
# 2. HÀM XỬ LÝ DỮ LIỆU
# ==============================================================================

@st.cache_data
def load_data(file_path):
    try:
        return pd.read_csv(file_path).fillna("")
    except FileNotFoundError:
        st.warning(f"⚠️ Chưa tìm thấy file '{file_path}'.")
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

        # Features & TF-IDF
        df_movies["combined_features"] = (
                df_movies["Đạo diễn"].astype(str) + " " +
                df_movies["Diễn viên chính"].astype(str) + " " +
                df_movies["Thể loại phim"].astype(str)
        )
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(df_movies["combined_features"])
        cosine_sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

        # Popularity
        if 'Độ phổ biến' in df_movies.columns:
            df_movies['Độ phổ biến'] = pd.to_numeric(df_movies['Độ phổ biến'], errors='coerce')
            mean_pop = df_movies['Độ phổ biến'].mean() if not df_movies['Độ phổ biến'].empty else 0
            df_movies['Độ phổ biến'] = df_movies['Độ phổ biến'].fillna(mean_pop)
            scaler = MinMaxScaler()
            df_movies["popularity_norm"] = scaler.fit_transform(df_movies[["Độ phổ biến"]])
        else:
            df_movies["popularity_norm"] = 0.5

        # Genres & Recency
        df_movies['parsed_genres'] = df_movies['Thể loại phim'].apply(parse_genres)
        if 'Năm phát hành' in df_movies.columns:
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
        st.error(f"LỖI DATA: {e}")
        return pd.DataFrame(), np.array([[]]), []

def initialize_user_data():
    if 'df_users' not in st.session_state:
        try:
            df_users = load_data(USER_DATA_FILE)
            if df_users.empty: raise Exception
            df_users.columns = [col.strip() for col in df_users.columns]
            if 'ID' in df_users.columns: df_users['ID'] = pd.to_numeric(df_users['ID'], errors='coerce')
            if 'Thể loại yêu thích' not in df_users.columns: df_users['Thể loại yêu thích'] = ""
        except:
            df_users = pd.DataFrame(columns=['ID', 'Tên người dùng', '5 phim coi gần nhất', 'Phim yêu thích nhất', 'Thể loại yêu thích'])
        st.session_state['df_users'] = df_users
    return st.session_state['df_users']

def get_unique_movie_titles(df_movies):
    if 'Tên phim' in df_movies.columns: return df_movies['Tên phim'].dropna().unique().tolist()
    return []

# ==============================================================================
# 3. HELPER HIỂN THỊ: GRID 5 CỘT (FULL WIDTH)
# ==============================================================================

def display_movie_grid(df_result, title="Kết quả gợi ý"):
    """Hiển thị phim dạng lưới 5 cột để tràn màn hình"""
    if df_result.empty:
        st.info("Chưa có dữ liệu phim.")
        return

    st.markdown(f"### {title}")
    
    # --- THAY ĐỔI Ở ĐÂY: SỬ DỤNG 5 CỘT ---
    num_columns = 5 
    cols = st.columns(num_columns)

    for index, (i, row) in enumerate(df_result.iterrows()):
        col = cols[index % num_columns]
        with col:
            with st.container(border=True):
                # 1. Ảnh Poster
                poster_url = row.get('Link Poster', '')
                has_image = False
                if isinstance(poster_url, str) and poster_url.startswith('http'):
                    try:
                        st.image(poster_url, use_container_width=True)
                        has_image = True
                    except: pass
                
                if not has_image:
                    random_color = f"hsl({np.random.randint(0, 360)}, 60%, 25%)"
                    st.markdown(f"""<div style="background:{random_color};padding:60px 0;border-radius:8px;text-align:center;margin-bottom:10px;"><div style="font-size:30px;">🎬</div></div>""", unsafe_allow_html=True)

                # 2. Thông tin phim
                # Tên phim cắt ngắn nếu quá dài
                st.markdown(f"#### {row['Tên phim']}")
                st.caption(f"📅 {row.get('Năm phát hành', 'N/A')}")

                # Thể loại (Chỉ lấy 2 cái đầu cho gọn grid 5 cột)
                genres_str = str(row.get('Thể loại phim', ''))
                genres = [g.strip() for g in genres_str.split(',')]
                genre_html = "".join([f"<span style='background:rgba(255,255,255,0.1);padding:2px 6px;border-radius:4px;font-size:0.7em;margin-right:4px;'>{g}</span>" for g in genres[:2]])
                st.markdown(f"<div style='margin-bottom:8px; height: 25px; overflow:hidden;'>{genre_html}</div>", unsafe_allow_html=True)

                # 3. Điểm số
                score = row.get('final_score', row.get('Similarity_Score', row.get('weighted_score', 0)))
                display_score = score if score <= 10 else 10
                
                st.progress(min(display_score / 10.0, 1.0))
                st.markdown(f"<div style='text-align:right;font-size:0.8em;color:#4CAF50;'>Match: {score:.1f}</div>", unsafe_allow_html=True)

# ==============================================================================
# 4. HỆ THỐNG XÁC THỰC
# ==============================================================================

def logout():
    st.session_state['logged_in_user'] = None
    st.rerun()

def register_new_user_form(df_movies, sorted_genres):
    st.markdown("<h3 style='text-align: center; color: #ff6b6b;'>📝 Đăng Ký</h3>", unsafe_allow_html=True)
    with st.container(border=True):
        with st.form("register_form_new"):
            username = st.text_input("Tên đăng nhập:")
            movie_titles = get_unique_movie_titles(df_movies)
            fav_movie = st.selectbox("Phim thích nhất:", ["-- Bỏ qua --"] + movie_titles)
            
            st.markdown("##### Thể loại yêu thích:")
            if hasattr(st, 'pills'):
                selected_genres = st.pills("", options=sorted_genres, selection_mode="multi")
            else:
                selected_genres = st.multiselect("", options=sorted_genres)

            if st.form_submit_button("Đăng Ký", type="primary", use_container_width=True):
                df_users = st.session_state['df_users']
                if not username: st.error("Thiếu tên đăng nhập!"); return
                if not df_users.empty and username in df_users['Tên người dùng'].values: st.error("Tên đã tồn tại!"); return
                if len(selected_genres) < 3: st.warning("Chọn ít nhất 3 thể loại!"); return

                max_id = df_users['ID'].max() if not df_users.empty and pd.notna(df_users['ID'].max()) else 0
                new_user = {
                    'ID': [int(max_id) + 1], 'Tên người dùng': [username],
                    '5 phim coi gần nhất': ["[]"],
                    'Phim yêu thích nhất': [fav_movie if fav_movie != "-- Bỏ qua --" else ""],
                    'Thể loại yêu thích': [", ".join(selected_genres)]
                }
                st.session_state['df_users'] = pd.concat([df_users, pd.DataFrame(new_user)], ignore_index=True)
                st.session_state['logged_in_user'] = username
                st.success("Đăng ký thành công!"); st.rerun()

def login_form():
    st.markdown("<h3 style='text-align: center; color: #4facfe;'>🔑 Đăng Nhập</h3>", unsafe_allow_html=True)
    with st.container(border=True):
        with st.form("login_form"):
            username = st.text_input("User:")
            if st.form_submit_button("Vào", use_container_width=True):
                df_users = st.session_state['df_users']
                if not df_users.empty and username in df_users['Tên người dùng'].values:
                    st.session_state['logged_in_user'] = username
                    st.success("OK!"); st.rerun()
                else: st.error("Sai user.")

def authentication_page(df_movies, sorted_genres):
    st.markdown("<h1 style='text-align:center;'>🍿 CINEMATCH</h1>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        t1, t2 = st.tabs(["Đăng Nhập", "Đăng Ký"])
        with t1:
            login_form()
            if st.button("Khách ghé thăm", use_container_width=True):
                st.session_state['logged_in_user'] = GUEST_USER; st.rerun()
        with t2: register_new_user_form(df_movies, sorted_genres)

# ==============================================================================
# 5. LOGIC ĐỀ XUẤT (SỐ LƯỢNG 10)
# ==============================================================================

def get_recommendations_weighted_genres(selected_genres, df_movies, num_recommendations=10):
    pattern = '|'.join([re.escape(g) for g in selected_genres])
    filtered = df_movies[df_movies['Thể loại phim'].astype(str).str.contains(pattern, case=False, na=False)].copy()
    if filtered.empty: return pd.DataFrame()

    def calc(row):
        score = row['popularity_norm'] * 2.0
        row_genres = [g.strip() for g in str(row['Thể loại phim']).split(',')]
        score += sum(1 for g in selected_genres if g in row_genres) * 1.5
        score += row['recency_score'] * 1.0
        return score

    filtered['final_score'] = filtered.apply(calc, axis=1)
    return filtered.sort_values(by='final_score', ascending=False).head(num_recommendations)

def get_recommendations(username, df_movies, num_recommendations=10):
    df_users = st.session_state['df_users']
    user_row = df_users[df_users['Tên người dùng'] == username]
    if user_row.empty: return pd.DataFrame()

    watched_str = user_row['5 phim coi gần nhất'].iloc[0]
    fav_movie = user_row['Phim yêu thích nhất'].iloc[0]
    fav_genres_str = str(user_row.get('Thể loại yêu thích', pd.Series([""])).iloc[0])

    watched_list = []
    try:
        watched_list = ast.literal_eval(watched_str)
        if not isinstance(watched_list, list): watched_list = []
    except:
        watched_list = [m.strip().strip("'") for m in str(watched_str).strip('[]').split(',') if m.strip()]

    if len(watched_list) > 0:
        watched_and_fav = set(watched_list + [fav_movie])
        watched_genres = df_movies[df_movies['Tên phim'].isin(watched_list)]
        user_genres_set = set()
        for genres in watched_genres['parsed_genres']: user_genres_set.update(genres)
        if not user_genres_set: return pd.DataFrame()

        candidates = df_movies[~df_movies['Tên phim'].isin(watched_and_fav)].copy()
        candidates['Similarity_Score'] = candidates['parsed_genres'].apply(lambda x: len(x.intersection(user_genres_set)))
        return candidates.sort_values(by=['Similarity_Score', 'Độ phổ biến'], ascending=[False, False]).head(num_recommendations)

    elif fav_genres_str and fav_genres_str.strip():
        selected_genres = [g.strip() for g in fav_genres_str.split(',') if g.strip()]
        return get_recommendations_weighted_genres(selected_genres, df_movies, num_recommendations)
    else: return pd.DataFrame()

def recommend_movies_smart(movie_name, df_movies, cosine_sim):
    try:
        mask = df_movies['Tên phim'].astype(str).str.lower() == movie_name.lower()
        if not mask.any(): return pd.DataFrame()
        idx = df_movies[mask].index[0]
    except: return pd.DataFrame()

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_df = pd.DataFrame(sim_scores, columns=['index', 'similarity'])
    res = pd.merge(df_movies, sim_df, left_index=True, right_on='index')
    res['weighted_score'] = (0.7 * res['similarity'] + 0.3 * res['popularity_norm'])
    res = res.drop(res[res['Tên phim'] == movie_name].index)
    # Trả về 10 phim
    return res.sort_values(by='weighted_score', ascending=False).head(10)

def plot_genre_popularity(df_recs):
    if df_recs.empty: return
    genres_data = []
    for _, row in df_recs.iterrows():
        for g in str(row['Thể loại phim']).split(','):
            if g.strip(): genres_data.append({'Type': g.strip(), 'Pop': row['Độ phổ biến']})
    
    df_plot = pd.DataFrame(genres_data)
    if df_plot.empty: return
    top_genres = df_plot.groupby('Type')['Pop'].mean().reset_index().sort_values(by='Pop', ascending=False).head(7)

    with plt.style.context('dark_background'):
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.bar(top_genres['Type'], top_genres['Pop'], color='#ff6b6b', alpha=0.8)
        ax.set_facecolor('#1e1e2f'); fig.patch.set_facecolor('#1e1e2f')
        plt.xticks(rotation=45, ha='right')
        st.pyplot(fig)

# ==============================================================================
# 6. MAIN PAGE
# ==============================================================================

def main_page(df_movies, cosine_sim, sorted_genres):
    username = st.session_state['logged_in_user']
    is_guest = username == GUEST_USER

    with st.sidebar:
        st.markdown(f"### 👋 {username}")
        menu = st.radio("", ('Trang Chủ', 'Tìm kiếm', 'Đăng Xuất'))
        if menu == 'Đăng Xuất' or (is_guest and st.button("Thoát")): logout()

    # --- MAIN CONTENT ---
    if is_guest:
        st.markdown("### 🎯 Khách: Chọn thể loại")
        if hasattr(st, 'pills'):
            sel_genres = st.pills("", options=sorted_genres, selection_mode="multi")
        else:
            sel_genres = st.multiselect("", options=sorted_genres)

        if sel_genres:
            recs = get_recommendations_weighted_genres(sel_genres, df_movies, 10)
            display_movie_grid(recs, f"Gợi ý: {', '.join(sel_genres)}")
        else:
            top_movies = df_movies.sort_values(by=['year_numeric', 'popularity_norm'], ascending=[False, False]).head(10)
            display_movie_grid(top_movies, "🔥 Phim Mới & Hot")
        return

    if menu == 'Trang Chủ':
        st.markdown(f"## 🎬 Gợi ý cho {username}")
        if st.button("🔄 Cập nhật phim mới", type="primary"):
            st.session_state['last_profile_recommendations'] = get_recommendations(username, df_movies, 10)
            st.session_state['show_profile_plot'] = True
        
        recs = st.session_state['last_profile_recommendations']
        if not recs.empty:
            if st.session_state['show_profile_plot']:
                with st.expander("📊 Biểu đồ sở thích"): plot_genre_popularity(recs)
            display_movie_grid(recs, "Top Phim Hợp Gu")
        else:
            st.info("Bấm nút trên để lấy gợi ý nhé!")

    elif menu == 'Tìm kiếm':
        st.markdown("## 🔍 Tìm phim tương tự")
        c1, c2 = st.columns([3, 1])
        with c1: selected_movie = st.selectbox("Chọn phim:", get_unique_movie_titles(df_movies))
        with c2: 
            if st.button("Tìm", use_container_width=True):
                res = recommend_movies_smart(selected_movie, df_movies, cosine_sim)
                display_movie_grid(res, f"Giống với '{selected_movie}'")

if __name__ == '__main__':
    df_movies, cosine_sim, sorted_genres = load_and_preprocess_static_data()
    initialize_user_data()
    if st.session_state['logged_in_user']:
        main_page(df_movies, cosine_sim, sorted_genres)
    else:
        authentication_page(df_movies, sorted_genres)
