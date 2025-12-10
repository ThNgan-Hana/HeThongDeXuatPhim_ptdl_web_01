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
# CẬP NHẬT TÊN FILE DỮ LIỆU CỦA BẠN TẠI ĐÂY
MOVIE_DATA_FILE = "data_phim_full_images.csv"  
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
        # fillna("") để tránh lỗi NaN khi xử lý chuỗi
        return pd.read_csv(file_path).fillna("")
    except FileNotFoundError:
        # Nếu không tìm thấy file, trả về DataFrame rỗng để không bị crash app
        st.warning(f"⚠️ Chưa tìm thấy file '{file_path}'. Vui lòng tải file lên thư mục dự án.")
        return pd.DataFrame()


def parse_genres(genre_string):
    # Hàm an toàn để tách chuỗi thể loại
    if not isinstance(genre_string, str) or not genre_string: return set()
    genres = [g.strip().replace('"', '') for g in genre_string.split(',')]
    return set(genres)


@st.cache_resource
def load_and_preprocess_static_data():
    try:
        df_movies = load_data(MOVIE_DATA_FILE)
        if df_movies.empty: return pd.DataFrame(), np.array([[]]), []

        df_movies.columns = [col.strip() for col in df_movies.columns]

        # Kiểm tra cột (Dựa trên file data_phim_full_images.csv)
        required_cols = ['Tên phim', 'Đạo diễn', 'Diễn viên chính', 'Thể loại phim']
        if not all(col in df_movies.columns for col in required_cols):
             st.error(f"File dữ liệu thiếu các cột quan trọng. Cần có: {required_cols}")
             return pd.DataFrame(), np.array([[]]), []

        # Content-Based Features
        # .astype(str) giúp tránh lỗi nếu dữ liệu có số hoặc NaN
        df_movies["combined_features"] = (
                df_movies["Đạo diễn"].astype(str) + " " +
                df_movies["Diễn viên chính"].astype(str) + " " +
                df_movies["Thể loại phim"].astype(str)
        )
        
        # XỬ LÝ NGÔN NGỮ (TF-IDF)
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
            df_movies["popularity_norm"] = 0.5 # Giá trị mặc định

        # Genre & Recency
        df_movies['parsed_genres'] = df_movies['Thể loại phim'].apply(parse_genres)
        
        if 'Năm phát hành' in df_movies.columns:
            # Xử lý năm (đôi khi là chuỗi)
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
                # Tạo DataFrame rỗng nếu file user chưa có
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
# 3. HELPER HIỂN THỊ (ĐÃ THÊM LOGIC ẢNH)
# ==============================================================================

def display_movie_grid(df_result, title="Kết quả gợi ý"):
    """Hàm hiển thị danh sách phim dạng lưới (Grid) đẹp mắt có ảnh Poster"""
    st.markdown(f"### {title}")

    # Chia lưới 3 cột (tùy chỉnh responsive)
    cols = st.columns(3)

    for index, (i, row) in enumerate(df_result.iterrows()):
        col = cols[index % 3]  # Xoay vòng qua 3 cột
        with col:
            # Container tạo khung card
            with st.container(border=True):
                
                # --- PHẦN XỬ LÝ ẢNH MỚI ---
                # Lấy link từ cột 'Link Poster'
                poster_url = row.get('Link Poster', '')
                
                # Kiểm tra link hợp lệ (phải là chuỗi và bắt đầu bằng http)
                has_image = False
                if isinstance(poster_url, str) and poster_url.startswith('http'):
                    try:
                        # use_container_width=True giúp ảnh co giãn full bề ngang cột
                        st.image(poster_url, use_container_width=True)
                        has_image = True
                    except:
                        pass # Nếu lỗi load ảnh thì xuống phần fallback

                if not has_image:
                    # Nếu không có ảnh, hiển thị placeholder
                    random_color = f"hsl({np.random.randint(0, 360)}, 60%, 25%)"
                    st.markdown(f"""
                    <div style="background-color: {random_color}; padding: 40px 15px; border-radius: 10px; text-align: center; margin-bottom: 10px;">
                        <div style="font-size: 40px;">🎬</div>
                        <p style="font-size: 12px; opacity: 0.7">No Image</p>
                    </div>
                    """, unsafe_allow_html=True)

                # Header card (Tên phim + Năm)
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
                
                # Chuẩn hóa điểm số để hiển thị cho đẹp
                # Giả sử score có thể rất lớn hoặc nhỏ, ta cap lại ở 10 để vẽ
                display_score = score
                if display_score > 10: 
                    # Nếu score quá lớn (ví dụ do popularity), scale lại
                    display_score = 10 
                
                # Vẽ thanh progress bar custom bằng HTML
                st.markdown(f"""
                <div style="display:flex; justify-content:space-between; align-items:center; font-size:0.9em; margin-top:5px;">
                    <span>🔥 Độ hợp:</span>
                    <span style="color:#4CAF50; font-weight:bold;">{score:.1f} pts</span>
                </div>
                """, unsafe_allow_html=True)
                st.progress(min(display_score / 10.0, 1.0)) 


# ==============================================================================
# 4. HỆ THỐNG XÁC THỰC
# ==============================================================================

def set_auth_mode(mode):
    st.session_state['auth_mode'] = mode
    st.session_state['last_profile_recommendations'] = pd.DataFrame()


def logout():
    st.session_state['logged_in_user'] = None
    st.session_state['auth_mode'] = 'login'
    st.rerun()


def register_new_user_form(df_movies, sorted_genres):
    st.markdown("<h2 style='text-align: center; color: #ff6b6b;'>📝 Đăng Ký Thành Viên</h2>", unsafe_allow_html=True)

    with st.container(border=True):
        with st.form("register_form_new"):
            col1, col2 = st.columns([1, 1])
            with col1:
                username = st.text_input("Tên đăng nhập (Duy nhất):", placeholder="VD: cine_fan_2025")
            with col2:
                movie_titles_list = get_unique_movie_titles(df_movies)
                favorite_movie = st.selectbox("⭐ Phim tâm đắc nhất (Tùy chọn):",
                                              options=["-- Bỏ qua --"] + movie_titles_list)

            st.write("---")
            st.markdown("### 🎯 Bạn thích thể loại nào?")
            st.caption("Chọn ít nhất **3 thể loại** để chúng tôi hiểu gu của bạn.")

            if hasattr(st, 'pills'):
                selected_genres = st.pills("", options=sorted_genres, selection_mode="multi")
            else:
                selected_genres = st.multiselect("", options=sorted_genres)

            st.write("")
            submitted = st.form_submit_button("✨ Đăng Ký Ngay", type="primary", use_container_width=True)

            if submitted:
                df_users = st.session_state['df_users']
                if not username: st.error("⚠️ Thiếu tên đăng nhập!"); return
                if not df_users.empty and username in df_users['Tên người dùng'].values: st.error(
                    "❌ Tên đã tồn tại!"); return
                if not selected_genres or len(selected_genres) < 3: st.warning(
                    "⚠️ Chọn ít nhất 3 thể loại nhé!"); return

                max_id = df_users['ID'].max() if not df_users.empty and pd.notna(df_users['ID'].max()) else 0
                new_user_data = {
                    'ID': [int(max_id) + 1], 'Tên người dùng': [username],
                    '5 phim coi gần nhất': ["[]"],
                    'Phim yêu thích nhất': [favorite_movie if favorite_movie != "-- Bỏ qua --" else ""],
                    'Thể loại yêu thích': [", ".join(selected_genres)]
                }
                st.session_state['df_users'] = pd.concat([df_users, pd.DataFrame(new_user_data)], ignore_index=True)
                st.session_state['logged_in_user'] = username
                st.success(f"🎉 Chào mừng {username}!");
                st.rerun()


def login_form():
    st.markdown("<h2 style='text-align: center; color: #4facfe;'>🔑 Đăng Nhập</h2>", unsafe_allow_html=True)
    with st.container(border=True):
        with st.form("login_form"):
            username = st.text_input("Tên người dùng:")
            submitted = st.form_submit_button("Truy cập hệ thống", use_container_width=True)
            if submitted:
                df_users = st.session_state['df_users']
                if not df_users.empty and username in df_users['Tên người dùng'].values:
                    st.session_state['logged_in_user'] = username
                    st.success("✅ Thành công!");
                    st.rerun()
                else:
                    st.error("❌ Không tìm thấy user này.")


def authentication_page(df_movies, sorted_genres):
    # Header lớn
    st.markdown("""
    <div style='text-align: center; padding: 50px 0;'>
        <h1 style='font-size: 3rem;'>🍿 CINEMATCH 🎬</h1>
        <p style='font-size: 1.2rem; opacity: 0.8;'>Khám phá thế giới điện ảnh dành riêng cho bạn</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        # Tab chuyển đổi Login/Register cho mượt
        tab1, tab2 = st.tabs(["Đăng Nhập", "Đăng Ký Mới"])

        with tab1:
            login_form()
            st.write("")
            if st.button("🚀 Chế độ Khách (Không cần tài khoản)", use_container_width=True):
                st.session_state['logged_in_user'] = GUEST_USER;
                st.rerun()

        with tab2:
            register_new_user_form(df_movies, sorted_genres)


# ==============================================================================
# 5. LOGIC ĐỀ XUẤT
# ==============================================================================

def get_recommendations_weighted_genres(selected_genres, df_movies, num_recommendations=12):
    # Tạo pattern regex để tìm các thể loại
    pattern = '|'.join([re.escape(g) for g in selected_genres])
    # Chuyển về chuỗi để đảm bảo không lỗi
    filtered_df = df_movies[df_movies['Thể loại phim'].astype(str).str.contains(pattern, case=False, na=False)].copy()
    if filtered_df.empty: return pd.DataFrame()

    def calculate_score(row):
        score = row['popularity_norm'] * 2.0
        row_genres = [g.strip() for g in str(row['Thể loại phim']).split(',')]
        match_count = sum(1 for g in selected_genres if g in row_genres)
        score += match_count * 1.5
        score += row['recency_score'] * 1.0
        return score

    filtered_df['final_score'] = filtered_df.apply(calculate_score, axis=1)
    return filtered_df.sort_values(by='final_score', ascending=False).head(num_recommendations)


def get_recommendations(username, df_movies, num_recommendations=12):
    df_users = st.session_state['df_users']
    user_row = df_users[df_users['Tên người dùng'] == username]
    if user_row.empty: return pd.DataFrame()

    watched_str = user_row['5 phim coi gần nhất'].iloc[0]
    favorite_movie = user_row['Phim yêu thích nhất'].iloc[0]
    fav_genres_str = str(user_row.get('Thể loại yêu thích', pd.Series([""])).iloc[0])

    watched_list = []
    try:
        watched_list = ast.literal_eval(watched_str)
        if not isinstance(watched_list, list): watched_list = []
    except:
        watched_list = [m.strip().strip("'") for m in str(watched_str).strip('[]').split(',') if m.strip()]

    # 1. User Cũ (Content-based)
    if len(watched_list) > 0:
        watched_and_favorite = set(watched_list + [favorite_movie])
        watched_genres = df_movies[df_movies['Tên phim'].isin(watched_list)]
        user_genres_set = set()
        for genres in watched_genres['parsed_genres']: user_genres_set.update(genres)
        if not user_genres_set: return pd.DataFrame()

        candidate_movies = df_movies[~df_movies['Tên phim'].isin(watched_and_favorite)].copy()
        candidate_movies['Similarity_Score'] = candidate_movies['parsed_genres'].apply(
            lambda x: len(x.intersection(user_genres_set)))
        return candidate_movies.sort_values(by=['Similarity_Score', 'Độ phổ biến'], ascending=[False, False]).head(
            num_recommendations)

    # 2. User Mới (Weighted)
    elif fav_genres_str and fav_genres_str.strip():
        selected_genres = [g.strip() for g in fav_genres_str.split(',') if g.strip()]
        return get_recommendations_weighted_genres(selected_genres, df_movies, num_recommendations)
    else:
        return pd.DataFrame()


def recommend_movies_smart(movie_name, weight_sim, weight_pop, df_movies, cosine_sim):
    try:
        # Tìm index phim (chuyển về lowercase để so sánh cho chính xác)
        mask = df_movies['Tên phim'].astype(str).str.lower() == movie_name.lower()
        if not mask.any(): return pd.DataFrame()
        
        idx = df_movies[mask].index[0]
    except IndexError:
        return pd.DataFrame()

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores_df = pd.DataFrame(sim_scores, columns=['index', 'similarity'])
    df_result = pd.merge(df_movies, sim_scores_df, left_index=True, right_on='index')
    df_result['weighted_score'] = (weight_sim * df_result['similarity'] + weight_pop * df_result['popularity_norm'])
    df_result = df_result.drop(df_result[df_result['Tên phim'] == movie_name].index)
    return df_result.sort_values(by='weighted_score', ascending=False).head(12)  # Tăng lên 12 cho đẹp grid


def plot_genre_popularity(recommended_movies_df):
    if recommended_movies_df.empty: return
    genres_data = []
    for index, row in recommended_movies_df.iterrows():
        genres_list = [g.strip() for g in str(row['Thể loại phim']).split(',') if g.strip()]
        for genre in genres_list: genres_data.append({'Thể loại': genre, 'Độ phổ biến': row['Độ phổ biến']})
    df_plot = pd.DataFrame(genres_data)
    if df_plot.empty: return

    genre_avg_pop = df_plot.groupby('Thể loại')['Độ phổ biến'].mean().reset_index()
    top_7_genres = genre_avg_pop.sort_values(by='Độ phổ biến', ascending=False).head(7)

    # Custom Dark Plot
    with plt.style.context('dark_background'):
        fig, ax = plt.subplots(figsize=(8, 4))
        bars = ax.bar(top_7_genres['Thể loại'], top_7_genres['Độ phổ biến'], color='#ff6b6b', alpha=0.8)
        ax.set_title(f"Xu hướng thể loại", fontsize=12, color='white')
        ax.set_facecolor('#1e1e2f')  # Màu nền plot khớp background
        fig.patch.set_facecolor('#1e1e2f')
        plt.xticks(rotation=45, ha='right')
        st.pyplot(fig)


# ==============================================================================
# 6. GIAO DIỆN CHÍNH (MAIN APP)
# ==============================================================================

def main_page(df_movies, cosine_sim, sorted_genres):
    is_guest = st.session_state['logged_in_user'] == GUEST_USER
    username = st.session_state['logged_in_user']

    # Sidebar đẹp hơn
    with st.sidebar:
        st.markdown(f"### 👋 Hi, {username}")
        if is_guest:
            if st.button("🚪 Đăng Xuất Khách"): logout()
        else:
            menu_choice = st.radio("Mục lục:", ('🏠 Trang Chủ (Gợi ý)', '🔍 Tìm kiếm', '🚪 Đăng Xuất'))
            if menu_choice == '🚪 Đăng Xuất': logout()

    # --- A. GIAO DIỆN KHÁCH ---
    if is_guest:
        st.markdown("## 👀 Chế độ Khách")
        st.info("💡 Bạn đang ở chế độ ẩn danh. Hãy chọn thể loại để tìm phim nhanh!")

        st.markdown("### 🎯 Bạn muốn xem thể loại gì?")
        if hasattr(st, 'pills'):
            selected_guest_genres = st.pills("", options=sorted_genres, selection_mode="multi", key="guest_pills")
        else:
            selected_guest_genres = st.multiselect("Chọn thể loại:", options=sorted_genres)

        st.write("---")

        if selected_guest_genres:
            recs = get_recommendations_weighted_genres(selected_guest_genres, df_movies, 12)
            if not recs.empty:
                display_movie_grid(recs, title=f"Top Phim {', '.join(selected_guest_genres)}")
            else:
                st.warning("Không tìm thấy phim nào phù hợp.")
        else:
            # Sắp xếp mặc định
            df_guest = df_movies.sort_values(by=['year_numeric', 'popularity_norm'], ascending=[False, False]).head(12)
            display_movie_grid(df_guest, title="🔥 Top Thịnh Hành Toàn Cầu")
        return

    # --- B. GIAO DIỆN USER ---
    if menu_choice == '🏠 Trang Chủ (Gợi ý)':
        st.markdown(f"## ✨ Gợi ý dành riêng cho **{username}**")

        df_users = st.session_state['df_users']
        user_info = df_users[df_users['Tên người dùng'] == username].iloc[0]
        has_watched = len(user_info['5 phim coi gần nhất']) > 5
        has_genres = len(str(user_info['Thể loại yêu thích'])) > 0

        if has_genres and not has_watched:
            st.caption(f"🎯 Dựa trên gu thể loại: {user_info['Thể loại yêu thích']}")
        elif has_watched:
            st.caption("🎯 Dựa trên lịch sử xem phim.")

        col_btn, _ = st.columns([1, 4])
        if col_btn.button("🔄 Cập nhật Gợi ý", type="primary"):
            recs = get_recommendations(username, df_movies, 12)  # Lấy 12 phim cho chẵn cột
            st.session_state['last_profile_recommendations'] = recs
            st.session_state['show_profile_plot'] = True

        recs = st.session_state['last_profile_recommendations']
        if not recs.empty:
            if st.session_state['show_profile_plot']:
                with st.expander("📊 Xem phân tích gu của bạn"):
                    plot_genre_popularity(recs)

            display_movie_grid(recs, title="Phim hợp gu nhất")
        else:
            st.info("👋 Nhấn nút 'Cập nhật Gợi ý' để bắt đầu nhé!")

    elif menu_choice == '🔍 Tìm kiếm':
        st.markdown("## 🔎 Tìm phim tương tự")
        col_search, col_act = st.columns([3, 1])
        with col_search:
            movie_titles = get_unique_movie_titles(df_movies)
            selected_movie = st.selectbox("Chọn phim gốc:", movie_titles, label_visibility="collapsed")
        with col_act:
            if st.button("Tìm kiếm", use_container_width=True):
                res = recommend_movies_smart(selected_movie, 0.7, 0.3, df_movies, cosine_sim)
                if not res.empty:
                    display_movie_grid(res, title=f"Tương tự '{selected_movie}'")
                else:
                    st.warning("Không tìm thấy kết quả.")


# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == '__main__':
    df_movies, cosine_sim, sorted_genres = load_and_preprocess_static_data()
    initialize_user_data()

    if st.session_state['logged_in_user']:
        main_page(df_movies, cosine_sim, sorted_genres)
    else:
        authentication_page(df_movies, sorted_genres)
