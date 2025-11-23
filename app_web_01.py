import streamlit as st
import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import re

# --- CẤU HÌNH TÊN FILE ---
USER_DATA_FILE = "danh_sach_nguoi_dung_moi.csv"
MOVIE_DATA_FILE = "movie_info_1000.csv"

# --- CONSTANT ---
GUEST_USER = "Guest_ZeroClick"  # Định danh cho người dùng chế độ Khách

# --- KHỞI TẠO BIẾN TRẠNG THÁI (SESSION STATE) ---
if 'logged_in_user' not in st.session_state:
    st.session_state['logged_in_user'] = None
if 'auth_mode' not in st.session_state:
    st.session_state['auth_mode'] = 'login'

# Biến lưu kết quả đề xuất
if 'last_sim_result' not in st.session_state: st.session_state['last_sim_result'] = pd.DataFrame()
if 'last_sim_movie' not in st.session_state: st.session_state['last_sim_movie'] = None
if 'show_sim_plot' not in st.session_state: st.session_state['show_sim_plot'] = False

if 'last_profile_recommendations' not in st.session_state: st.session_state[
    'last_profile_recommendations'] = pd.DataFrame()
if 'show_profile_plot' not in st.session_state: st.session_state['show_profile_plot'] = False
if 'last_guest_result' not in st.session_state: st.session_state['last_guest_result'] = pd.DataFrame()
if 'show_guest_plot' not in st.session_state: st.session_state['show_guest_plot'] = False


# ==============================================================================
# I. PHẦN TIỀN XỬ LÝ DỮ LIỆU & HELPERS
# ==============================================================================

@st.cache_data
def load_data(file_path):
    """Hàm helper để tải dữ liệu CSV với cache."""
    return pd.read_csv(file_path).fillna("")


def parse_genres(genre_string):
    """Chuyển chuỗi thể loại thành tập hợp genres."""
    if not isinstance(genre_string, str) or not genre_string:
        return set()
    # Tách theo dấu phẩy và làm sạch
    genres = [g.strip().replace('"', '') for g in genre_string.split(',')]
    return set(genres)


@st.cache_resource  # Chỉ tải dữ liệu tĩnh một lần
def load_and_preprocess_static_data():
    """Tải và tiền xử lý dữ liệu tĩnh (movies và mô hình)."""
    try:
        df_movies = load_data(MOVIE_DATA_FILE)
        df_movies.columns = [col.strip() for col in df_movies.columns]

        # 1. Tiền xử lý cho Content-Based (TF-IDF/Cosine Sim)
        df_movies["combined_features"] = (
                df_movies["Đạo diễn"] + " " +
                df_movies["Diễn viên chính"] + " " +
                df_movies["Thể loại phim"]
        )
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(df_movies["combined_features"])
        cosine_sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

        # Chuẩn hóa Độ phổ biến
        df_movies['Độ phổ biến'] = pd.to_numeric(df_movies['Độ phổ biến'], errors='coerce')
        mean_popularity = df_movies['Độ phổ biến'].mean() if not df_movies['Độ phổ biến'].empty else 0
        df_movies['Độ phổ biến'] = df_movies['Độ phổ biến'].fillna(mean_popularity)

        scaler = MinMaxScaler()
        df_movies["popularity_norm"] = scaler.fit_transform(df_movies[["Độ phổ biến"]])

        # 2. Tiền xử lý cho User-Based
        df_movies['parsed_genres'] = df_movies['Thể loại phim'].apply(parse_genres)

        # 3. Tiền xử lý cho Zero-Click (Recency)
        if 'Năm phát hành' in df_movies.columns:
            df_movies['year_numeric'] = pd.to_numeric(df_movies['Năm phát hành'], errors='coerce').fillna(0).astype(int)
            # Chuẩn hóa Recency (2025 là max)
            current_year = 2025
            df_movies['recency_score'] = df_movies['year_numeric'].apply(
                lambda x: 1 if x >= current_year - 1 else (0.8 if x >= current_year - 5 else 0.5))
        else:
            df_movies['year_numeric'] = 0
            df_movies['recency_score'] = 0.5

        # 4. Lấy danh sách tất cả thể loại duy nhất để hiển thị lên UI
        all_genres = set()
        for genres_str in df_movies['Thể loại phim']:
            if genres_str:
                parts = [g.strip() for g in genres_str.split(',')]
                all_genres.update(parts)
        sorted_genres = sorted(list(all_genres))

        # 5. Tính điểm phổ biến thể loại toàn cầu (Global Genre Score)
        genres_pop = {}
        for index, row in df_movies.iterrows():
            popularity = row['Độ phổ biến']
            for genre in row['Thể loại phim'].split(','):
                genre = genre.strip()
                if genre:
                    genres_pop.setdefault(genre, []).append(popularity)

        global_genre_popularity = {g: sum(p) / len(p) for g, p in genres_pop.items() if len(p) > 0}
        max_pop = max(global_genre_popularity.values()) if global_genre_popularity else 1
        normalized_genre_pop = {g: p / max_pop for g, p in global_genre_popularity.items()}

        df_movies['global_genre_score'] = df_movies['Thể loại phim'].apply(
            lambda x: max([normalized_genre_pop.get(g.strip(), 0) for g in x.split(',')], default=0) if x else 0
        )

        return df_movies, cosine_sim_matrix, sorted_genres

    except Exception as e:
        st.error(f"LỖI TẢI DỮ LIỆU: {e}")
        return pd.DataFrame(), np.array([[]]), []


def initialize_user_data():
    """Khởi tạo hoặc tải dữ liệu người dùng vào Session State."""
    if 'df_users' not in st.session_state:
        try:
            df_users = load_data(USER_DATA_FILE)
            df_users.columns = [col.strip() for col in df_users.columns]
            df_users['ID'] = pd.to_numeric(df_users['ID'], errors='coerce')
            df_users = df_users.dropna(subset=['ID'])

            # Đảm bảo có cột 'Thể loại yêu thích'
            if 'Thể loại yêu thích' not in df_users.columns:
                df_users['Thể loại yêu thích'] = ""

        except Exception:
            # Tạo DataFrame mới nếu file lỗi hoặc không tồn tại
            df_users = pd.DataFrame(
                columns=['ID', 'Tên người dùng', '5 phim coi gần nhất', 'Phim yêu thích nhất', 'Thể loại yêu thích'])

        st.session_state['df_users'] = df_users

    return st.session_state['df_users']


def get_unique_movie_titles(df_movies):
    return df_movies['Tên phim'].dropna().unique().tolist()


# ==============================================================================
# II. CHỨC NĂNG ĐĂNG KÝ / ĐĂNG NHẬP
# ==============================================================================

def set_auth_mode(mode):
    st.session_state['auth_mode'] = mode
    st.session_state['last_sim_result'] = pd.DataFrame()
    st.session_state['last_profile_recommendations'] = pd.DataFrame()


def login_as_guest():
    st.session_state['logged_in_user'] = GUEST_USER
    st.session_state['auth_mode'] = 'login'
    st.rerun()


def logout():
    st.session_state['logged_in_user'] = None
    st.session_state['auth_mode'] = 'login'
    st.session_state['last_profile_recommendations'] = pd.DataFrame()
    st.rerun()


# ---------------------------
# PHẦN ĐĂNG KÝ MỚI (CHỈNH SỬA THEO YÊU CẦU: CHỌN THỂ LOẠI)
# ---------------------------
def register_new_user_form(df_movies, sorted_genres):
    """Form đăng ký người dùng mới với UI chọn thể loại (Netflix Style)."""

    # CSS tùy chỉnh để làm đẹp st.pills nếu cần (tùy chọn)
    st.markdown("""
    <style>
    div[data-testid="stForm"] {border: 1px solid #333; padding: 20px; border-radius: 10px;}
    </style>
    """, unsafe_allow_html=True)

    st.header("📝 Đăng Ký Tài Khoản Mới")
    st.caption("Hãy cho chúng tôi biết sở thích của bạn để nhận gợi ý ngay lập tức!")

    df_users = st.session_state['df_users']

    with st.form("register_form_new"):
        # 1. Tên người dùng
        username = st.text_input("Tên đăng nhập (Duy nhất):", placeholder="Ví dụ: movie_lover_99").strip()

        st.write("---")

        # 2. Chọn thể loại (Thay thế phần chọn 5 phim cũ)
        st.subheader("🎯 Bạn thích thể loại nào?")
        st.caption("Chọn ít nhất **3 thể loại** để hồ sơ của bạn chính xác hơn.")

        # Sử dụng st.pills (Streamlit 1.40+) hoặc st.multiselect
        # Nếu st.pills chưa chạy được ở bản cũ, đổi thành st.multiselect
        selected_genres = st.pills(
            "Danh sách thể loại:",
            options=sorted_genres,
            selection_mode="multi"
        )

        st.write("")

        # 3. Phim yêu thích nhất (Giữ lại làm tùy chọn)
        st.markdown("**⭐ Phim Yêu Thích Nhất (Tùy chọn):**")
        st.caption("Nếu có một phim bạn cực kỳ tâm đắc, hãy chọn nó.")
        movie_titles_list = get_unique_movie_titles(df_movies)
        favorite_movie = st.selectbox(
            "Chọn phim:",
            options=["-- Bỏ qua --"] + movie_titles_list,
            index=0
        )
        if favorite_movie == "-- Bỏ qua --":
            favorite_movie = ""

        st.write("---")
        submitted = st.form_submit_button("✨ Đăng Ký & Khám Phá Ngay", type="primary", use_container_width=True)

        if submitted:
            # Validate Input
            if not username:
                st.error("⚠️ Vui lòng nhập tên người dùng.")
                return

            if username in df_users['Tên người dùng'].values:
                st.error(f"❌ Tên '{username}' đã tồn tại. Vui lòng chọn tên khác.")
                return

            if not selected_genres or len(selected_genres) < 1:
                st.warning("⚠️ Vui lòng chọn ít nhất 1 thể loại.")
                return

            # Tạo ID mới
            max_id = df_users['ID'].max() if not df_users.empty and pd.notna(df_users['ID'].max()) else 0
            new_id = int(max_id) + 1

            # Lưu dữ liệu
            # Lưu ý: Cột '5 phim coi gần nhất' sẽ để trống list '[]' vì user mới chưa xem phim nào
            # Thay vào đó ta lưu vào cột 'Thể loại yêu thích'
            new_user_data = {
                'ID': [new_id],
                'Tên người dùng': [username],
                '5 phim coi gần nhất': ["[]"],
                'Phim yêu thích nhất': [favorite_movie],
                'Thể loại yêu thích': [", ".join(selected_genres)]  # Lưu danh sách thể loại dạng chuỗi
            }
            new_user_df = pd.DataFrame(new_user_data)

            # Cập nhật Session State
            st.session_state['df_users'] = pd.concat([df_users, new_user_df], ignore_index=True)
            st.session_state['logged_in_user'] = username

            st.success(f"🎉 Chào mừng {username}! Hệ thống đang tạo gợi ý cho bạn...")
            st.rerun()


def login_form():
    """Form đăng nhập."""
    st.header("🔑 Đăng Nhập")
    df_users = st.session_state['df_users']
    with st.form("login_form"):
        username = st.text_input("Tên người dùng:").strip()
        submitted = st.form_submit_button("Đăng Nhập", use_container_width=True)
        if submitted:
            if username in df_users['Tên người dùng'].values:
                st.session_state['logged_in_user'] = username
                st.success(f"✅ Chào mừng trở lại, {username}.")
                st.rerun()
            else:
                st.error("❌ Tên người dùng không tồn tại.")


def authentication_page(df_movies, sorted_genres):
    """Trang Xác thực."""
    st.title("🎬 HỆ THỐNG ĐỀ XUẤT PHIM")

    col1, col2 = st.columns(2)
    with col1:
        st.button("Đăng Nhập", key="btn_login", on_click=set_auth_mode, args=('login',), use_container_width=True)
    with col2:
        st.button("Đăng Ký Mới", key="btn_register", on_click=set_auth_mode, args=('register',),
                  use_container_width=True)

    st.write("---")

    if st.session_state['auth_mode'] == 'login':
        login_form()
        st.write("")
        st.button("🚀 Chỉ muốn xem dạo? (Chế độ Khách)", on_click=login_as_guest)
    elif st.session_state['auth_mode'] == 'register':
        register_new_user_form(df_movies, sorted_genres)


# ==============================================================================
# III. CHỨC NĂNG ĐỀ XUẤT (UPDATED LOGIC)
# ==============================================================================

def get_recommendations_weighted_genres(selected_genres, df_movies, num_recommendations=10):
    """
    Logic Đề xuất cho User Mới (Cold Start) dựa trên Thể loại đã chọn.
    Sử dụng trọng số: Popularity + Recency + Genre Match Count
    """
    # 1. Lọc phim chứa ít nhất 1 thể loại
    pattern = '|'.join([re.escape(g) for g in selected_genres])
    filtered_df = df_movies[df_movies['Thể loại phim'].str.contains(pattern, case=False, na=False)].copy()

    if filtered_df.empty:
        return pd.DataFrame()

    # 2. Tính điểm
    def calculate_score(row):
        score = 0
        # A. Điểm Phổ biến (Scale 0-1) * Trọng số
        score += row['popularity_norm'] * 2.0

        # B. Điểm Trùng khớp Thể loại (Quan trọng nhất)
        # Đếm số lượng thể loại trùng
        row_genres = [g.strip() for g in row['Thể loại phim'].split(',')]
        match_count = sum(1 for g in selected_genres if g in row_genres)
        score += match_count * 1.5

        # C. Điểm Phim Mới (Recency)
        score += row['recency_score'] * 1.0

        return score

    filtered_df['final_score'] = filtered_df.apply(calculate_score, axis=1)

    # 3. Sắp xếp
    recs = filtered_df.sort_values(by='final_score', ascending=False).head(num_recommendations)
    return recs[['Tên phim', 'Thể loại phim', 'Độ phổ biến', 'final_score', 'Năm phát hành']]


def get_recommendations(username, df_movies, num_recommendations=10):
    """
    Hàm Đề xuất Thông minh: Tự động chọn thuật toán dựa trên dữ liệu user.
    """
    df_users = st.session_state['df_users']
    user_row = df_users[df_users['Tên người dùng'] == username]
    if user_row.empty: return pd.DataFrame()

    # Lấy dữ liệu user
    watched_str = user_row['5 phim coi gần nhất'].iloc[0]
    favorite_movie = user_row['Phim yêu thích nhất'].iloc[0]
    fav_genres_str = str(user_row.get('Thể loại yêu thích', pd.Series([""])).iloc[0])  # Lấy cột thể loại an toàn

    # Xử lý list phim đã xem
    watched_list = []
    try:
        watched_list = ast.literal_eval(watched_str)
        if not isinstance(watched_list, list): watched_list = []
    except:
        watched_list = [m.strip().strip("'") for m in watched_str.strip('[]').split(',') if m.strip()]

    # === TRƯỜNG HỢP 1: NGƯỜI DÙNG CŨ (CÓ LỊCH SỬ XEM) ===
    # Sử dụng logic cũ (Content-based Similarity)
    if len(watched_list) > 0:
        watched_and_favorite = set(watched_list + [favorite_movie])
        # Lấy tập hợp genres từ các phim đã xem
        watched_genres = df_movies[df_movies['Tên phim'].isin(watched_list)]
        user_genres_set = set()
        for genres in watched_genres['parsed_genres']:
            user_genres_set.update(genres)

        if not user_genres_set: return pd.DataFrame()

        # Tìm phim chưa xem
        candidate_movies = df_movies[~df_movies['Tên phim'].isin(watched_and_favorite)].copy()

        # Tính điểm giống genre
        candidate_movies['Similarity_Score'] = candidate_movies['parsed_genres'].apply(
            lambda x: len(x.intersection(user_genres_set))
        )

        # Sắp xếp
        recs = candidate_movies.sort_values(by=['Similarity_Score', 'Độ phổ biến'], ascending=[False, False])
        return recs[['Tên phim', 'Thể loại phim', 'Độ phổ biến', 'Similarity_Score']].head(num_recommendations)

    # === TRƯỜNG HỢP 2: NGƯỜI DÙNG MỚI (CHỈ CÓ THỂ LOẠI) ===
    # Sử dụng logic mới (Weighted Scoring)
    elif fav_genres_str and fav_genres_str.strip():
        selected_genres = [g.strip() for g in fav_genres_str.split(',') if g.strip()]
        return get_recommendations_weighted_genres(selected_genres, df_movies, num_recommendations)

    # === TRƯỜNG HỢP 3: KHÔNG CÓ GÌ ===
    else:
        return pd.DataFrame()  # Trả về rỗng


# (Các hàm hỗ trợ cũ giữ nguyên)
def get_movie_index(movie_name, df_movies):
    try:
        return df_movies[df_movies['Tên phim'].str.lower() == movie_name.lower()].index[0]
    except IndexError:
        return -1


def recommend_movies_smart(movie_name, weight_sim, weight_pop, df_movies, cosine_sim):
    idx = get_movie_index(movie_name, df_movies)
    if idx == -1: return pd.DataFrame()
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores_df = pd.DataFrame(sim_scores, columns=['index', 'similarity'])
    df_result = pd.merge(df_movies, sim_scores_df, left_index=True, right_on='index')
    df_result['weighted_score'] = (weight_sim * df_result['similarity'] + weight_pop * df_result['popularity_norm'])
    df_result = df_result.drop(df_result[df_result['Tên phim'] == movie_name].index)
    df_result = df_result.sort_values(by='weighted_score', ascending=False)
    return df_result[['Tên phim', 'weighted_score', 'similarity', 'Độ phổ biến', 'Thể loại phim']].head(10)


def plot_genre_popularity(movie_name, recommended_movies_df, df_movies, is_user_based=False):
    # (Giữ nguyên logic vẽ biểu đồ của bạn, chỉ thêm check để tránh lỗi)
    if recommended_movies_df.empty: return

    genres_data = []
    # Lấy dữ liệu từ df đề xuất
    for index, row in recommended_movies_df.iterrows():
        genres_list = [g.strip() for g in row['Thể loại phim'].split(',') if g.strip()]
        for genre in genres_list:
            genres_data.append({'Thể loại': genre, 'Độ phổ biến': row['Độ phổ biến']})

    df_plot = pd.DataFrame(genres_data)
    if df_plot.empty: return

    genre_avg_pop = df_plot.groupby('Thể loại')['Độ phổ biến'].mean().reset_index()
    top_7_genres = genre_avg_pop.sort_values(by='Độ phổ biến', ascending=False).head(7)

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(top_7_genres['Thể loại'], top_7_genres['Độ phổ biến'], color='#E50914', alpha=0.8)  # Màu đỏ Netflix
    ax.set_title(f"Phân phối độ phổ biến thể loại (Top 7)", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig)


# ==============================================================================
# IV. GIAO DIỆN CHÍNH (MAIN PAGE)
# ==============================================================================

def main_page(df_movies, cosine_sim):
    is_guest = st.session_state['logged_in_user'] == GUEST_USER
    username = st.session_state['logged_in_user']

    st.title(f"🍿 Chào {username}, hôm nay xem gì?")

    # --- SIDEBAR ---
    st.sidebar.title("Menu")
    if is_guest:
        if st.sidebar.button("Đăng Xuất Khách", on_click=logout): pass
    else:
        menu_choice = st.sidebar.radio("Chức năng:", ('Đề xuất Cá Nhân', 'Tìm theo Phim', 'Đăng Xuất'))
        if menu_choice == 'Đăng Xuất': logout()

    # --- NỘI DUNG CHÍNH ---

    # 1. GIAO DIỆN KHÁCH (ZERO-CLICK GLOBAL)
    if is_guest:
        st.subheader("🔥 Top Thịnh Hành & Mới Nhất")
        # Logic Zero-Click thuần túy (Global)
        df_guest = df_movies.sort_values(by=['year_numeric', 'popularity_norm'], ascending=[False, False]).head(10)
        st.dataframe(df_guest[['Tên phim', 'Năm phát hành', 'Thể loại phim', 'Độ phổ biến']], use_container_width=True)
        return

    # 2. GIAO DIỆN USER ĐĂNG NHẬP
    if menu_choice == 'Đề xuất Cá Nhân':
        st.header("✨ Gợi ý dành riêng cho bạn")

        # Hiển thị thông tin user đang có
        df_users = st.session_state['df_users']
        user_info = df_users[df_users['Tên người dùng'] == username].iloc[0]

        # Check xem user này là kiểu Mới (Có Genre) hay Cũ (Có Phim đã xem)
        has_watched = len(user_info['5 phim coi gần nhất']) > 5  # >5 ký tự nghĩa là list không rỗng
        has_genres = 'Thể loại yêu thích' in user_info and len(str(user_info['Thể loại yêu thích'])) > 0

        if has_genres and not has_watched:
            st.info(f"🎯 Dựa trên các thể loại bạn thích: **{user_info['Thể loại yêu thích']}**")
        elif has_watched:
            st.info("🎯 Dựa trên lịch sử xem phim của bạn.")

        # Nút tìm kiếm
        if st.button("🔄 Làm mới đề xuất", type="primary"):
            recs = get_recommendations(username, df_movies, 15)
            st.session_state['last_profile_recommendations'] = recs
            st.session_state['show_profile_plot'] = True

        # Hiển thị kết quả
        if not st.session_state['last_profile_recommendations'].empty:
            recs = st.session_state['last_profile_recommendations']

            # Hiển thị dạng lưới đẹp hơn
            for i, row in recs.iterrows():
                with st.container(border=True):
                    c1, c2 = st.columns([4, 1])
                    with c1:
                        st.subheader(f"#{i + 1} {row['Tên phim']}")
                        st.caption(f"Thể loại: {row['Thể loại phim']}")
                    with c2:
                        score = row.get('final_score', row.get('Similarity_Score', 0))
                        st.metric("Điểm hợp", f"{score:.1f}")

            # Biểu đồ
            if st.checkbox("Hiển thị phân tích thể loại", value=True):
                plot_genre_popularity(None, recs, df_movies, True)
        else:
            st.warning("Chưa có đề xuất nào. Hãy nhấn nút 'Làm mới'!")

    elif menu_choice == 'Tìm theo Phim':
        st.header("🔎 Tìm phim tương tự")
        movie_titles = get_unique_movie_titles(df_movies)
        selected_movie = st.selectbox("Chọn phim gốc:", movie_titles)
        if st.button("Tìm kiếm"):
            res = recommend_movies_smart(selected_movie, 0.7, 0.3, df_movies, cosine_sim)
            st.dataframe(res, use_container_width=True)


# ==============================================================================
# V. CHẠY ỨNG DỤNG
# ==============================================================================

if __name__ == '__main__':
    # 1. Load Data
    df_movies, cosine_sim, sorted_genres = load_and_preprocess_static_data()
    initialize_user_data()

    # 2. Điều hướng
    if st.session_state['logged_in_user']:
        main_page(df_movies, cosine_sim)
    else:
        authentication_page(df_movies, sorted_genres)