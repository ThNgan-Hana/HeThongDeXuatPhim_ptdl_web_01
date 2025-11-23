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
# 1. CẤU HÌNH & KHỞI TẠO
# ==============================================================================

# --- Tên file dữ liệu (Phải khớp với file trên GitHub) ---
USER_DATA_FILE = "danh_sach_nguoi_dung_moi.csv"
MOVIE_DATA_FILE = "movie_info_1000.csv"
GUEST_USER = "Guest_ZeroClick"

# --- Khởi tạo Session State (Lưu trạng thái phiên làm việc) ---
if 'logged_in_user' not in st.session_state:
    st.session_state['logged_in_user'] = None
if 'auth_mode' not in st.session_state:
    st.session_state['auth_mode'] = 'login'

# Biến lưu kết quả để không bị mất khi thao tác
if 'last_profile_recommendations' not in st.session_state: st.session_state[
    'last_profile_recommendations'] = pd.DataFrame()
if 'show_profile_plot' not in st.session_state: st.session_state['show_profile_plot'] = False


# ==============================================================================
# 2. HÀM XỬ LÝ DỮ LIỆU (DATA LOADING & PREPROCESSING)
# ==============================================================================

@st.cache_data
def load_data(file_path):
    """Tải dữ liệu CSV an toàn."""
    try:
        return pd.read_csv(file_path).fillna("")
    except FileNotFoundError:
        st.error(f"⚠️ LỖI: Không tìm thấy file '{file_path}'. Vui lòng kiểm tra lại GitHub.")
        return pd.DataFrame()


def parse_genres(genre_string):
    """Chuyển chuỗi thể loại 'Hành động, Hài' thành tập hợp {'Hành động', 'Hài'}."""
    if not isinstance(genre_string, str) or not genre_string:
        return set()
    genres = [g.strip().replace('"', '') for g in genre_string.split(',')]
    return set(genres)


@st.cache_resource
def load_and_preprocess_static_data():
    """
    Tải và xử lý dữ liệu Phim (Chạy 1 lần duy nhất để tối ưu tốc độ).
    """
    try:
        df_movies = load_data(MOVIE_DATA_FILE)
        if df_movies.empty:
            return pd.DataFrame(), np.array([[]]), []

        df_movies.columns = [col.strip() for col in df_movies.columns]

        # A. Xử lý cho Content-Based (Dành cho user cũ)
        df_movies["combined_features"] = (
                df_movies["Đạo diễn"] + " " +
                df_movies["Diễn viên chính"] + " " +
                df_movies["Thể loại phim"]
        )
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(df_movies["combined_features"])
        cosine_sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

        # B. Chuẩn hóa Độ phổ biến (Popularity)
        df_movies['Độ phổ biến'] = pd.to_numeric(df_movies['Độ phổ biến'], errors='coerce')
        mean_popularity = df_movies['Độ phổ biến'].mean() if not df_movies['Độ phổ biến'].empty else 0
        df_movies['Độ phổ biến'] = df_movies['Độ phổ biến'].fillna(mean_popularity)

        scaler = MinMaxScaler()
        df_movies["popularity_norm"] = scaler.fit_transform(df_movies[["Độ phổ biến"]])

        # C. Xử lý Thể loại (Parsing)
        df_movies['parsed_genres'] = df_movies['Thể loại phim'].apply(parse_genres)

        # D. Xử lý Độ mới (Recency Score) - Ưu tiên phim 2024, 2025
        if 'Năm phát hành' in df_movies.columns:
            df_movies['year_numeric'] = pd.to_numeric(df_movies['Năm phát hành'], errors='coerce').fillna(0).astype(int)
            current_year = 2025
            # Logic: Phim mới nhất (2024-2025) điểm cao, phim cũ điểm thấp dần
            df_movies['recency_score'] = df_movies['year_numeric'].apply(
                lambda x: 1.0 if x >= current_year - 1 else (0.8 if x >= current_year - 5 else 0.5))
        else:
            df_movies['year_numeric'] = 0
            df_movies['recency_score'] = 0.5

        # E. Tạo danh sách thể loại duy nhất để hiển thị lên UI
        all_genres = set()
        for genres_str in df_movies['Thể loại phim']:
            if genres_str:
                parts = [g.strip() for g in genres_str.split(',')]
                all_genres.update(parts)
        sorted_genres = sorted(list(all_genres))

        return df_movies, cosine_sim_matrix, sorted_genres

    except Exception as e:
        st.error(f"LỖI TẢI DỮ LIỆU: {e}")
        return pd.DataFrame(), np.array([[]]), []


def initialize_user_data():
    """Khởi tạo hoặc tải dữ liệu User."""
    if 'df_users' not in st.session_state:
        try:
            df_users = load_data(USER_DATA_FILE)
            if not df_users.empty:
                df_users.columns = [col.strip() for col in df_users.columns]
                # Đảm bảo cột ID là số
                if 'ID' in df_users.columns:
                    df_users['ID'] = pd.to_numeric(df_users['ID'], errors='coerce')
                    df_users = df_users.dropna(subset=['ID'])

                # Tạo cột 'Thể loại yêu thích' nếu chưa có
                if 'Thể loại yêu thích' not in df_users.columns:
                    df_users['Thể loại yêu thích'] = ""
            else:
                df_users = pd.DataFrame(columns=['ID', 'Tên người dùng', '5 phim coi gần nhất', 'Phim yêu thích nhất',
                                                 'Thể loại yêu thích'])

        except Exception:
            df_users = pd.DataFrame(
                columns=['ID', 'Tên người dùng', '5 phim coi gần nhất', 'Phim yêu thích nhất', 'Thể loại yêu thích'])

        st.session_state['df_users'] = df_users

    return st.session_state['df_users']


def get_unique_movie_titles(df_movies):
    if 'Tên phim' in df_movies.columns:
        return df_movies['Tên phim'].dropna().unique().tolist()
    return []


# ==============================================================================
# 3. HỆ THỐNG XÁC THỰC (AUTH & FORM)
# ==============================================================================

def set_auth_mode(mode):
    st.session_state['auth_mode'] = mode
    st.session_state['last_profile_recommendations'] = pd.DataFrame()


def logout():
    st.session_state['logged_in_user'] = None
    st.session_state['auth_mode'] = 'login'
    st.session_state['last_profile_recommendations'] = pd.DataFrame()
    st.rerun()


def register_new_user_form(df_movies, sorted_genres):
    """
    Form Đăng ký Mới: Thay vì chọn 5 phim, giờ chọn Thể loại (Genre).
    """
    st.header("📝 Đăng Ký Tài Khoản Mới")
    st.caption("Hãy cho chúng tôi biết sở thích của bạn để nhận gợi ý ngay lập tức!")

    df_users = st.session_state['df_users']

    with st.form("register_form_new"):
        username = st.text_input("Tên đăng nhập (Duy nhất):", placeholder="Ví dụ: movie_lover_99").strip()
        st.write("---")

        st.subheader("🎯 Bạn thích thể loại nào?")
        st.caption("Chọn ít nhất **3 thể loại**.")

        # --- SAFE FALLBACK: Kiểm tra xem có st.pills không ---
        # Nếu Streamlit cũ chưa có pills, tự động dùng multiselect
        if hasattr(st, 'pills'):
            selected_genres = st.pills("Danh sách thể loại:", options=sorted_genres, selection_mode="multi")
        else:
            selected_genres = st.multiselect("Danh sách thể loại:", options=sorted_genres)

        st.write("")
        st.markdown("**⭐ Phim Yêu Thích Nhất (Tùy chọn):**")
        movie_titles_list = get_unique_movie_titles(df_movies)
        favorite_movie = st.selectbox("Chọn phim:", options=["-- Bỏ qua --"] + movie_titles_list, index=0)

        if favorite_movie == "-- Bỏ qua --": favorite_movie = ""

        st.write("---")
        submitted = st.form_submit_button("✨ Đăng Ký & Khám Phá Ngay", type="primary", use_container_width=True)

        if submitted:
            # 1. Validate
            if not username:
                st.error("⚠️ Vui lòng nhập tên người dùng.")
                return
            if not df_users.empty and username in df_users['Tên người dùng'].values:
                st.error(f"❌ Tên '{username}' đã tồn tại.")
                return
            if not selected_genres or len(selected_genres) < 1:
                st.warning("⚠️ Vui lòng chọn ít nhất 1 thể loại.")
                return

            # 2. Tạo ID mới
            max_id = 0
            if not df_users.empty and 'ID' in df_users.columns:
                max_id = df_users['ID'].max() if pd.notna(df_users['ID'].max()) else 0
            new_id = int(max_id) + 1

            # 3. Lưu user mới
            # Lưu ý: '5 phim coi gần nhất' là list rỗng "[]" vì user mới chưa xem gì.
            # Dữ liệu quan trọng nhất là 'Thể loại yêu thích'.
            new_user_data = {
                'ID': [new_id],
                'Tên người dùng': [username],
                '5 phim coi gần nhất': ["[]"],
                'Phim yêu thích nhất': [favorite_movie],
                'Thể loại yêu thích': [", ".join(selected_genres)]
            }
            new_user_df = pd.DataFrame(new_user_data)

            st.session_state['df_users'] = pd.concat([df_users, new_user_df], ignore_index=True)
            st.session_state['logged_in_user'] = username
            st.success(f"🎉 Chào mừng {username}!")
            st.rerun()


def login_form():
    """Form đăng nhập."""
    st.header("🔑 Đăng Nhập")
    df_users = st.session_state['df_users']
    with st.form("login_form"):
        username = st.text_input("Tên người dùng:").strip()
        submitted = st.form_submit_button("Đăng Nhập", use_container_width=True)
        if submitted:
            if not df_users.empty and username in df_users['Tên người dùng'].values:
                st.session_state['logged_in_user'] = username
                st.success(f"✅ Chào mừng trở lại, {username}.")
                st.rerun()
            else:
                st.error("❌ Tên người dùng không tồn tại.")


def authentication_page(df_movies, sorted_genres):
    """Trang điều hướng Đăng nhập / Đăng ký."""
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
        # Nút cho khách vãng lai
        if st.button("🚀 Chỉ muốn xem dạo? (Chế độ Khách)"):
            st.session_state['logged_in_user'] = GUEST_USER
            st.rerun()

    elif st.session_state['auth_mode'] == 'register':
        register_new_user_form(df_movies, sorted_genres)


# ==============================================================================
# 4. LOGIC ĐỀ XUẤT PHIM (RECOMMENDATION ENGINE)
# ==============================================================================

def get_recommendations_weighted_genres(selected_genres, df_movies, num_recommendations=10):
    """
    LOGIC MỚI: Dành cho User Mới (Cold Start).
    Tính điểm dựa trên: Độ phổ biến + Số lượng thể loại trùng khớp + Độ mới.
    """
    # Lọc các phim có chứa ít nhất 1 thể loại đã chọn
    pattern = '|'.join([re.escape(g) for g in selected_genres])
    filtered_df = df_movies[df_movies['Thể loại phim'].str.contains(pattern, case=False, na=False)].copy()

    if filtered_df.empty: return pd.DataFrame()

    def calculate_score(row):
        score = 0
        # 1. Điểm Phổ biến (Scale 0-1) * Trọng số 2.0
        score += row['popularity_norm'] * 2.0

        # 2. Điểm Trùng khớp Thể loại * Trọng số 1.5
        row_genres = [g.strip() for g in row['Thể loại phim'].split(',')]
        match_count = sum(1 for g in selected_genres if g in row_genres)
        score += match_count * 1.5

        # 3. Điểm Phim Mới (Recency) * Trọng số 1.0
        score += row['recency_score'] * 1.0
        return score

    filtered_df['final_score'] = filtered_df.apply(calculate_score, axis=1)

    # Sắp xếp điểm từ cao xuống thấp
    recs = filtered_df.sort_values(by='final_score', ascending=False).head(num_recommendations)
    return recs[['Tên phim', 'Thể loại phim', 'Độ phổ biến', 'final_score', 'Năm phát hành']]


def get_recommendations(username, df_movies, num_recommendations=10):
    """
    Hàm điều phối thông minh:
    - User cũ (có lịch sử xem): Dùng Content-Based.
    - User mới (chỉ có thể loại): Dùng Weighted Scoring.
    """
    df_users = st.session_state['df_users']
    user_row = df_users[df_users['Tên người dùng'] == username]
    if user_row.empty: return pd.DataFrame()

    # Lấy dữ liệu user
    watched_str = user_row['5 phim coi gần nhất'].iloc[0]
    favorite_movie = user_row['Phim yêu thích nhất'].iloc[0]
    # Lấy thể loại yêu thích (nếu có)
    fav_genres_str = str(user_row.get('Thể loại yêu thích', pd.Series([""])).iloc[0])

    # Parse lịch sử xem phim
    watched_list = []
    try:
        watched_list = ast.literal_eval(watched_str)
        if not isinstance(watched_list, list): watched_list = []
    except:
        watched_list = [m.strip().strip("'") for m in watched_str.strip('[]').split(',') if m.strip()]

    # --- TRƯỜNG HỢP 1: NGƯỜI DÙNG CŨ (Đã xem phim) ---
    if len(watched_list) > 0:
        watched_and_favorite = set(watched_list + [favorite_movie])
        watched_genres = df_movies[df_movies['Tên phim'].isin(watched_list)]

        user_genres_set = set()
        for genres in watched_genres['parsed_genres']:
            user_genres_set.update(genres)

        if not user_genres_set: return pd.DataFrame()

        candidate_movies = df_movies[~df_movies['Tên phim'].isin(watched_and_favorite)].copy()
        # Tính độ giống nhau dựa trên tập hợp thể loại
        candidate_movies['Similarity_Score'] = candidate_movies['parsed_genres'].apply(
            lambda x: len(x.intersection(user_genres_set))
        )
        recs = candidate_movies.sort_values(by=['Similarity_Score', 'Độ phổ biến'], ascending=[False, False])
        return recs[['Tên phim', 'Thể loại phim', 'Độ phổ biến', 'Similarity_Score']].head(num_recommendations)

    # --- TRƯỜNG HỢP 2: NGƯỜI DÙNG MỚI (Chỉ chọn thể loại) ---
    elif fav_genres_str and fav_genres_str.strip():
        selected_genres = [g.strip() for g in fav_genres_str.split(',') if g.strip()]
        return get_recommendations_weighted_genres(selected_genres, df_movies, num_recommendations)

    # --- TRƯỜNG HỢP 3: KHÔNG CÓ DỮ LIỆU ---
    else:
        return pd.DataFrame()


def recommend_movies_smart(movie_name, weight_sim, weight_pop, df_movies, cosine_sim):
    """Tìm phim tương tự theo tên (Chức năng tìm kiếm)."""
    try:
        idx = df_movies[df_movies['Tên phim'].str.lower() == movie_name.lower()].index[0]
    except IndexError:
        return pd.DataFrame()

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores_df = pd.DataFrame(sim_scores, columns=['index', 'similarity'])
    df_result = pd.merge(df_movies, sim_scores_df, left_index=True, right_on='index')

    df_result['weighted_score'] = (weight_sim * df_result['similarity'] + weight_pop * df_result['popularity_norm'])

    df_result = df_result.drop(df_result[df_result['Tên phim'] == movie_name].index)
    df_result = df_result.sort_values(by='weighted_score', ascending=False)

    return df_result[['Tên phim', 'weighted_score', 'similarity', 'Độ phổ biến', 'Thể loại phim']].head(10)


def plot_genre_popularity(recommended_movies_df):
    """Vẽ biểu đồ phân phối thể loại."""
    if recommended_movies_df.empty: return
    genres_data = []
    for index, row in recommended_movies_df.iterrows():
        genres_list = [g.strip() for g in row['Thể loại phim'].split(',') if g.strip()]
        for genre in genres_list:
            genres_data.append({'Thể loại': genre, 'Độ phổ biến': row['Độ phổ biến']})

    df_plot = pd.DataFrame(genres_data)
    if df_plot.empty: return

    genre_avg_pop = df_plot.groupby('Thể loại')['Độ phổ biến'].mean().reset_index()
    top_7_genres = genre_avg_pop.sort_values(by='Độ phổ biến', ascending=False).head(7)

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(top_7_genres['Thể loại'], top_7_genres['Độ phổ biến'], color='#E50914', alpha=0.8)
    ax.set_title(f"Xu hướng thể loại trong danh sách gợi ý", fontsize=10)
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig)


# ==============================================================================
# 5. GIAO DIỆN CHÍNH (MAIN APP)
# ==============================================================================

def main_page(df_movies, cosine_sim):
    is_guest = st.session_state['logged_in_user'] == GUEST_USER
    username = st.session_state['logged_in_user']

    st.title(f"🍿 Chào {username}, hôm nay xem gì?")

    st.sidebar.title("Menu")
    if is_guest:
        if st.sidebar.button("Đăng Xuất Khách", on_click=logout): pass
    else:
        menu_choice = st.sidebar.radio("Chức năng:", ('Đề xuất Cá Nhân', 'Tìm theo Phim', 'Đăng Xuất'))
        if menu_choice == 'Đăng Xuất': logout()

    # --- A. GIAO DIỆN KHÁCH (GLOBAL TOP - ZERO CLICK) ---
    if is_guest:
        st.subheader("🔥 Top Thịnh Hành & Mới Nhất (Zero-Click)")
        # Sắp xếp theo Năm mới nhất -> Độ phổ biến cao nhất
        df_guest = df_movies.sort_values(by=['year_numeric', 'popularity_norm'], ascending=[False, False]).head(10)
        st.dataframe(df_guest[['Tên phim', 'Năm phát hành', 'Thể loại phim', 'Độ phổ biến']], use_container_width=True)
        return

    # --- B. GIAO DIỆN USER ĐĂNG NHẬP ---
    if menu_choice == 'Đề xuất Cá Nhân':
        st.header("✨ Gợi ý dành riêng cho bạn")
        df_users = st.session_state['df_users']
        user_info = df_users[df_users['Tên người dùng'] == username].iloc[0]

        has_watched = len(user_info['5 phim coi gần nhất']) > 5
        has_genres = 'Thể loại yêu thích' in user_info and len(str(user_info['Thể loại yêu thích'])) > 0

        # Hiển thị thông báo để user biết hệ thống đang dùng dữ liệu gì
        if has_genres and not has_watched:
            st.info(f"🎯 Gợi ý dựa trên sở thích thể loại: **{user_info['Thể loại yêu thích']}**")
        elif has_watched:
            st.info("🎯 Gợi ý dựa trên lịch sử xem phim của bạn.")

        if st.button("🔄 Lấy Đề Xuất Mới Nhất", type="primary"):
            recs = get_recommendations(username, df_movies, 15)
            st.session_state['last_profile_recommendations'] = recs
            st.session_state['show_profile_plot'] = True

        # Hiển thị kết quả
        if not st.session_state['last_profile_recommendations'].empty:
            recs = st.session_state['last_profile_recommendations']

            # 1. Vẽ biểu đồ trước
            if st.session_state['show_profile_plot']:
                with st.expander("📊 Phân tích xu hướng (Biểu đồ)", expanded=True):
                    plot_genre_popularity(recs)

            # 2. Hiển thị danh sách phim đẹp mắt
            st.write("---")
            for i, row in recs.iterrows():
                with st.container(border=True):
                    c1, c2 = st.columns([4, 1])
                    with c1:
                        st.subheader(f"#{i + 1} {row['Tên phim']}")
                        st.caption(f"📅 Năm: {row.get('Năm phát hành', 'N/A')} | 🏷️ {row['Thể loại phim']}")
                    with c2:
                        score = row.get('final_score', row.get('Similarity_Score', 0))
                        st.metric("Điểm Hợp", f"{score:.1f}")
        else:
            if not has_watched and not has_genres:
                st.warning("Hồ sơ của bạn chưa có dữ liệu. Hãy cập nhật sở thích!")
            else:
                st.info("Hãy nhấn nút 'Lấy Đề Xuất Mới Nhất' để xem kết quả.")

    elif menu_choice == 'Tìm theo Phim':
        st.header("🔎 Tìm phim tương tự")
        movie_titles = get_unique_movie_titles(df_movies)
        selected_movie = st.selectbox("Chọn phim gốc:", movie_titles)
        if st.button("Tìm kiếm"):
            res = recommend_movies_smart(selected_movie, 0.7, 0.3, df_movies, cosine_sim)
            st.dataframe(res, use_container_width=True)


# ==============================================================================
# 6. ENTRY POINT
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