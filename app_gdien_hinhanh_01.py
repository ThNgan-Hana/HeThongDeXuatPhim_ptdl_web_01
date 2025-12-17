import streamlit as st
import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from collections import Counter

# ==============================================================================
# 1. CẤU HÌNH TRANG & CSS
# ==============================================================================
st.set_page_config(
    page_title="Movie RecSys AI",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #ff4b4b;
        color: white;
    }
    .movie-card {
        background-color: #262730;
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 10px;
        text-align: center;
    }
    .movie-title {
        font-weight: bold;
        font-size: 1.1em;
        margin-top: 5px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. HÀM TIỀN XỬ LÝ DỮ LIỆU (ĐÃ FIX LỖI DATA)
# ==============================================================================
@st.cache_resource
def load_and_process_data():
    # Load data
    movies = pd.read_csv("data_phim_full_images.csv")
    users = pd.read_csv("danh_sach_nguoi_dung_moi.csv")

    # --- QUAN TRỌNG: LÀM SẠCH TÊN PHIM ---
    # Xóa khoảng trắng thừa ở đầu/cuối để khớp chính xác hơn
    movies['Tên phim'] = movies['Tên phim'].astype(str).str.strip()
    
    # --- Xử lý dữ liệu Movies ---
    movies['Đạo diễn'] = movies['Đạo diễn'].fillna('')
    movies['Thể loại phim'] = movies['Thể loại phim'].fillna('')
    movies['Mô tả'] = movies['Mô tả'].fillna('')
    
    # Tạo cột đặc trưng kết hợp (Combined Features) cho AI
    movies['combined_features'] = (
        movies['Tên phim'] + " " + 
        movies['Đạo diễn'] + " " + 
        movies['Thể loại phim']
    )

    # Chuẩn hóa độ phổ biến
    scaler = MinMaxScaler()
    movies['popularity_scaled'] = scaler.fit_transform(movies[['Độ phổ biến']])

    # TF-IDF & Cosine Similarity
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies['combined_features'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # --- Xử lý dữ liệu Users ---
    # Chuyển chuỗi list thành list Python và làm sạch từng phần tử
    def parse_history(x):
        try:
            lst = ast.literal_eval(x) if isinstance(x, str) else []
            # Làm sạch tên phim trong lịch sử người dùng (strip whitespace)
            return [item.strip() for item in lst]
        except:
            return []

    users['history_list'] = users['5 phim coi gần nhất'].apply(parse_history)
    
    # Làm sạch tên phim yêu thích
    users['Phim yêu thích nhất'] = users['Phim yêu thích nhất'].astype(str).str.strip()

    # Lấy danh sách tất cả thể loại
    all_genres = set()
    for genres in movies['Thể loại phim']:
        for g in str(genres).split(','):
            if g.strip():
                all_genres.add(g.strip())
    
    return movies, users, cosine_sim, sorted(list(all_genres))

# Gọi hàm load dữ liệu
movies_df, users_df, cosine_sim, ALL_GENRES = load_and_process_data()

# ==============================================================================
# 3. CÁC HÀM CHỨC NĂNG CỐT LÕI (ALGORITHMS)
# ==============================================================================

def get_ai_recommendations(history_titles, top_k=10, w_sim=0.7, w_pop=0.3):
    """Chức năng 1: Đề xuất AI (Hybrid: Content + Popularity)"""
    indices = []
    for title in history_titles:
        # Tìm chính xác vì đã strip() cả 2 bên
        idx = movies_df[movies_df['Tên phim'] == title].index
        if not idx.empty:
            indices.append(idx[0])
    
    # Nếu không tìm thấy lịch sử hoặc lịch sử rỗng -> Gợi ý phim phổ biến
    if not indices:
        return movies_df.sort_values(by='Độ phổ biến', ascending=False).head(top_k)

    # Tính điểm
    sim_scores = np.mean(cosine_sim[indices], axis=0)
    pop_scores = movies_df['popularity_scaled'].values
    final_scores = (w_sim * sim_scores) + (w_pop * pop_scores)
    
    # Sắp xếp
    scores_with_idx = list(enumerate(final_scores))
    scores_with_idx = sorted(scores_with_idx, key=lambda x: x[1], reverse=True)
    
    # Lọc bỏ phim đã xem
    rec_indices = [i[0] for i in scores_with_idx if i[0] not in indices][:top_k]
    return movies_df.iloc[rec_indices]

def search_movie_func(query):
    """Chức năng 2: Tìm kiếm phim"""
    return movies_df[movies_df['Tên phim'].str.contains(query, case=False, na=False)]

def get_genre_recommendations(selected_genres, top_k=10):
    """Chức năng 3: Đề xuất theo thể loại"""
    if not selected_genres:
        return pd.DataFrame()
    
    pattern = '|'.join(selected_genres)
    filtered = movies_df[movies_df['Thể loại phim'].str.contains(pattern, case=False, na=False)]
    
    if filtered.empty:
        return pd.DataFrame()
    
    return filtered.sort_values(by='Độ phổ biến', ascending=False).head(top_k)

def draw_user_charts(history_titles):
    """Vẽ biểu đồ (Đã thêm Debug lỗi dữ liệu)"""
    if not history_titles:
        st.warning("Chưa có dữ liệu lịch sử để vẽ biểu đồ.")
        return

    genres_count = []
    missing_movies = [] 
    
    for title in history_titles:
        movie_row = movies_df[movies_df['Tên phim'] == title]
        if not movie_row.empty:
            g_str = str(movie_row.iloc[0]['Thể loại phim'])
            if g_str and g_str.lower() != 'nan':
                # Tách và làm sạch thể loại
                current_genres = [x.strip() for x in g_str.split(',') if x.strip()]
                genres_count.extend(current_genres)
        else:
            missing_movies.append(title)
    
    # Báo cáo lỗi nếu tên phim không khớp
    if missing_movies:
        with st.expander("⚠️ Chi tiết lỗi dữ liệu (Bấm để xem)"):
            st.error(f"Không tìm thấy {len(missing_movies)} phim trong CSDL:")
            st.write(missing_movies)
            st.caption("Nguyên nhân: Tên phim trong lịch sử người dùng và file dữ liệu phim không khớp hoàn toàn.")

    if not genres_count:
        st.warning("Không trích xuất được thể loại nào từ lịch sử xem.")
        return

    # Vẽ biểu đồ
    counts = Counter(genres_count)
    df_chart = pd.DataFrame.from_dict(counts, orient='index', columns=['Count']).reset_index()
    df_chart.columns = ['Thể loại', 'Số lượng']
    df_chart = df_chart.sort_values(by='Số lượng', ascending=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Pie Chart (Gộp nhóm nhỏ nếu cần)
    if len(df_chart) > 10:
        top_df = df_chart.head(8)
        other_count = df_chart.iloc[8:]['Số lượng'].sum()
        new_row = pd.DataFrame({'Thể loại': ['Khác'], 'Số lượng': [other_count]})
        chart_data = pd.concat([top_df, new_row])
    else:
        chart_data = df_chart

    ax1.pie(chart_data['Số lượng'], labels=chart_data['Thể loại'], autopct='%1.1f%%', startangle=140, colors=sns.color_palette('pastel'))
    ax1.set_title('Tỷ lệ Thể loại yêu thích')

    # Bar Chart
    sns.barplot(x='Số lượng', y='Thể loại', data=df_chart.head(15), ax=ax2, palette='viridis')
    ax2.set_title('Top Thể loại xem nhiều nhất')
    ax2.set_xlabel("Số phim")
    
    st.pyplot(fig)

# ==============================================================================
# 4. GIAO DIỆN NGƯỜI DÙNG (UI)
# ==============================================================================

# --- Session State Management ---
if 'user_mode' not in st.session_state:
    st.session_state.user_mode = None  # 'member', 'guest', 'register'
if 'current_user' not in st.session_state:
    st.session_state.current_user = None
if 'user_genres' not in st.session_state:
    st.session_state.user_genres = []

# --- Sidebar ---
with st.sidebar:
    st.title("🎬 DreamStream")
    
    if st.session_state.user_mode == 'member':
        st.success(f"Chào, {st.session_state.current_user['Tên người dùng']}!")
        menu = st.radio("Menu", ["Đề xuất AI", "Tìm kiếm Phim", "Theo Thể loại Yêu thích", "Thống kê Cá nhân"])
        if st.button("Đăng xuất"):
            st.session_state.user_mode = None
            st.session_state.current_user = None
            st.rerun()
            
    elif st.session_state.user_mode in ['guest', 'register']:
        # Hiển thị đúng vai trò
        role_label = "KHÁCH" if st.session_state.user_mode == 'guest' else "THÀNH VIÊN MỚI"
        st.info(f"Chế độ: {role_label}")
        
        menu = st.radio("Menu", ["Đề xuất AI (Cơ bản)", "Theo Thể loại Đã chọn"])
        
        # Nút thoát hiển thị linh hoạt
        btn_label = "Thoát chế độ Khách" if st.session_state.user_mode == 'guest' else "Đăng xuất / Quay lại"
        if st.button(btn_label):
            st.session_state.user_mode = None
            st.session_state.user_genres = []
            st.rerun()
            
    else:
        st.warning("Vui lòng đăng nhập.")
        menu = "Login"

# --- Main Content ---

# 1. MÀN HÌNH LOGIN / REGISTER
if st.session_state.user_mode is None:
    tab1, tab2, tab3 = st.tabs(["Đăng nhập", "Đăng ký", "Khách"])
    
    with tab1: # Login
        username = st.text_input("Tên đăng nhập")
        if st.button("Đăng nhập"):
            user_row = users_df[users_df['Tên người dùng'] == username]
            if not user_row.empty:
                st.session_state.user_mode = 'member'
                st.session_state.current_user = user_row.iloc[0]
                st.rerun()
            else:
                st.error("Không tồn tại user này.")

    with tab2: # Register
        new_user = st.text_input("Tên người dùng mới")
        selected_g = st.multiselect("Chọn thể loại:", ALL_GENRES)
        if st.button("Đăng ký"):
            if new_user and selected_g:
                st.session_state.user_mode = 'register'
                st.session_state.current_user = {'Tên người dùng': new_user}
                st.session_state.user_genres = selected_g
                st.rerun()
            else:
                st.warning("Nhập tên và chọn thể loại.")

    with tab3: # Guest
        guest_g = st.multiselect("Chọn thể loại muốn xem:", ALL_GENRES, key='guest')
        if st.button("Truy cập ngay"):
            if guest_g:
                st.session_state.user_mode = 'guest'
                st.session_state.user_genres = guest_g
                st.rerun()
            else:
                st.warning("Chọn ít nhất 1 thể loại.")

# 2. CHỨC NĂNG - MEMBER
elif st.session_state.user_mode == 'member':
    user_history = st.session_state.current_user['history_list']
    
    if menu == "Đề xuất AI":
        st.header("🤖 Đề xuất Phim Thông minh")
        st.write("Dựa trên lịch sử xem của bạn kết hợp xu hướng phổ biến.")
        st.caption(f"Lịch sử đã ghi nhận: {len(user_history)} phim")
        
        recs = get_ai_recommendations(user_history)
        cols = st.columns(5)
        for i, (idx, row) in enumerate(recs.iterrows()):
            with cols[i % 5]:
                st.image(row['Link Poster'], use_container_width=True)
                st.caption(row['Tên phim'])

    elif menu == "Tìm kiếm Phim":
        st.header("🔍 Tìm kiếm Phim")
        search_query = st.text_input("Nhập tên phim:", "")
        if search_query:
            results = search_movie_func(search_query)
            if not results.empty:
                m = results.iloc[0]
                c1, c2 = st.columns([1, 2])
                with c1: st.image(m['Link Poster'])
                with c2:
                    st.subheader(m['Tên phim'])
                    st.write(f"**Thể loại:** {m['Thể loại phim']}")
                    st.write(m['Mô tả'])
                
                st.markdown("---")
                st.subheader("Phim tương tự:")
                sims = get_ai_recommendations([m['Tên phim']], top_k=5, w_sim=1.0, w_pop=0.0)
                scols = st.columns(5)
                for i, (idx, r) in enumerate(sims.iterrows()):
                    with scols[i]:
                        st.image(r['Link Poster'], use_container_width=True)
            else:
                st.warning("Không tìm thấy.")

    elif menu == "Theo Thể loại Yêu thích":
        st.header("❤️ Thể loại Yêu thích")
        fav_movie = st.session_state.current_user['Phim yêu thích nhất']
        
        # Tìm thể loại từ phim yêu thích
        row = movies_df[movies_df['Tên phim'] == fav_movie]
        if not row.empty:
            genres = [x.strip() for x in row.iloc[0]['Thể loại phim'].split(',')]
            st.info(f"Dựa trên phim yêu thích '{fav_movie}', bạn thích: {', '.join(genres)}")
            recs = get_genre_recommendations(genres)
            cols = st.columns(5)
            for i, (idx, r) in enumerate(recs.iterrows()):
                with cols[i % 5]:
                    st.image(r['Link Poster'], use_container_width=True)
                    st.caption(r['Tên phim'])
        else:
            st.error(f"Không tìm thấy dữ liệu về phim '{fav_movie}'.")

    elif menu == "Thống kê Cá nhân":
        st.header("📊 Biểu đồ Sở thích")
        draw_user_charts(user_history)

# 3. CHỨC NĂNG - GUEST / REGISTER
elif st.session_state.user_mode in ['guest', 'register']:
    genres = st.session_state.user_genres
    
    if menu == "Đề xuất AI (Cơ bản)":
        st.header("✨ Đề xuất Phim (Theo lựa chọn)")
        st.write(f"Thể loại quan tâm: {', '.join(genres)}")
        recs = get_genre_recommendations(genres)
        cols = st.columns(5)
        for i, (idx, r) in enumerate(recs.iterrows()):
            with cols[i % 5]:
                st.image(r['Link Poster'], use_container_width=True)
                st.caption(r['Tên phim'])
                
    elif menu == "Theo Thể loại Đã chọn":
        st.header("📂 Lọc chi tiết")
        sub = st.selectbox("Chọn 1 thể loại cụ thể:", genres)
        if sub:
            recs = get_genre_recommendations([sub])
            cols = st.columns(5)
            for i, (idx, r) in enumerate(recs.iterrows()):
                with cols[i % 5]:
                    st.image(r['Link Poster'], use_container_width=True)
                    st.caption(r['Tên phim'])
