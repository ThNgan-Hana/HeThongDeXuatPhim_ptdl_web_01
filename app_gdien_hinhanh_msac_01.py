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

# Custom CSS cho giao diện đẹp hơn
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
# 2. HÀM TIỀN XỬ LÝ DỮ LIỆU (QUAN TRỌNG)
# ==============================================================================
@st.cache_resource
def load_and_process_data():
    # Load data
    movies = pd.read_csv("data_phim_full_images.csv")
    users = pd.read_csv("danh_sach_nguoi_dung_moi.csv")

    # --- Xử lý dữ liệu Movies ---
    # 1. Điền giá trị trống
    movies['Đạo diễn'] = movies['Đạo diễn'].fillna('')
    movies['Thể loại phim'] = movies['Thể loại phim'].fillna('')
    movies['Mô tả'] = movies['Mô tả'].fillna('')
    
    # 2. Tạo cột đặc trưng kết hợp (Combined Features) cho AI
    # Kết hợp Tên phim + Đạo diễn + Thể loại
    movies['combined_features'] = (
        movies['Tên phim'] + " " + 
        movies['Đạo diễn'] + " " + 
        movies['Thể loại phim']
    )

    # 3. Chuẩn hóa độ phổ biến (Scaling Popularity) về khoảng 0-1
    # Để có thể cộng trọng số với điểm cosine similarity (vốn cũng là 0-1)
    scaler = MinMaxScaler()
    movies['popularity_scaled'] = scaler.fit_transform(movies[['Độ phổ biến']])

    # 4. Tạo ma trận TF-IDF
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies['combined_features'])

    # 5. Tính ma trận tương đồng Cosine
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # --- Xử lý dữ liệu Users ---
    # Chuyển chuỗi list "['Phim A', 'Phim B']" thành list Python thật
    users['history_list'] = users['5 phim coi gần nhất'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else [])

    # Lấy danh sách tất cả thể loại để dùng cho Dropdown
    all_genres = set()
    for genres in movies['Thể loại phim']:
        for g in genres.split(','):
            all_genres.add(g.strip())
    
    return movies, users, cosine_sim, sorted(list(all_genres))

# Gọi hàm load dữ liệu
movies_df, users_df, cosine_sim, ALL_GENRES = load_and_process_data()

# ==============================================================================
# 3. CÁC HÀM CHỨC NĂNG CỐT LÕI (ALGORITHMS)
# ==============================================================================

def get_ai_recommendations(history_titles, top_k=10, w_sim=0.7, w_pop=0.3):
    """
    Chức năng 1: Đề xuất AI dựa trên trọng số (Similarity + Popularity)
    """
    # Tìm index của các phim trong lịch sử xem
    indices = []
    for title in history_titles:
        idx = movies_df[movies_df['Tên phim'] == title].index
        if not idx.empty:
            indices.append(idx[0])
    
    if not indices:
        return movies_df.sort_values(by='Độ phổ biến', ascending=False).head(top_k)

    # Tính điểm tương đồng trung bình với lịch sử xem
    # Lấy các dòng tương ứng với phim đã xem trong ma trận cosine, sau đó tính trung bình dọc
    sim_scores = np.mean(cosine_sim[indices], axis=0)
    
    # Lấy điểm phổ biến đã chuẩn hóa
    pop_scores = movies_df['popularity_scaled'].values
    
    # Tính điểm lai (Hybrid Score)
    final_scores = (w_sim * sim_scores) + (w_pop * pop_scores)
    
    # Sắp xếp và lấy top k (loại bỏ phim đã xem)
    # Tạo list tuple (index, score)
    scores_with_idx = list(enumerate(final_scores))
    scores_with_idx = sorted(scores_with_idx, key=lambda x: x[1], reverse=True)
    
    rec_indices = [i[0] for i in scores_with_idx if i[0] not in indices][:top_k]
    return movies_df.iloc[rec_indices]

def search_movie_func(query):
    """
    Chức năng 2: Tìm kiếm phim và gợi ý tương tự
    """
    # Tìm kiếm gần đúng (chứa chuỗi)
    result = movies_df[movies_df['Tên phim'].str.contains(query, case=False, na=False)]
    return result

def get_genre_recommendations(selected_genres, top_k=10):
    """
    Chức năng 3: Đề xuất dựa trên thể loại
    """
    if not selected_genres:
        return pd.DataFrame()
    
    # Lọc các phim có chứa ÍT NHẤT 1 trong các thể loại đã chọn
    # Tạo regex pattern ví dụ: "Hành động|Hài"
    pattern = '|'.join(selected_genres)
    filtered = movies_df[movies_df['Thể loại phim'].str.contains(pattern, case=False, na=False)]
    
    if filtered.empty:
        return pd.DataFrame()
    
    # Sắp xếp theo độ phổ biến để gợi ý phim hay nhất trong thể loại đó
    return filtered.sort_values(by='Độ phổ biến', ascending=False).head(top_k)

def draw_user_charts(history_titles):
    """
    Vẽ biểu đồ thống kê xu hướng xem phim
    """
    if not history_titles:
        st.warning("Chưa có dữ liệu lịch sử để vẽ biểu đồ.")
        return

    # Lấy danh sách thể loại từ các phim đã xem
    genres_count = []
    for title in history_titles:
        movie_row = movies_df[movies_df['Tên phim'] == title]
        if not movie_row.empty:
            g_str = movie_row.iloc[0]['Thể loại phim']
            g_list = [x.strip() for x in g_str.split(',')]
            genres_count.extend(g_list)
    
    if not genres_count:
        st.warning("Không tìm thấy thông tin thể loại.")
        return

    # Đếm số lượng
    counts = Counter(genres_count)
    df_chart = pd.DataFrame.from_dict(counts, orient='index', columns=['Count']).reset_index()
    df_chart.columns = ['Thể loại', 'Số phim đã xem']
    df_chart = df_chart.sort_values(by='Số phim đã xem', ascending=False)

    # Vẽ biểu đồ
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Biểu đồ 1: Pie Chart (Phân bố)
    ax1.pie(df_chart['Số phim đã xem'], labels=df_chart['Thể loại'], autopct='%1.1f%%', startangle=90, colors=sns.color_palette('pastel'))
    ax1.set_title('Phân bố thể loại đã xem')

    # Biểu đồ 2: Bar Chart (Số lượng)
    sns.barplot(x='Số phim đã xem', y='Thể loại', data=df_chart, ax=ax2, palette='viridis')
    ax2.set_title('Số lượng phim theo thể loại')
    
    st.pyplot(fig)

# ==============================================================================
# 4. GIAO DIỆN NGƯỜI DÙNG (UI)
# ==============================================================================

# --- Session State Management ---
if 'user_mode' not in st.session_state:
    st.session_state.user_mode = None  # 'member', 'guest', 'register'
if 'current_user' not in st.session_state:
    st.session_state.current_user = None
if 'user_genres' not in st.session_state: # Cho Guest/Register
    st.session_state.user_genres = []

# --- Sidebar ---
with st.sidebar:
    st.title("🎬 DreamStream")
    st.write("Hệ thống gợi ý phim thông minh")
    
    if st.session_state.user_mode == 'member':
        st.success(f"Xin chào, {st.session_state.current_user['Tên người dùng']}!")
        menu = st.radio("Chức năng", ["Đề xuất AI", "Tìm kiếm Phim", "Theo Thể loại Yêu thích", "Thống kê Cá nhân"])
        if st.button("Đăng xuất"):
            st.session_state.user_mode = None
            st.session_state.current_user = None
            st.rerun()
            
    elif st.session_state.user_mode in ['guest', 'register']:
        st.info(f"Chế độ: {st.session_state.user_mode.upper()}")
        menu = st.radio("Chức năng", ["Đề xuất AI (Cơ bản)", "Theo Thể loại Đã chọn"])
        if st.button("Thoát chế độ Khách"):
            st.session_state.user_mode = None
            st.session_state.user_genres = []
            st.rerun()
            
    else:
        st.warning("Vui lòng đăng nhập hoặc chọn chế độ khách.")
        menu = "Login"

# --- Main Content ---

# 1. MÀN HÌNH LOGIN / REGISTER
if st.session_state.user_mode is None:
    tab1, tab2, tab3 = st.tabs(["Đăng nhập Thành viên", "Đăng ký Mới", "Chế độ Khách"])
    
    with tab1: # Login
        username = st.text_input("Tên đăng nhập")
        if st.button("Đăng nhập"):
            user_row = users_df[users_df['Tên người dùng'] == username]
            if not user_row.empty:
                st.session_state.user_mode = 'member'
                st.session_state.current_user = user_row.iloc[0]
                st.toast("Đăng nhập thành công!", icon="✅")
                st.rerun()
            else:
                st.error("Tên người dùng không tồn tại.")

    with tab2: # Register
        new_user = st.text_input("Tạo tên người dùng mới")
        selected_g = st.multiselect("Chọn thể loại bạn thích:", ALL_GENRES, key='reg_genres')
        if st.button("Đăng ký & Vào ngay"):
            if new_user and selected_g:
                st.session_state.user_mode = 'register'
                st.session_state.current_user = {'Tên người dùng': new_user}
                st.session_state.user_genres = selected_g
                st.rerun()
            else:
                st.warning("Vui lòng nhập tên và chọn ít nhất 1 thể loại.")

    with tab3: # Guest
        guest_g = st.multiselect("Chọn thể loại muốn xem:", ALL_GENRES, key='guest_genres')
        if st.button("Truy cập ngay"):
            if guest_g:
                st.session_state.user_mode = 'guest'
                st.session_state.user_genres = guest_g
                st.rerun()
            else:
                st.warning("Vui lòng chọn ít nhất 1 thể loại.")

# 2. CHỨC NĂNG DÀNH CHO THÀNH VIÊN CŨ
elif st.session_state.user_mode == 'member':
    user_history = st.session_state.current_user['history_list']
    
    if menu == "Đề xuất AI":
        st.header(f"🤖 Đề xuất Phim Thông minh cho {st.session_state.current_user['Tên người dùng']}")
        st.write("Dựa trên sự kết hợp giữa **lịch sử xem** và **độ phổ biến** của phim.")
        
        st.subheader("Lịch sử xem gần nhất của bạn:")
        st.write(", ".join(user_history))
        
        st.markdown("---")
        st.subheader("Gợi ý dành riêng cho bạn:")
        
        recs = get_ai_recommendations(user_history)
        
        # Hiển thị kết quả dạng lưới
        cols = st.columns(5)
        for i, (idx, row) in enumerate(recs.iterrows()):
            with cols[i % 5]:
                st.image(row['Link Poster'], use_container_width=True)
                st.caption(f"**{row['Tên phim']}**")
                with st.expander("Chi tiết"):
                    st.write(f"⭐ {row['Độ phổ biến']:.1f}")
                    st.write(f"🎭 {row['Thể loại phim']}")

    elif menu == "Tìm kiếm Phim":
        st.header("🔍 Tìm kiếm Phim")
        search_query = st.text_input("Nhập tên phim cần tìm:", "")
        
        if search_query:
            results = search_movie_func(search_query)
            if not results.empty:
                selected_movie = results.iloc[0] # Lấy kết quả đầu tiên
                
                c1, c2 = st.columns([1, 2])
                with c1:
                    st.image(selected_movie['Link Poster'], width=300)
                with c2:
                    st.title(selected_movie['Tên phim'])
                    st.write(f"**Đạo diễn:** {selected_movie['Đạo diễn']}")
                    st.write(f"**Thể loại:** {selected_movie['Thể loại phim']}")
                    st.write(f"**Mô tả:** {selected_movie['Mô tả']}")
                    
                st.markdown("---")
                st.subheader("Phim tương tự (Similar Movies):")
                # Tìm phim tương tự dựa trên content của phim đang xem
                sim_recs = get_ai_recommendations([selected_movie['Tên phim']], top_k=5, w_sim=1.0, w_pop=0.0)
                
                cols = st.columns(5)
                for i, (idx, row) in enumerate(sim_recs.iterrows()):
                    with cols[i]:
                        st.image(row['Link Poster'], use_container_width=True)
                        st.caption(row['Tên phim'])
            else:
                st.warning("Không tìm thấy phim nào.")

    elif menu == "Theo Thể loại Yêu thích":
        st.header("❤️ Đề xuất theo Thể loại Yêu thích")
        # Với user cũ, lấy từ cột Phim yêu thích nhất để suy ra thể loại, hoặc dùng lịch sử
        fav_movie = st.session_state.current_user['Phim yêu thích nhất']
        st.write(f"Phim yêu thích nhất của bạn: **{fav_movie}**")
        
        # Lấy thể loại của phim yêu thích này
        row = movies_df[movies_df['Tên phim'] == fav_movie]
        if not row.empty:
            genres_str = row.iloc[0]['Thể loại phim']
            fav_genres = [x.strip() for x in genres_str.split(',')]
            
            st.info(f"Hệ thống xác định thể loại yêu thích của bạn là: **{', '.join(fav_genres)}**")
            
            recs = get_genre_recommendations(fav_genres)
            cols = st.columns(5)
            for i, (idx, r) in enumerate(recs.iterrows()):
                with cols[i % 5]:
                    st.image(r['Link Poster'], use_container_width=True)
                    st.caption(r['Tên phim'])
        else:
            st.error("Không tìm thấy thông tin phim yêu thích trong dữ liệu.")

    elif menu == "Thống kê Cá nhân":
        st.header("📊 Thống kê Xu hướng Xem phim")
        draw_user_charts(user_history)

# 3. CHỨC NĂNG DÀNH CHO KHÁCH / NGƯỜI ĐĂNG KÝ
elif st.session_state.user_mode in ['guest', 'register']:
    
    selected_g = st.session_state.user_genres
    
    if menu == "Đề xuất AI (Cơ bản)":
        # Với khách, AI sẽ dựa trên thể loại họ chọn + độ phổ biến
        st.header("✨ Đề xuất Phim (Dựa trên lựa chọn của bạn)")
        st.write(f"Các thể loại bạn quan tâm: {', '.join(selected_g)}")
        
        # Dùng hàm genre recommendation nhưng hiển thị dưới dạng 'AI' (phổ biến nhất trong thể loại đó)
        recs = get_genre_recommendations(selected_g, top_k=10)
        
        cols = st.columns(5)
        for i, (idx, row) in enumerate(recs.iterrows()):
            with cols[i % 5]:
                st.image(row['Link Poster'], use_container_width=True)
                st.caption(row['Tên phim'])
                
    elif menu == "Theo Thể loại Đã chọn":
        st.header("📂 Duyệt phim theo Thể loại")
        # Cho phép lọc kỹ hơn trong các thể loại đã chọn
        sub_genre = st.selectbox("Chọn cụ thể:", selected_g)
        if sub_genre:
            recs = get_genre_recommendations([sub_genre], top_k=10)
            cols = st.columns(5)
            for i, (idx, row) in enumerate(recs.iterrows()):
                with cols[i % 5]:
                    st.image(row['Link Poster'], use_container_width=True)
                    st.caption(row['Tên phim'])

