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
    users = pd.read_csv("danh_sach_nguoi_dung_gia_lap.csv")

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

def get_ai_recommendations(history_titles, top_k=10, w_sim=0.7, w_pop=0.3, exclude=None):
    # 1. Tìm index phim đã xem
    indices = []
    for title in history_titles:
        idx = movies_df[movies_df['Tên phim'] == title].index
        if not idx.empty:
            indices.append(idx[0])
    
    # 2. Xử lý khi không có lịch sử hoặc loại trừ
    if exclude is None: exclude = []
    
    if not indices:
        # Lấy top phim phổ biến TRỪ những phim đã hiển thị (exclude)
        popular_movies = movies_df.drop(exclude, errors='ignore').sort_values(by='Độ phổ biến', ascending=False)
        recs = popular_movies.head(top_k)
        return recs, recs.index.tolist()

    # 3. Tính toán đề xuất AI
    sim_scores = np.mean(cosine_sim[indices], axis=0)
    pop_scores = movies_df['popularity_scaled'].values
    final_scores = (w_sim * sim_scores) + (w_pop * pop_scores)
    
    scores_with_idx = list(enumerate(final_scores))
    scores_with_idx = sorted(scores_with_idx, key=lambda x: x[1], reverse=True)
    
    # 4. Lọc kết quả (Bỏ phim đã xem và phim đã hiển thị)
    final_indices = []
    for i, score in scores_with_idx:
        if i not in indices and i not in exclude:
            final_indices.append(i)
            if len(final_indices) >= top_k:
                break
    
    return movies_df.iloc[final_indices], final_indices
    
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

    # --- PHẦN CHỈNH SỬA: TÁCH THÀNH 2 biểu đồ ---
    
    # 1. BIỂU ĐỒ
    tab1, tab2 = st.tabs(["Biểu đồ Tròn (Phân bố)", "Biểu đồ Cột (Số lượng)"])

    # 2. Vẽ biểu đồ tròn
    with tab1:
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        ax1.pie(
            df_chart['Số phim đã xem'], 
            labels=df_chart['Thể loại'], 
            autopct='%1.1f%%', 
            startangle=90, 
            colors=sns.color_palette('pastel')
        )
        ax1.set_title('Phân bố thể loại đã xem')
        ax1.axis('equal')  # Đảm bảo biểu đồ tròn
        st.pyplot(fig1)

    # 3. Vẽ biểu đồ cột
    with tab2:
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        sns.barplot(
            x='Số phim đã xem', 
            y='Thể loại', 
            data=df_chart, 
            ax=ax2, 
            palette='viridis'
        )
        ax2.set_title('Số lượng phim theo thể loại')
        st.pyplot(fig2)

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
        menu = st.radio("Chức năng", ["Theo Thể loại Đã chọn"])
        if st.button("Thoát"):
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
# 2. CHỨC NĂNG DÀNH CHO THÀNH VIÊN CŨ
elif st.session_state.user_mode == 'member':
    # Lấy lịch sử xem
    user_history = st.session_state.current_user['history_list']
    
    # --- 1. MENU ĐỀ XUẤT AI ---
    if menu == "Đề xuất AI":
        st.header(f"🤖 Đề xuất Phim Thông minh cho {st.session_state.current_user['Tên người dùng']}")
        st.write("Dựa trên sự kết hợp giữa **lịch sử xem** và **độ phổ biến** của phim.")
        
        st.subheader("Lịch sử xem gần nhất của bạn:")
        st.info(", ".join(user_history)) # Dùng st.info cho đẹp hơn
        
        st.markdown("---")
        st.subheader("Gợi ý dành riêng cho bạn:")
        
        if 'ai_seen' not in st.session_state:
            st.session_state.ai_seen = []
            
        # Nút làm mới
        if st.button("🔄 Làm mới đề xuất"):
            recs, idxs = get_ai_recommendations(user_history, exclude=st.session_state.ai_seen)
            if idxs:
                st.session_state.ai_seen.extend(idxs)
        else:
            recs, idxs = get_ai_recommendations(user_history, exclude=st.session_state.ai_seen)
            if not st.session_state.ai_seen:
                st.session_state.ai_seen.extend(idxs)

        # HIỂN THỊ KẾT QUẢ
        if not recs.empty:
            cols = st.columns(5)
            for i, (idx, row) in enumerate(recs.iterrows()):
                with cols[i % 5]:
                    st.image(row['Link Poster'], use_container_width=True)
                    st.write(f"**{row['Tên phim']}**")
                    # --- PHẦN THÊM CHI TIẾT ---
                    with st.expander("ℹ️ Xem chi tiết"):
                        st.write(f"🎬 **Đạo diễn:** {row['Đạo diễn']}")
                        st.write(f"🏷️ **Thể loại:** {row['Thể loại phim']}")
                        st.write(f"⭐ **Điểm:** {round(row['Độ phổ biến'], 1)}")
                        st.caption(f"📝 {row['Mô tả'][:150]}...") # Cắt bớt mô tả nếu quá dài

    # --- 2. MENU TÌM KIẾM PHIM ---
    elif menu == "Tìm kiếm Phim":
        st.header("🔍 Tìm kiếm Phim")
        search_query = st.text_input("Nhập tên phim bạn muốn tìm:", placeholder="Ví dụ: Avengers, Harry Potter...")
        
        if search_query:
            results = search_movie_func(search_query)
            if not results.empty:
                st.success(f"Tìm thấy {len(results)} kết quả:")
                cols = st.columns(5)
                for i, (idx, row) in enumerate(results.iterrows()):
                    with cols[i % 5]:
                        st.image(row['Link Poster'], use_container_width=True)
                        st.write(f"**{row['Tên phim']}**")
                        # --- PHẦN THÊM CHI TIẾT ---
                        with st.expander("ℹ️ Xem chi tiết"):
                            st.write(f"🎬 **Đạo diễn:** {row['Đạo diễn']}")
                            st.write(f"🏷️ **Thể loại:** {row['Thể loại phim']}")
                            st.write(f"⭐ **Điểm:** {round(row['Độ phổ biến'], 1)}")
                            st.caption(f"📝 {row['Mô tả']}")
            else:
                st.warning("Không tìm thấy phim nào khớp với từ khóa.")

    # --- 3. MENU THEO THỂ LOẠI YÊU THÍCH ---
    elif menu == "Theo Thể loại Yêu thích":
        st.header("❤️ Đề xuất theo Thể loại Yêu thích")
        
        fav_movie = st.session_state.current_user.get('Phim yêu thích nhất', '')
        
        if fav_movie:
            st.write(f"Phim tâm đắc nhất của bạn: **{fav_movie}**")
            
            row = movies_df[movies_df['Tên phim'] == fav_movie]
            if not row.empty:
                genres_str = row.iloc[0]['Thể loại phim']
                fav_genres = [x.strip() for x in genres_str.split(',')]
                
                st.info(f"Thể loại ưa thích: **{', '.join(fav_genres)}**")
                
                recs = get_genre_recommendations(fav_genres)
                if not recs.empty:
                    cols = st.columns(5)
                    for i, (idx, r) in enumerate(recs.iterrows()):
                        with cols[i % 5]:
                            st.image(r['Link Poster'], use_container_width=True)
                            st.write(f"**{r['Tên phim']}**")
                            # --- PHẦN THÊM CHI TIẾT ---
                            with st.expander("ℹ️ Xem chi tiết"):
                                st.write(f"🎬 **Đạo diễn:** {r['Đạo diễn']}")
                                st.write(f"🏷️ **Thể loại:** {r['Thể loại phim']}")
                                st.write(f"⭐ **Điểm:** {round(r['Độ phổ biến'], 1)}")
                                st.caption(f"📝 {r['Mô tả'][:150]}...")
                else:
                    st.warning("Không tìm thấy đề xuất phù hợp.")
            else:
                st.error("Không tìm thấy thông tin phim yêu thích trong dữ liệu gốc.")
        else:
            st.warning("Bạn chưa cập nhật phim yêu thích trong hồ sơ.")

    # --- 4. MENU THỐNG KÊ ---
    elif menu == "Thống kê Cá nhân":
        st.header("📊 Thống kê Xu hướng Xem phim")
        draw_user_charts(user_history)


# 3. CHỨC NĂNG DÀNH CHO KHÁCH / NGƯỜI ĐĂNG KÝ
# 3. CHỨC NĂNG DÀNH CHO KHÁCH / NGƯỜI ĐĂNG KÝ
elif st.session_state.user_mode in ['guest', 'register']:
    
    selected_g = st.session_state.user_genres
    
    if menu == "Theo Thể loại Đã chọn":
        st.header("📂 Duyệt phim theo Thể loại")
        
        # Selectbox chọn thể loại
        sub_genre = st.selectbox("Chọn cụ thể:", selected_g)
        
        # --- LOGIC QUẢN LÝ TRẠNG THÁI (STATE) ---
        # 1. Khởi tạo các biến nhớ (session_state) nếu chưa có
        if 'guest_current_genre' not in st.session_state:
            st.session_state.guest_current_genre = None # Lưu thể loại đang chọn
        if 'guest_seen_ids' not in st.session_state:
            st.session_state.guest_seen_ids = []        # Lưu danh sách ID phim đã hiện (để tránh lặp)
        if 'guest_recs_df' not in st.session_state:
            st.session_state.guest_recs_df = None       # Lưu DataFrame phim đang hiển thị trên màn hình

        # 2. Kiểm tra: Nếu người dùng đổi sang thể loại khác -> Reset lại từ đầu
        if sub_genre != st.session_state.guest_current_genre:
            st.session_state.guest_current_genre = sub_genre
            st.session_state.guest_seen_ids = []  # Xóa lịch sử đã xem cũ
            st.session_state.guest_recs_df = None # Xóa phim đang hiện cũ
            # (Streamlit sẽ chạy tiếp xuống dưới để tải dữ liệu mới)

        # 3. Xử lý nút "Làm mới" HOẶC Tải lần đầu
        col_btn, col_empty = st.columns([1, 4])
        is_click_refresh = col_btn.button("🔄 Làm mới đề xuất")
        
        # Logic tải dữ liệu chạy khi: (Bấm nút Làm mới) HOẶC (Chưa có phim nào đang hiện)
        if is_click_refresh or st.session_state.guest_recs_df is None:
            if sub_genre:
                # Gọi hàm get_genre_recommendations với tham số exclude
                # để loại bỏ những phim đã nằm trong danh sách guest_seen_ids
                new_recs = get_genre_recommendations(
                    [sub_genre], 
                    top_k=10, 
                    exclude=st.session_state.guest_seen_ids
                )
                
                if not new_recs.empty:
                    # Lưu phim mới vào state để hiển thị
                    st.session_state.guest_recs_df = new_recs
                    # Cập nhật danh sách ID đã xem vào kho lưu trữ
                    st.session_state.guest_seen_ids.extend(new_recs.index.tolist())
                    
                    if is_click_refresh:
                        st.success("Đã làm mới danh sách phim!")
                else:
                    # Nếu không còn phim nào mới để hiện
                    st.warning("Đã hiển thị hết các phim nổi bật thuộc thể loại này!")
        
        # --- 4. HIỂN THỊ DANH SÁCH PHIM TỪ STATE RA MÀN HÌNH ---
        if st.session_state.guest_recs_df is not None and not st.session_state.guest_recs_df.empty:
            cols = st.columns(5)
            for i, (idx, row) in enumerate(st.session_state.guest_recs_df.iterrows()):
                with cols[i % 5]:
                    st.image(row['Link Poster'], use_container_width=True)
                    st.write(f"**{row['Tên phim']}**")
                    
                    # Expander xem chi tiết (Giống giao diện Member)
                    with st.expander("ℹ️ Chi tiết"):
                        st.write(f"🎬 **Đạo diễn:** {row['Đạo diễn']}")
                        st.write(f"🏷️ **Thể loại:** {row['Thể loại phim']}")
                        st.write(f"⭐ **Điểm:** {round(row['Độ phổ biến'], 1)}")
                        st.caption(f"📝 {row['Mô tả'][:100]}...")

