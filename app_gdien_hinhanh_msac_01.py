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
# 1. CẤU HÌNH TRANG & CSS (GIAO DIỆN NÂNG CAO)
# ==============================================================================
st.set_page_config(
    page_title="DreamStream - Movie AI",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS cho giao diện đẹp, hiện đại (Netflix Style)
st.markdown("""
<style>
    /* Import Font hiện đại */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }

    /* 1. Tiêu đề Gradient */
    h1 {
        background: linear-gradient(to right, #ff4b4b, #ff9068);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800 !important;
        margin-bottom: 0px;
    }

    /* 2. Nút bấm (Button) - Hiệu ứng Glow */
    .stButton>button {
        width: 100%;
        border-radius: 12px;
        height: 3em;
        background: linear-gradient(90deg, #ff4b4b 0%, #ff416c 100%);
        color: white;
        border: none;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(255, 75, 75, 0.3);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(255, 75, 75, 0.5);
        background: linear-gradient(90deg, #ff416c 0%, #ff4b4b 100%);
    }

    /* 3. Poster Phim - Hiệu ứng Zoom & Shadow */
    div[data-testid="stImage"] img {
        border-radius: 12px;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    div[data-testid="stImage"] img:hover {
        transform: scale(1.05);
        box-shadow: 0 12px 24px rgba(0,0,0,0.6);
        z-index: 10;
        cursor: pointer;
    }

    /* 4. Expander (Nút xem chi tiết) */
    .streamlit-expanderHeader {
        background-color: #1f1f1f; /* Màu nền tối hơn */
        border-radius: 8px;
        font-size: 0.9em;
        color: #e0e0e0;
    }
    
    /* 5. Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #161616; /* Sidebar tối màu */
    }
    
    /* 6. Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        border-radius: 8px 8px 0 0;
        background-color: #262730;
        padding: 0 20px;
        border: none;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ff4b4b !important;
        color: white !important;
    }

    /* Tùy chỉnh thanh cuộn cho gọn */
    ::-webkit-scrollbar {
        width: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #0e1117; 
    }
    ::-webkit-scrollbar-thumb {
        background: #555; 
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #888; 
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
    """
    Chức năng 1: Đề xuất dựa trên lịch sử xem (Content-based Filtering)
    """
    # 1. Tìm index phim đã xem trong dữ liệu
    indices = []
    for title in history_titles:
        idx = movies_df[movies_df['Tên phim'] == title].index
        if not idx.empty:
            indices.append(idx[0])
    
    # 2. Xử lý danh sách loại trừ (nếu có)
    if exclude is None: exclude = []
    
    # Nếu chưa xem phim nào -> Gợi ý theo độ phổ biến (trừ những phim đã hiện)
    if not indices:
        popular_movies = movies_df.drop(exclude, errors='ignore').sort_values(by='Độ phổ biến', ascending=False)
        recs = popular_movies.head(top_k)
        return recs, recs.index.tolist()

    # 3. Tính toán điểm số đề xuất (AI)
    # Lấy trung bình độ tương đồng của các phim đã xem với tất cả phim còn lại
    sim_scores = np.mean(cosine_sim[indices], axis=0)
    
    # Lấy điểm độ phổ biến
    pop_scores = movies_df['popularity_scaled'].values
    
    # Tính điểm tổng hợp: (Trọng số Sim * Điểm Sim) + (Trọng số Pop * Điểm Pop)
    final_scores = (w_sim * sim_scores) + (w_pop * pop_scores)
    
    # Tạo danh sách (index, score) và sắp xếp giảm dần
    scores_with_idx = list(enumerate(final_scores))
    scores_with_idx = sorted(scores_with_idx, key=lambda x: x[1], reverse=True)
    
    # 4. Lọc kết quả (Bỏ phim đã xem và phim nằm trong danh sách loại trừ)
    final_indices = []
    for i, score in scores_with_idx:
        # i không nằm trong danh sách đã xem (indices) VÀ không nằm trong danh sách loại trừ (exclude)
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


def get_genre_recommendations(selected_genres, top_k=10, exclude=None):
    """
    Chức năng 3: Đề xuất dựa trên thể loại (Có loại trừ phim đã xem)
    """
    if not selected_genres:
        return pd.DataFrame()
    
    # 1. Xử lý danh sách loại trừ (nếu có)
    if exclude is None:
        exclude = []

    # 2. Lọc các phim theo thể loại
    pattern = '|'.join(selected_genres)
    filtered = movies_df[movies_df['Thể loại phim'].str.contains(pattern, case=False, na=False)]
    
    # 3. Loại bỏ các phim nằm trong danh sách exclude
    if exclude:
        filtered = filtered.drop(exclude, errors='ignore')

    if filtered.empty:
        return pd.DataFrame()

    # 4. Trả về top phim phổ biến nhất còn lại
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
                
                # Hiển thị kết quả tìm kiếm
                cols = st.columns(5)
                for i, (idx, row) in enumerate(results.iterrows()):
                    with cols[i % 5]:
                        st.image(row['Link Poster'], use_container_width=True)
                        st.write(f"**{row['Tên phim']}**")
                        with st.expander("ℹ️ Chi tiết"):
                            st.write(f"🎬 {row['Đạo diễn']}")
                            st.write(f"🏷️ {row['Thể loại phim']}")
                            st.caption(row['Mô tả'][:100])

                # --- PHẦN MỚI: GỢI Ý TƯƠNG TỰ ---
                st.markdown("---")
                st.subheader("💡 Có thể bạn cũng thích (Tương tự kết quả đầu tiên):")
                
                # Lấy phim đầu tiên trong kết quả tìm kiếm để làm gốc
                first_movie = results.iloc[0]
                first_movie_genres = [g.strip() for g in first_movie['Thể loại phim'].split(',')]
                
                # Tìm phim tương tự (loại trừ chính những phim vừa tìm thấy)
                similar_recs = get_genre_recommendations(
                    first_movie_genres, 
                    top_k=5, 
                    exclude=results.index.tolist() # Không hiện lại phim vừa tìm
                )
                
                if not similar_recs.empty:
                    cols_sim = st.columns(5)
                    for i, (idx, row) in enumerate(similar_recs.iterrows()):
                        with cols_sim[i % 5]:
                            st.image(row['Link Poster'], use_container_width=True)
                            st.write(f"**{row['Tên phim']}**")
                            with st.expander("Xem thêm"):
                                st.caption(f"Thể loại: {row['Thể loại phim']}")
                else:
                    st.info("Không tìm thấy phim tương tự khác.")
            else:
                st.warning("Không tìm thấy phim nào khớp với từ khóa.")
    

    # --- 3. MENU THEO THỂ LOẠI YÊU THÍCH ---
    elif menu == "Theo Thể loại Yêu thích":
        st.header("❤️ Đề xuất theo Thể loại Yêu thích")
        
        fav_movie = st.session_state.current_user.get('Phim yêu thích nhất', '')
        
        if fav_movie:
            st.write(f"Phim tâm đắc nhất của bạn: **{fav_movie}**")
            
            # Lấy thông tin phim yêu thích
            row = movies_df[movies_df['Tên phim'] == fav_movie]
            if not row.empty:
                genres_str = row.iloc[0]['Thể loại phim']
                fav_genres = [x.strip() for x in genres_str.split(',')]
                st.info(f"Thể loại ưa thích xác định được: **{', '.join(fav_genres)}**")
                
                # --- LOGIC STATE CHO MEMBER (Giống Guest) ---
                if 'member_fav_seen' not in st.session_state:
                    st.session_state.member_fav_seen = [] # Danh sách ID đã xem
                if 'member_fav_recs' not in st.session_state:
                    st.session_state.member_fav_recs = None # DataFrame đang hiện

                # Nút làm mới
                col_btn, _ = st.columns([1, 4])
                is_refresh = col_btn.button("🔄 Làm mới danh sách")
                
                # Logic tải dữ liệu: Chạy khi (Bấm nút) HOẶC (Chưa có dữ liệu)
                if is_refresh or st.session_state.member_fav_recs is None:
                    new_recs = get_genre_recommendations(
                        fav_genres, 
                        top_k=10, 
                        exclude=st.session_state.member_fav_seen
                    )
                    
                    if not new_recs.empty:
                        st.session_state.member_fav_recs = new_recs
                        st.session_state.member_fav_seen.extend(new_recs.index.tolist())
                        if is_refresh: st.success("Đã cập nhật phim mới!")
                    else:
                        st.warning("Đã hiển thị hết các phim nổi bật trong thể loại này.")
                
                # Hiển thị từ State
                if st.session_state.member_fav_recs is not None and not st.session_state.member_fav_recs.empty:
                    cols = st.columns(5)
                    for i, (idx, r) in enumerate(st.session_state.member_fav_recs.iterrows()):
                        with cols[i % 5]:
                            st.image(r['Link Poster'], use_container_width=True)
                            st.write(f"**{r['Tên phim']}**")
                            with st.expander("ℹ️ Chi tiết"):
                                st.write(f"🎬 {r['Đạo diễn']}")
                                st.write(f"⭐ {round(r['Độ phổ biến'], 1)}")
                                st.caption(r['Mô tả'][:100])
            else:
                st.error("Không tìm thấy thông tin phim yêu thích trong dữ liệu gốc.")
        else:
            st.warning("Bạn chưa cập nhật phim yêu thích trong hồ sơ.")

    # --- 4. MENU THỐNG KÊ ---
    elif menu == "Thống kê Cá nhân":
        st.header("📊 Thống kê Xu hướng Xem phim")
        draw_user_charts(user_history)


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





