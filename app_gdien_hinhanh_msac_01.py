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
# 1. CẤU HÌNH TRANG & CSS (GIAO DIỆN NETFLIX STYLE)
# ==============================================================================
st.set_page_config(
    page_title="DreamStream",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS: Dark Theme & Netflix Style
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700;900&display=swap');

    /* 1. CẤU HÌNH CHUNG */
    html, body, [class*="css"] {
        font-family: 'Roboto', sans-serif;
        background-color: #141414;
        color: #ffffff;
    }
    .stApp {
        background-color: #141414;
    }

    /* 2. SIDEBAR */
    section[data-testid="stSidebar"] {
        background-color: #000000;
        border-right: 1px solid #333;
    }
    
    /* 3. TIÊU ĐỀ */
    h1, h2, h3 {
        color: white !important;
        font-weight: 700;
    }

    /* 4. NÚT BẤM (BUTTON) */
    .stButton>button {
        background-color: #E50914;
        color: white;
        border: none;
        border-radius: 4px;
        height: 3em;
        font-weight: bold;
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        background-color: #f40612;
        transform: scale(1.02);
    }

    /* 5. POSTER PHIM */
    div[data-testid="stImage"] img {
        border-radius: 4px;
        transition: transform 0.3s ease;
    }
    div[data-testid="stImage"] img:hover {
        transform: scale(1.08);
        z-index: 10;
        cursor: pointer;
        box-shadow: 0 10px 20px rgba(0,0,0,0.8);
    }

    /* 6. INPUT FORM */
    .stTextInput>div>div>input {
        background-color: #333;
        color: white;
        border: 1px solid #555;
    }
    .stSelectbox>div>div>div {
        background-color: #333;
        color: white;
    }
    
    /* 7. TABS */
    .stTabs [aria-selected="true"] {
        color: #E50914 !important;
        border-bottom-color: #E50914 !important;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. HÀM TIỀN XỬ LÝ DỮ LIỆU
# ==============================================================================
@st.cache_resource
def load_and_process_data():
    # Load data
    movies = pd.read_csv("data_phim_full_images.csv")
    users = pd.read_csv("danh_sach_nguoi_dung_gia_lap.csv")

    # --- Xử lý dữ liệu Movies ---
    movies['Đạo diễn'] = movies['Đạo diễn'].fillna('')
    movies['Thể loại phim'] = movies['Thể loại phim'].fillna('')
    movies['Mô tả'] = movies['Mô tả'].fillna('')
    
    # Tạo cột đặc trưng kết hợp
    movies['combined_features'] = (
        movies['Tên phim'] + " " + 
        movies['Đạo diễn'] + " " + 
        movies['Thể loại phim']
    )

    # Chuẩn hóa độ phổ biến
    scaler = MinMaxScaler()
    movies['popularity_scaled'] = scaler.fit_transform(movies[['Độ phổ biến']])

    # Tạo ma trận TF-IDF
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies['combined_features'])

    # Tính ma trận tương đồng Cosine
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # --- Xử lý dữ liệu Users ---
    users['history_list'] = users['5 phim coi gần nhất'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else [])

    # Lấy danh sách thể loại
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
    
    # 2. Xử lý loại trừ
    if exclude is None: exclude = []
    
    if not indices:
        popular_movies = movies_df.drop(exclude, errors='ignore').sort_values(by='Độ phổ biến', ascending=False)
        recs = popular_movies.head(top_k)
        return recs, recs.index.tolist()

    # 3. Tính toán AI
    sim_scores = np.mean(cosine_sim[indices], axis=0)
    pop_scores = movies_df['popularity_scaled'].values
    final_scores = (w_sim * sim_scores) + (w_pop * pop_scores)
    
    scores_with_idx = list(enumerate(final_scores))
    scores_with_idx = sorted(scores_with_idx, key=lambda x: x[1], reverse=True)
    
    # 4. Lọc kết quả
    final_indices = []
    for i, score in scores_with_idx:
        if i not in indices and i not in exclude:
            final_indices.append(i)
            if len(final_indices) >= top_k:
                break
    
    return movies_df.iloc[final_indices], final_indices

def search_movie_func(query):
    result = movies_df[movies_df['Tên phim'].str.contains(query, case=False, na=False)]
    return result

def get_genre_recommendations(selected_genres, top_k=10, exclude=None):
    if not selected_genres:
        return pd.DataFrame()
    
    if exclude is None:
        exclude = []

    pattern = '|'.join(selected_genres)
    filtered = movies_df[movies_df['Thể loại phim'].str.contains(pattern, case=False, na=False)]
    
    if exclude:
        filtered = filtered.drop(exclude, errors='ignore')

    if filtered.empty:
        return pd.DataFrame()

    return filtered.sort_values(by='Độ phổ biến', ascending=False).head(top_k)

def draw_user_charts(history_titles):
    if not history_titles:
        st.warning("Chưa có dữ liệu lịch sử.")
        return

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

    counts = Counter(genres_count)
    df_chart = pd.DataFrame.from_dict(counts, orient='index', columns=['Count']).reset_index()
    df_chart.columns = ['Thể loại', 'Số phim đã xem']
    df_chart = df_chart.sort_values(by='Số phim đã xem', ascending=False)

    tab1, tab2 = st.tabs(["Biểu đồ Tròn", "Biểu đồ Cột"])

    with tab1:
        fig1, ax1 = plt.subplots(figsize=(10, 6), facecolor='none')
        ax1.pie(df_chart['Số phim đã xem'], labels=df_chart['Thể loại'], autopct='%1.1f%%', 
                startangle=90, colors=sns.color_palette('pastel'), textprops={'color':"w"})
        ax1.set_title('Phân bố thể loại', color='white')
        st.pyplot(fig1)

    with tab2:
        fig2, ax2 = plt.subplots(figsize=(10, 6), facecolor='none')
        sns.barplot(x='Số phim đã xem', y='Thể loại', data=df_chart, ax=ax2, palette='viridis')
        ax2.set_title('Số lượng phim', color='white')
        ax2.tick_params(colors='white')
        ax2.set_xlabel('Số lượng', color='white')
        ax2.set_ylabel('Thể loại', color='white')
        ax2.set_facecolor('none')
        st.pyplot(fig2)

# ==============================================================================
# 4. GIAO DIỆN NGƯỜI DÙNG (UI)
# ==============================================================================

if 'user_mode' not in st.session_state:
    st.session_state.user_mode = None
if 'current_user' not in st.session_state:
    st.session_state.current_user = None
if 'user_genres' not in st.session_state:
    st.session_state.user_genres = []

# --- Sidebar ---
with st.sidebar:
    st.title("DreamStream")
    
    if st.session_state.user_mode == 'member':
        st.success(f"Chào, {st.session_state.current_user['Tên người dùng']}!")
        menu = st.radio("Menu", ["Đề xuất AI", "Tìm kiếm Phim", "Theo Thể loại Yêu thích", "Thống kê Cá nhân"])
        if st.button("Đăng xuất"):
            st.session_state.user_mode = None
            st.session_state.current_user = None
            st.rerun()
            
    elif st.session_state.user_mode in ['guest', 'register']:
        st.info(f"Chế độ: {st.session_state.user_mode}")
        menu = st.radio("Menu", ["Theo Thể loại Đã chọn"])
        if st.button("Thoát"):
            st.session_state.user_mode = None
            st.session_state.user_genres = []
            st.rerun()
    else:
        st.warning("Vui lòng đăng nhập.")
        menu = "Login"

# --- Main Content ---
if st.session_state.user_mode is None:
    tab1, tab2, tab3 = st.tabs(["Đăng nhập", "Đăng ký", "Khách"])
    
    with tab1:
        username = st.text_input("Tên đăng nhập")
        if st.button("Login"):
            user_row = users_df[users_df['Tên người dùng'] == username]
            if not user_row.empty:
                st.session_state.user_mode = 'member'
                st.session_state.current_user = user_row.iloc[0]
                st.rerun()
            else:
                st.error("Sai tên đăng nhập.")

    with tab2:
        new_user = st.text_input("Tên mới")
        selected_g = st.multiselect("Sở thích:", ALL_GENRES)
        if st.button("Đăng ký"):
            if new_user and selected_g:
                st.session_state.user_mode = 'register'
                st.session_state.current_user = {'Tên người dùng': new_user}
                st.session_state.user_genres = selected_g
                st.rerun()

    with tab3:
        guest_g = st.multiselect("Chọn thể loại:", ALL_GENRES)
        if st.button("Vào ngay"):
            if guest_g:
                st.session_state.user_mode = 'guest'
                st.session_state.user_genres = guest_g
                st.rerun()

elif st.session_state.user_mode == 'member':
    user_history = st.session_state.current_user['history_list']
    
    if menu == "Đề xuất AI":
        st.header(f"🤖 Đề xuất cho {st.session_state.current_user['Tên người dùng']}")
        st.info("Lịch sử: " + ", ".join(user_history))
        
        if 'ai_seen' not in st.session_state: st.session_state.ai_seen = []
            
        if st.button("🔄 Làm mới"):
            recs, idxs = get_ai_recommendations(user_history, exclude=st.session_state.ai_seen)
            if idxs: st.session_state.ai_seen.extend(idxs)
        else:
            recs, idxs = get_ai_recommendations(user_history, exclude=st.session_state.ai_seen)
            if not st.session_state.ai_seen: st.session_state.ai_seen.extend(idxs)

        if not recs.empty:
            cols = st.columns(5)
            for i, (idx, row) in enumerate(recs.iterrows()):
                with cols[i % 5]:
                    st.image(row['Link Poster'], use_container_width=True)
                    st.write(f"**{row['Tên phim']}**")
                    with st.expander("Chi tiết"):
                        st.write(f"⭐ {round(row['Độ phổ biến'], 1)}")
                        st.caption(row['Mô tả'][:100])

    elif menu == "Tìm kiếm Phim":
        st.header("🔍 Tìm kiếm")
        search_query = st.text_input("Nhập tên phim:")
        
        if search_query:
            results = search_movie_func(search_query)
            if not results.empty:
                st.success(f"Tìm thấy {len(results)} phim:")
                cols = st.columns(5)
                for i, (idx, row) in enumerate(results.iterrows()):
                    with cols[i % 5]:
                        st.image(row['Link Poster'], use_container_width=True)
                        st.write(f"**{row['Tên phim']}**")
                        with st.expander("Chi tiết"):
                            st.caption(row['Mô tả'][:100])
                
                # Gợi ý tương tự
                st.markdown("---")
                st.subheader("💡 Có thể bạn cũng thích:")
                first_genres = [g.strip() for g in results.iloc[0]['Thể loại phim'].split(',')]
                sim_recs = get_genre_recommendations(first_genres, top_k=5, exclude=results.index.tolist())
                
                if not sim_recs.empty:
                    cols2 = st.columns(5)
                    for i, (idx, row) in enumerate(sim_recs.iterrows()):
                        with cols2[i % 5]:
                            st.image(row['Link Poster'], use_container_width=True)
                            st.write(f"**{row['Tên phim']}**")

    elif menu == "Theo Thể loại Yêu thích":
        st.header("❤️ Theo sở thích")
        fav = st.session_state.current_user.get('Phim yêu thích nhất', '')
        if fav:
            st.write(f"Phim tâm đắc: **{fav}**")
            row = movies_df[movies_df['Tên phim'] == fav]
            if not row.empty:
                genres = [x.strip() for x in row.iloc[0]['Thể loại phim'].split(',')]
                
                if 'mem_seen' not in st.session_state: st.session_state.mem_seen = []
                if 'mem_recs' not in st.session_state: st.session_state.mem_recs = None
                
                if st.button("🔄 Làm mới danh sách") or st.session_state.mem_recs is None:
                    new_recs = get_genre_recommendations(genres, top_k=10, exclude=st.session_state.mem_seen)
                    if not new_recs.empty:
                        st.session_state.mem_recs = new_recs
                        st.session_state.mem_seen.extend(new_recs.index.tolist())
                
                if st.session_state.mem_recs is not None:
                    cols = st.columns(5)
                    for i, (idx, r) in enumerate(st.session_state.mem_recs.iterrows()):
                        with cols[i % 5]:
                            st.image(r['Link Poster'], use_container_width=True)
                            st.write(f"**{r['Tên phim']}**")
                            with st.expander("Chi tiết"):
                                st.caption(r['Mô tả'][:100])
        else:
            st.warning("Chưa có phim yêu thích.")

    elif menu == "Thống kê Cá nhân":
        draw_user_charts(user_history)

elif st.session_state.user_mode in ['guest', 'register']:
    if menu == "Theo Thể loại Đã chọn":
        st.header("📂 Duyệt phim")
        sub_genre = st.selectbox("Chọn thể loại:", st.session_state.user_genres)
        
        if 'guest_cur' not in st.session_state: st.session_state.guest_cur = None
        if 'guest_seen' not in st.session_state: st.session_state.guest_seen = []
        if 'guest_df' not in st.session_state: st.session_state.guest_df = None

        if sub_genre != st.session_state.guest_cur:
            st.session_state.guest_cur = sub_genre
            st.session_state.guest_seen = []
            st.session_state.guest_df = None

        if st.button("🔄 Làm mới") or st.session_state.guest_df is None:
            new_recs = get_genre_recommendations([sub_genre], top_k=10, exclude=st.session_state.guest_seen)
            if not new_recs.empty:
                st.session_state.guest_df = new_recs
                st.session_state.guest_seen.extend(new_recs.index.tolist())
        
        if st.session_state.guest_df is not None:
            cols = st.columns(5)
            for i, (idx, row) in enumerate(st.session_state.guest_df.iterrows()):
                with cols[i % 5]:
                    st.image(row['Link Poster'], use_container_width=True)
                    st.write(f"**{row['Tên phim']}**")
                    with st.expander("Chi tiết"):
                        st.caption(row['Mô tả'][:100])
