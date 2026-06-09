import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import time

# ==========================
# CONFIG
# ==========================
st.set_page_config(
    page_title="Waste AI Classifier",
    page_icon="♻️",
    layout="centered"
)

# ==========================
# CUSTOM CSS
# ==========================
st.markdown("""
<style>

.stApp {
    background: linear-gradient(180deg,#e8fdf5,#f6fffc);
}

.main-title{
    text-align:center;
    font-size:42px;
    font-weight:bold;
    color:#0f9d58;
}

.subtitle{
    text-align:center;
    color:#555;
    font-size:18px;
}

.card{
    background:white;
    padding:20px;
    border-radius:20px;
    box-shadow:0px 4px 15px rgba(0,0,0,0.08);
    margin-bottom:15px;
}

.result-card{
    background:#ffffff;
    padding:20px;
    border-radius:20px;
    box-shadow:0px 4px 15px rgba(0,0,0,0.08);
}

.center{
    text-align:center;
}

.small-text{
    color:#777;
    text-align:center;
}

</style>
""", unsafe_allow_html=True)

# ==========================
# MODEL
# ==========================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model/waste_model.h5")

model = load_model()

CLASS_NAMES = [
    "cardboard",
    "compost",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash"
]

waste_info = {
    "cardboard": "📦 Kardus dapat didaur ulang menjadi kemasan atau produk kertas baru.",
    "compost": "🌱 Sampah organik dapat diolah menjadi kompos untuk pupuk tanaman.",
    "glass": "🍾 Kaca dapat dilebur dan didaur ulang menjadi produk kaca baru.",
    "metal": "🔩 Logam dapat dilebur dan digunakan kembali menjadi berbagai produk.",
    "paper": "📄 Kertas dapat didaur ulang menjadi kertas baru atau kerajinan.",
    "plastic": "🧴 Plastik sebaiknya dipilah dan didaur ulang untuk mengurangi pencemaran.",
    "trash": "🗑️ Sampah residu yang tidak dapat didaur ulang harus dibuang ke TPA."
}

# ==========================
# SESSION STATE
# ==========================
if "page" not in st.session_state:
    st.session_state.page = "home"

if "captured_image" not in st.session_state:
    st.session_state.captured_image = None

# ==========================
# HOME PAGE
# ==========================
if st.session_state.page == "home":

    st.image("assets/logo.png", width=100)

    st.markdown(
        "<div class='main-title'>Waste AI Classifier</div>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<div class='subtitle'>Deteksi Jenis Sampah Menggunakan Artificial Intelligence</div>",
        unsafe_allow_html=True
    )

    st.write("")

    st.image("assets/hero.png", use_container_width=True)

    st.write("")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.info("🤖 AI Detection")

    with col2:
        st.success("⚡ Fast Analysis")

    with col3:
        st.warning("♻️ Eco Friendly")

    st.write("")

    st.markdown("""
    <div class='card'>
    <h4>🌍 Kenapa Waste AI?</h4>

    Waste AI membantu mengidentifikasi jenis sampah secara otomatis menggunakan teknologi Deep Learning sehingga proses pemilahan sampah menjadi lebih mudah, cepat, dan akurat.
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    if st.button("📸 Ambil Foto Sampah", use_container_width=True):
        st.session_state.page = "camera"
        st.rerun()

# ==========================
# CAMERA PAGE
# ==========================
elif st.session_state.page == "camera":

    st.title("📸 Ambil Gambar Sampah")

    image = st.camera_input("Arahkan kamera ke objek sampah")

    if image is not None:

        st.session_state.captured_image = image

        st.image(image, caption="Preview Gambar", use_container_width=True)

        if st.button("🔍 Analisis Sampah", use_container_width=True):

            with st.spinner("🤖 AI sedang menganalisis gambar..."):
                time.sleep(2)

            st.session_state.page = "result"
            st.rerun()

    if st.button("⬅️ Kembali"):
        st.session_state.page = "home"
        st.rerun()

# ==========================
# RESULT PAGE
# ==========================
elif st.session_state.page == "result":

    st.title("📊 Hasil Analisis")

    image = Image.open(st.session_state.captured_image)

    st.image(image, caption="Gambar Sampah", use_container_width=True)

    img = image.resize((150,150))
    img_array = np.array(img)

    if len(img_array.shape) == 2:
        img_array = np.stack((img_array,)*3, axis=-1)

    if img_array.shape[-1] == 4:
        img_array = img_array[:,:,:3]

    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array, verbose=0)

    class_index = np.argmax(prediction)
    confidence = float(np.max(prediction))

    result = CLASS_NAMES[class_index]

    st.success(f"♻️ Jenis Sampah : {result.upper()}")

    st.info(f"🎯 Akurasi Prediksi : {confidence*100:.2f}%")

    st.markdown(
        f"""
        <div class='result-card'>
        <h4>📝 Cara Pengelolaan</h4>
        {waste_info[result]}
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")
    st.write("")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🏠 Home", use_container_width=True):
            st.session_state.page = "home"
            st.session_state.captured_image = None
            st.rerun()

    with col2:
        if st.button("📸 Ambil Lagi", use_container_width=True):
            st.session_state.page = "camera"
            st.rerun()