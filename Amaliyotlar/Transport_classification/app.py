import streamlit as st
from fastai.vision.all import load_learner, PILImage
import PIL
import pathlib
temp=pathlib.PosixPath
pathlib.PosixPath=pathlib.WindowsPath
# Sarlavha
st.title("Transport Tasviri Klassifikatori 🚗🚚🛵")

# Modelni yuklash
@st.cache_resource
def load_model():
    return load_learner('transport_model2.pkl')

model = load_model()

# Rasm yuklash
uploaded_file = st.file_uploader("Rasm yuklang", type=['png', 'jpg', 'jpeg'])

if uploaded_file:
    try:
        img = PILImage.create(uploaded_file)
        st.image(img, caption="Yuklangan rasm", use_column_width=True)

        pred, pred_idx, probs = model.predict(img)

        st.success(f"**Bashorat:** {pred}")
        st.info(f"**Ishonch darajasi:** {probs[pred_idx]:.2%}")
    except PIL.UnidentifiedImageError:
        st.error("❌ Bu rasm fayli emas. Iltimos, PNG yoki JPG formatda rasm yuklang.")
