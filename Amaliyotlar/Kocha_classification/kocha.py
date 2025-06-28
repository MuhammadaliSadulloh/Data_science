import streamlit as st
from fastai.vision.all import *
import pathlib

import plotly.express as px
import platform

plt=platform.system()
if plt=='Linux':
    pathlib.WindowsPath=pathlib.PosixPath
else:
    temp=pathlib.PosixPath
    pathlib.PosixPath=pathlib.WindowsPath


st.title('Ko\'cha tasvirlarini klassifikatsiya qilish')

file=st.file_uploader('Rasm yuklash', type=['png', 'jpeg', 'gif', 'jpg'])
if file:
    st.image(file)
    # PIL convert
    img=PILImage.create(file)
    # model
    model=load_learner('Amaliyotlar/Kocha_classification/kocha_model.pkl')

    # prediction
    pred,pred_id,probs=model.predict(img)
    st.success(f'Bashorat:{pred}')
    st.info(f'Ehtimollik:{probs[pred_id]:.3%}')
    fig=px.bar(x=model.dls.vocab,y=probs*100)
    st.plotly_chart(fig)