# streamlit run 05_10_web.py
# 載入套件
import streamlit as st 
from skimage import io
from skimage.transform import resize
import numpy as np  
import tensorflow as tf

# 模型載入
model = tf.keras.models.load_model('./mnist_model.h5')

# 標題
st.title("上傳圖片(0~9)辨識")

# 上傳圖檔
uploaded_file = st.file_uploader("上傳圖片(.png)", type="png")
if uploaded_file is not None:
    # 讀取上傳圖檔
    image1 = io.imread(uploaded_file, as_gray=True)
    # 縮小圖形為(28, 28)
    image_resized = resize(image1, (28, 28), anti_aliasing=True)    
    # 插入第一維，代表筆數
    X1 = image_resized.reshape(1,28,28,1) 
    # 顏色反轉
    X1 = np.abs(1-X1)

    # 預測
    predictions = model.predict_classes(X1)[0]
    # 顯示預測結果
    st.write(f'預測結果:{predictions}')
    # 顯示上傳圖檔
    st.image(image1)
