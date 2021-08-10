import json
import numpy as np
import requests
from skimage import io
from skimage.transform import resize

uploaded_file='./myDigits/4.png'
image1 = io.imread(uploaded_file, as_gray=True)
# 縮小圖形為(28, 28)
image_resized = resize(image1, (28, 28), anti_aliasing=True)    
# 插入第一維，代表筆數
X1 = image_resized.reshape(1,28,28,1) 
# 顏色反轉
X1 = np.abs(1-X1)

# 將預測資料轉為 Json 格式
data = json.dumps({
    "instances": X1.tolist()
    })
    
# 呼叫 TensorFlow Serving API    
headers = {"content-type": "application/json"}
json_response = requests.post(
    'http://localhost:8501/v1/models/MLP:predict',
    data=data, headers=headers)
    
# 解析預測結果    
predictions = np.array(json.loads(json_response.text)['predictions'])
print(np.argmax(predictions, axis=-1))
