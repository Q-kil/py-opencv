from fastapi import FastAPI
import cv2
from datetime import datetime
import torch

app = FastAPI()

@app.get("/")
def read_root():
  return {"Hello": "World"}

@app.get("/cv")
def imgCV():
  imagePath ='./temp/img1.png'
  image = cv2.imread(imagePath)
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  result = cv2.Laplacian(gray, cv2.CV_64F).var()
  print('ai_hellohello.jpg blur:', result)
  return {}

@app.get("/cv1000")
def imgCV():
  current_time = datetime.now()
  print(current_time)
  for i in range(1, 1001):
    # print(i)
    imagePath ='./temp/img1.png'
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    result = cv2.Laplacian(gray, cv2.CV_64F).var()
    # print('ai_hellohello.jpg blur:', result)
  current_time2 = datetime.now()
  print(current_time2)

@app.get("/tf")
def tfFn():
  # 检查是否有可用的GPU设备
  # if torch.cuda.is_available():
  #     print('GPU可用')
  # else:
  #     print('没有可用的GPU')
  print(torch.backends.mps.is_available())
  print(torch.backends.mps.is_built())
