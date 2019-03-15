import cv2, requests


url = "http://localhost:8000/alien-detection/api/"

image_path = './queries/Root_Leinwand.jpg'
tracker = {"url": "./queries/Root_Leinwand.jpg"}
req = requests.post(url, data=tracker).json()
print (req)
cv2.waitKey(0)