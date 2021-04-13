import cv2
import glob, numpy as np
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
print(cv2.__version__)
caltech_dir = "C:\\Users\\Byun\\Desktop"
categories = ["recycle", "no_recycle"] #카테고리이자 두가지 분류
nb_classes = len(categories)


X = []
y = []

for idx, cat in enumerate(categories): #enumerate : 리스트의 순서와 값을 전달 idx=index, cat= value

    label = [0 for i in range(nb_classes)]   # [0,0]
    label[idx] = 1                      # 해당 순서에서 [1,0] or [0,1]

    image_dir = caltech_dir + "/" + cat
    files = glob.glob(image_dir + "/*.jpg") #현재 디렉토리의 .jpg파일
    for i, f in enumerate(files):
        img = cv2.imread(f)

        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        img_checker = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 191,
                                            15)
        _, contours, hierarchy = cv2.findContours(img_checker, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rects = [cv2.boundingRect(contour) for contour in contours]

        for rect in rects:
            if rect[2] * rect[3] < 5000:
                continue

            if rect[2] < 50:
                margin = 100
            else:
                margin = 30

            roi = img[rect[1] - margin:rect[1] + rect[3] + margin, rect[0] - margin:rect[0] + rect[2] + margin]
            try:
                roi = cv2.resize(roi, (28, 28), cv2.INTER_AREA)  # (28,28,3)
                x = roi / 255.0
                x = image.img_to_array(x)
                x = np.expand_dims(x, axis=0)  # 앞으로 차원증가가 (1,28,28,3)
                x = preprocess_input(x)
                # cv2.rectangle(img, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0),
                #               3)
                X.append(x)
                print(X)
                y.append(label)


            except Exception as e:
                print(str(e))





# cv2.imshow("roi", roi)
# cv2.imshow("imsg", img)
# cv2.imwrite("R10sam.jpg", roi)
# cv2.imwrite("imgsam.jpg", img)
# cv2.waitKey(0)



