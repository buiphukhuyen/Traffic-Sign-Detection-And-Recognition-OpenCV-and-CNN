import cv2
import numpy as np

def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def equalize(img):
    img = cv2.equalizeHist(img)
    return img


def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img / 255  # image normalization
    return img

# Tiền xử lý hình ảnh
def preprocess_img(imgBGR, erode_dilate=True):  # Bật phương pháp Erode (xói mòn) và Dilate (giãn nỡ)
    rows, cols, _ = imgBGR.shape
    # Chuyển mã BGR sang HSV (Xác định được một màu cụ thể hơn, dựa trên các màu sắc và các dải bão hòa)
    imgHSV = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2HSV)
    Bmin = np.array([100, 43, 46])
    Bmax = np.array([124, 255, 255])
    img_Bbin = cv2.inRange(imgHSV, Bmin, Bmax)

    Rmin1 = np.array([0, 43, 46])
    Rmax1 = np.array([10, 255, 255])
    img_Rbin1 = cv2.inRange(imgHSV, Rmin1, Rmax1)

    Rmin2 = np.array([156, 43, 46])
    Rmax2 = np.array([180, 255, 255])
    img_Rbin2 = cv2.inRange(imgHSV, Rmin2, Rmax2)

    # img_Rbin = np.maximum(img_Rbin1, img_Rbin2)
    # img_bin = np.maximum(img_Bbin, img_Rbin)

    # # Erode và Dilate
    # if erode_dilate is True:
    #     kernelErosion = np.ones((3, 3), np.uint8)
    #     kernelDilation = np.ones((3, 3), np.uint8)
    #     img_bin = cv2.erode(img_bin, kernelErosion, iterations=2)
    #     img_bin = cv2.dilate(img_bin, kernelDilation, iterations=2)

    return img_Rbin2

img = cv2.imread('signs.png')

img_s = preprocess_img(img)

cv2.imshow("Orgin Image", img)

cv2.imshow("Final Image", img_s)

cv2.waitKey(0)
