import numpy as np
import cv2
from tensorflow import keras

threshold = 0.75  # Ngưỡng Threshold (dự đoán > 75%)
font = cv2.FONT_HERSHEY_SIMPLEX  # Thiết lập Font chữ
model = keras.models.load_model('traffic_sign_model.h5')  # Model đã Training


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

    img_Rbin = np.maximum(img_Rbin1, img_Rbin2)
    img_bin = np.maximum(img_Bbin, img_Rbin)

    # Erode và Dilate
    if erode_dilate is True:
        kernelErosion = np.ones((3, 3), np.uint8)
        kernelDilation = np.ones((3, 3), np.uint8)
        img_bin = cv2.erode(img_bin, kernelErosion, iterations=2)
        img_bin = cv2.dilate(img_bin, kernelDilation, iterations=2)

    return img_bin


# Biến đổi ảnh xám (Grayscale)
def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


# Cân bằng sáng (Histogram Equalization)
def equalize(img):
    img = cv2.equalizeHist(img)
    return img


# Xử lý và chuẩn hoá hình ảnh
def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img / 255
    return img


# Xác định các contour
def contour_detect(img_bin, min_area, max_area=-1, wh_ratio=2.0):
    rects = []

    contours, _ = cv2.findContours(img_bin.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if len(contours) == 0:
        return rects

    max_area = img_bin.shape[0] * img_bin.shape[1] if max_area < 0 else max_area

    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= min_area and area <= max_area:
            x, y, w, h = cv2.boundingRect(contour)
            if 1.0 * w / h < wh_ratio and 1.0 * h / w < wh_ratio:
                rects.append([x, y, w, h])

    # Trả về tọa độ và kích thước của hình chữ nhật bao quanh contour
    return rects


# Lấy nhãn của đối tượng biển báo
def getCalssName(classNo):
    if classNo == 0:
        return 'Speed Limit 20 km/h'
    elif classNo == 1:
        return 'Speed Limit 30 km/h'
    elif classNo == 2:
        return 'Speed Limit 50 km/h'
    elif classNo == 3:
        return 'Speed Limit 60 km/h'
    elif classNo == 4:
        return 'Speed Limit 70 km/h'
    elif classNo == 5:
        return 'Speed Limit 80 km/h'
    elif classNo == 6:
        return 'End of Speed Limit 80 km/h'
    elif classNo == 7:
        return 'Speed Limit 100 km/h'
    elif classNo == 8:
        return 'Speed Limit 120 km/h'
    elif classNo == 9:
        return 'No passing'
    elif classNo == 10:
        return 'No passing for vechiles over 3.5 metric tons'
    elif classNo == 11:
        return 'Right-of-way at the next intersection'
    elif classNo == 12:
        return 'Priority road'
    elif classNo == 13:
        return 'Yield'
    elif classNo == 14:
        return 'Stop'
    elif classNo == 15:
        return 'No vechiles'
    elif classNo == 16:
        return 'Vechiles over 3.5 metric tons prohibited'
    elif classNo == 17:
        return 'No entry'
    elif classNo == 18:
        return 'General caution'
    elif classNo == 19:
        return 'Dangerous curve to the left'
    elif classNo == 20:
        return 'Dangerous curve to the right'
    elif classNo == 21:
        return 'Double curve'
    elif classNo == 22:
        return 'Bumpy road'
    elif classNo == 23:
        return 'Slippery road'
    elif classNo == 24:
        return 'Road narrows on the right'
    elif classNo == 25:
        return 'Road work'
    elif classNo == 26:
        return 'Traffic signals'
    elif classNo == 27:
        return 'Pedestrians'
    elif classNo == 28:
        return 'Children crossing'
    elif classNo == 29:
        return 'Bicycles crossing'
    elif classNo == 30:
        return 'Beware of ice/snow'
    elif classNo == 31:
        return 'Wild animals crossing'
    elif classNo == 32:
        return 'End of all speed and passing limits'
    elif classNo == 33:
        return 'Turn right ahead'
    elif classNo == 34:
        return 'Turn left ahead'
    elif classNo == 35:
        return 'Ahead only'
    elif classNo == 36:
        return 'Go straight or right'
    elif classNo == 37:
        return 'Go straight or left'
    elif classNo == 38:
        return 'Keep right'
    elif classNo == 39:
        return 'Keep left'
    elif classNo == 40:
        return 'Roundabout mandatory'
    elif classNo == 41:
        return 'End of no passing'
    elif classNo == 42:
        return 'End of no passing by vechiles over 3.5 metric tons'


if __name__ == "__main__":

    cap = cv2.VideoCapture(0)  # Sử dụng Webcam

    cols = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    rows = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    while True:
        ret, img = cap.read()

        img = cv2.resize(img, (640, 480))  # Resize

        # Chuẩn hoá ảnh nhị phân
        img_bin = preprocess_img(img, True)

        cv2.imshow("Binary Image", img_bin)

        min_area = img_bin.shape[0] * img.shape[1] / (25 * 25)

        rects = contour_detect(img_bin, min_area=min_area)  # Lấy tọa độ và kích thước của hình chữ nhật bao quanh contour

        img_bbx = img.copy()

        for rect in rects:
            xc = int(rect[0] + rect[2] / 2)
            yc = int(rect[1] + rect[3] / 2)

            size = max(rect[2], rect[3])
            x1 = max(0, int(xc - size / 2))
            y1 = max(0, int(yc - size / 2))
            x2 = min(cols, int(xc + size / 2))
            y2 = min(rows, int(yc + size / 2))

            # rect[2] là chiều rộng (width) và rect[3] là chiều cao (height)
            if rect[2] > 100 and rect[3] > 100:  # Chỉ phát hiện những biển báo có chiều cao và chiều rộng > 100
                cv2.rectangle(img_bbx, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 0, 255), 2)

            # Cắt ảnh sau khi phát hiện và tiền xử lý lại
            crop_img = np.asarray(img[y1:y2, x1:x2])
            crop_img = cv2.resize(crop_img, (32, 32))
            crop_img = preprocessing(crop_img)

            cv2.imshow("After Processing", crop_img)

            # Đưa ra dự đoán nhãn của ảnh
            crop_img = crop_img.reshape(1, 32, 32, 1)  # (1,32,32) sau reshape it trở thành (1,32,32,1)
            predictions = model.predict(crop_img)
            classIndex = np.argmax(predictions, axis=1)  # Lấy ra index của nhãn sau khi dự đoán
            probabilityValue = np.amax(predictions)  # Lấy ra độ chính xác dự đoán

            if probabilityValue > threshold:  # (>75%)
                # In kết quả nhãn của biển báo
                cv2.putText(img_bbx, str(classIndex) + " " + str(getCalssName(classIndex)), (rect[0], rect[1] - 10),
                            font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)

                # In độ chính xác
                cv2.putText(img_bbx, str(round(probabilityValue * 100, 2)) + "%", (rect[0], rect[1] - 40), font, 0.75,
                            (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow("Detect Result", img_bbx)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Nhấn q để dừng
            break

cap.release()
cv2.destroyAllWindows()
