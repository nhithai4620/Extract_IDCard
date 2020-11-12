import numpy as np
import cv2
import imutils
import pytesseract


image = cv2.imread('cmnd5.jpg')
image = imutils.resize(image, height = 500)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #ảnh xám
blur = cv2.GaussianBlur(gray, (5, 5), 0) # làm mờ ảnh
bilateral = cv2.bilateralFilter(blur, 5, 10, 10) # làm mịn ảnh


# _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
edged = cv2.Canny(bilateral, 30, 120) # tìm cạnh bằng Canny

contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # đường viền

largest_area = sorted(contours, key = cv2.contourArea)

for cnts in largest_area:
    peri = cv2.arcLength(cnts, True)
    approx = cv2.approxPolyDP(cnts, 0.09 * peri, True)

print(approx)
cv2.drawContours(image, [approx], 0, (0, 255, 255), 5) 

cv2.imshow('Image', image)
cv2.waitKey(0)

coordinates_0 = (approx[0,0,0], approx[0,0,1])
coordinates_1 = (approx[1,0,0], approx[1,0,1])
coordinates_2 = (approx[2,0,0], approx[2,0,1])
coordinates_3 = (approx[3,0,0], approx[3,0,1])

if approx[0,0,0] < 320:
    top_left = coordinates_0
    bottom_left = coordinates_1
    bottom_right = coordinates_2
    top_right = coordinates_3

else:
    top_right = coordinates_0
    top_left = coordinates_1
    bottom_left = coordinates_2
    bottom_right = coordinates_3

pts1 = np.float32([[top_left], [top_right], [bottom_right], [bottom_left]])
pts2 = np.float32([[0, 0], [450, 0], [450, 300], [0, 300]])
matrix = cv2.getPerspectiveTransform(pts1, pts2)
result = cv2.warpPerspective(image, matrix, (450, 300))


cv2.imshow('Output', result)
cv2.waitKey(0)
 
so_cmnd = result[70:103, 225:390]
ho_ten = result[100:150, 187:420] # cắt ảnh

cv2.imshow('so_cmnd', so_cmnd)
cv2.waitKey(0)
cv2.imshow('hoten', ho_ten)
cv2.waitKey(0)




# def edit_image_after(image):
#     gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     blur_image = cv2.GaussianBlur(gray_image, (3, 3), 1)
#     bilateral_image = cv2.bilateralFilter(blur_image, 5, 10, 10)
#     # thresh_ho_ten = cv2.adaptiveThreshold(bilateral_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 3, 3)
#     # _, thresh_ho_ten = cv2.threshold(bilateral_image ,125,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

#     return bilateral_image

gray_so_cmnd = cv2.cvtColor(so_cmnd, cv2.COLOR_BGR2GRAY)
blur_so_cmnd = cv2.GaussianBlur(gray_so_cmnd, (3, 3), 1)
bilateral_so_cmnd = cv2.bilateralFilter(blur_so_cmnd, 5, 10, 10)
# thresh_so_cmnd = cv2.adaptiveThreshold(bilateral_so_cmnd, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 4)
cv2.imshow('socmnd', bilateral_so_cmnd)
cv2.waitKey(0)

gray_ho_ten = cv2.cvtColor(ho_ten, cv2.COLOR_BGR2GRAY)
blur_ho_ten = cv2.GaussianBlur(gray_ho_ten, (3, 3), 1)
bilateral_ho_ten = cv2.bilateralFilter(blur_ho_ten, 5, 10, 10)
cv2.imshow('hoten', bilateral_ho_ten)
cv2.waitKey(0)

custom_config = r'-c tessedit_char_whitelist=0123456789' # cấu hình
so_cmnd_text = pytesseract.image_to_string(bilateral_so_cmnd, lang='vie',
                                config=custom_config)
print('So CMND: ' + so_cmnd_text)

ho_ten_text = pytesseract.image_to_string(bilateral_ho_ten, lang='vie')
print('Ho va ten: ' + ho_ten_text)


