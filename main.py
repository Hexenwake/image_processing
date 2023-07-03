# importing libraries
import numpy as np
import cv2
from sklearn.metrics import pairwise
from custom_firestore import Firestore


def remove_btm_pxl(img):
    row, column = img.shape
    bright_count = np.sum(np.array(img) >= 200)
    dark_count = np.sum(np.array(img) <= 20)
    img_np = np.zeros((row, column), dtype='uint8')

    last_3_row = row - 2
    for i in range(row):
        for j in range(column):
            if i > last_3_row:
                if bright_count > dark_count:
                    img_np[i, j] = 255
                else:
                    img_np[i, j] = 0
            else:
                img_np[i, j] = img[i, j]

    return img_np


def resize_show(text, image):
    resized = cv2.resize(image, (500, 500), interpolation=cv2.INTER_AREA)
    cv2.imshow(text, resized)


def contouring(th_img):
    areaArray = []
    contours, hierarchy3 = cv2.findContours(th_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    con_num = len(contours)
    # print(con_num)
    bright_count = np.sum(np.array(th_img) >= 200)
    dark_count = np.sum(np.array(th_img) <= 20)

    if bright_count > dark_count:
        for i, c in enumerate(contours):
            area = cv2.contourArea(c)
            areaArray.append(area)

        # first sort the array by area
        sorted_data = sorted(zip(areaArray, contours), key=lambda x: x[0], reverse=True)

        # find the nth largest contour [n-1][1], in this case 2
        secondlargestcontour = sorted_data[1][1]
        cnt = secondlargestcontour
    else:
        cnt = contours[con_num - 1]

    # mask = np.zeros(th_img.shape, np.uint8)
    # cv2.drawContours(mask, [cnt], -1, (255, 255, 255), 2, cv2.LINE_AA)

    leftmost = tuple(cnt[cnt[:, :, 0].argmin()][0])
    rightmost = tuple(cnt[cnt[:, :, 0].argmax()][0])
    topmost = tuple(cnt[cnt[:, :, 1].argmin()][0])
    bottommost = tuple(cnt[cnt[:, :, 1].argmax()][0])

    # center point
    cX = int((leftmost[0] + rightmost[0]) / 2)
    cY = int((topmost[1] + bottommost[1]) / 2)
    # print("Center point : " + str(tuple((cX, cY))))

    # cv2.circle(mask, (cX, cY), radius=5, color=(255, 0, 0), thickness=5)
    # cv2.circle(mask, leftmost, 0, (255, 0, 0), 20)
    # cv2.circle(mask, rightmost, 0, (255, 0, 0), 20)
    # cv2.circle(mask, topmost, 0, (255, 0, 0), 20)
    # cv2.circle(mask, bottommost, 0, (255, 0, 0), 20)
    # resize_show('mask', mask)
    # resize_show('blur', img_blur)

    distances = pairwise.euclidean_distances([(cX, cY)], Y=[topmost])[0]
    max_distance = distances[distances.argmax()]

    # calculate the radius of the circle with 80% of the max Euclidean distance obtained
    radius = int(0.70 * max_distance)

    circular_roi = np.zeros(th_img.shape[:2], dtype="uint8")
    print("Circular ROI shape : " + str(circular_roi.shape))
    # resize_show("Threshold", th_img)

    # draw the circular ROI with radius and center point of convex hull calculated above
    cv2.circle(circular_roi, (cX, cY), radius, 255, 1)
    # resize_show("Circular ROI Circle", circular_roi)

    # take bit-wise AND between thresholded hand using the circular ROI as the mask
    # which gives the cuts obtained using mask on the thresholded hand image
    circular_roi = cv2.bitwise_and(th_img, th_img, mask=circular_roi)
    # resize_show("Bitwise AND", circular_roi)

    (cnts, _) = cv2.findContours(circular_roi.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    total_fingers = len(cnts) - 1
    print("Number of fingers found = " + str(total_fingers))
    # cv2.drawContours(original, cnts, -1, (0, 255, 0), 3)
    # resize_show('ori+circle', original)
    # cv2.waitKey(0)
    return total_fingers


def main():
    original = Firestore().get_image()
    if original is None:
        return 1
    r1 = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    # blurring
    img_blur = cv2.GaussianBlur(r1, (13, 13), 0)
    # Otsu Thresholding
    ret3, th_Otsu = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    img_fil = remove_btm_pxl(th_Otsu)
    res_fingers = contouring(img_fil)
    
