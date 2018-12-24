import cv2
import numpy as np

def remove(path: str):
    ## (1) Read
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ## (2) Threshold
    th, threshed = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    ## (3) Find the min-area contour
    _cnts = cv2.findContours(threshed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]
    cnts = sorted(_cnts, key=cv2.contourArea)
    for cnt in cnts:
        if cv2.contourArea(cnt) > 100:
            break

    ## (4) Create mask and do bitwise-op
    mask = np.zeros(img.shape[:2], np.uint8)
    cv2.drawContours(mask, [cnt], -1, 255, -1)
    dst = cv2.bitwise_and(img, img, mask=mask)

    ## Save it
    cv2.imwrite("./dst.png", dst)

if __name__ == "__main__":
    remove('/home/atticus/Изображения/apple-announces-new-emojis-design-news_dezeen_2364_col_9-852x852.jpg')
