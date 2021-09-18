import cv2
import os
import numpy as np

img = cv2.imread('object-detection/img.jpg')
H, W = img.shape[0], img.shape[1]
print(f"Height : {H}, Width: {W}")

def convolv(mat, kernel, y, x):
  row_l, row_h = (-1+y+H)%H, (2+y+H)%H
  col_l, col_h = (-1+x+W)%W, (2+x+W)%W
  row_l, row_h = -1+y, 2+y
  col_l, col_h = -1+x, 2+x
  accm = np.sum(mat[row_l : row_h, col_l : col_h] * kernel)
  return accm

def translate(pl, ph, nl, nh, x):
  slope = (nh - nl) / (ph - pl)
  dist = (x - pl)
  return np.rint(nl + slope * dist)

def transform_range(pl, pr, nl, nr, x):
    ps = pr - pl
    ns = nr - nl
    scale = (x - pl) / ps
    return np.rint(nl + scale*ns)

def sobel(img):
  row_kernel = [-1, -2, -1, 0, 0, 0, 1, 2, 1]
  col_kernel = [-1, 0, 1, -2, 0, 2 , -1, 0, 1]
  right_diag_kernel = [-2, -1, 0, -1, 0, 1, 0, 1, 2]
  left_diag_kernel = [0, 1, 2, -1, 0, 1, -2, -1, 0]
  filters = list(map(lambda x: np.array(x).reshape(3,3),
          [row_kernel, col_kernel, right_diag_kernel, left_diag_kernel]))
  #B, G, R = img[:,:,0], img[:,:,1], img[:,:,2]
  B, G, R = cv2.split(img)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  #new_img = np.zeros(img.shape)
  new_img = img.copy()
  INF = int(1e12)
  components = []
  for idx, _filter in enumerate(filters[:2]):
    big, small = -INF, INF
    for i in range(1, H-1):
      for j in range(1, W-1):
        conv = convolv(img, _filter, i, j)
        big, small = max(big, conv), min(small, conv)
        new_img[i][j] = conv
    #print(big, small)
    for i in range(1, H-1):
      for j in range(1, W-1):
        now = new_img[i][j]
        got = translate(small, big, 0, 255, now)
        new_img[i][j] = got
    components.append(new_img)
  return components


#img = np.float(img)/255.0
#gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=1)
#gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=1)
#mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)

s = sobel(img)
gx, gy = s[0], s[1]
cv2.imshow('img', img)
cv2.imshow('gx', gx)
cv2.imshow('gy', gy)
cv2.waitKey(0)
cv2.destroyAllWindows()
