import cv2
import os
import numpy as np
import math

H, W = None, None
def convolv(mat, kernel, y, x):
  lr, lc = len(kernel) // 2, len(kernel[0]) // 2
  row_l, row_h = (-lr+y+H)%H, (lr + 1+y+H)%H
  col_l, col_h = (-lc+x+W)%W, (lc + 1+x+W)%W
  row_l, row_h = -lr+y, lr+1+y
  col_l, col_h = -lc+x, lc+1+x
  accm = np.sum(mat[row_l : row_h, col_l : col_h] * kernel)
  return accm

def translate(pl, ph, nl, nh, x):
  slope = (nh - nl) / (ph - pl)
  dist = (x - pl)
  return np.rint(nl + slope * dist)


def gauss_smt(img, kernel=5):
    new_img = img.copy()
    sigma, r, normfactor = 1.0, 0.0, 0.0
    #sigma, r, normfactor = eval(input('Enter magnitude of std: ')), 0.0, 0.0
    s = 2.0 * sigma * sigma
    gkernel = [[0 for _ in range(5)] for _ in range(5)]
    for row in range(-2, 3, 1):
      for col in range(-2, 3, 1):
        r = row * row + col * col
        gkernel[row + 2][col + 2] = math.exp(-r / s) / (math.pi * s)
        normfactor += gkernel[row + 2][col + 2]
    for row in range(5):
      for col in range(5):
        gkernel[row][col] /= normfactor
    # applying gaussian blur
    lr, lc = len(gkernel)//2, len(gkernel[0])//2
    for row in range(lr, H-lr):
      for col in range(lc, W-lc):
        accm = convolv(img, gkernel, row, col)
        new_img[row][col] = accm

    return new_img

def avg_smt(img, kernel=3):
    new_img = img.copy()
    sq_k = kernel * kernel
    kernel = [[1/sq_k for _ in range(kernel)] for _ in range(kernel)]
    lr, lc = len(kernel)//2, len(kernel[0])//2
    for row in range(lr, H-lr):
      for col in range(lc, W-lc):
        new_img[row][col] = convol(img, kernel, row, col)

    return new_img

def med_smt(img):
  new_img = img.copy()
  for row in range(img.shape[0]):
    for col in range(img.shape[1]):
      v = [0 for _ in range(9)]
      cc = 0
      for r in range(3):
        for c in range(3):
          v[cc] = img[(row + r)%img.shape[0]][(col + c)%img.shape[1]]
          cc += 1
      v.sort()
      new_img[row][col] = v[4]
  return new_img

def blur(reps, fn, img):
  for i in range(reps):
    img = fn(img)
  return img


def sobel(img):
  row_kernel = [-1, -2, -1, 0, 0, 0, 1, 2, 1]
  col_kernel = [-1, 0, 1, -2, 0, 2 , -1, 0, 1]
  right_diag_kernel = [-2, -1, 0, -1, 0, 1, 0, 1, 2]
  left_diag_kernel = [0, 1, 2, -1, 0, 1, -2, -1, 0]
  filters = list(map(lambda x: np.array(x).reshape(3,3),
          [row_kernel, col_kernel, right_diag_kernel, left_diag_kernel]))
  new_img = img.copy()
  INF = int(1e12)
  components = []
  for idx, _filter in enumerate(filters[:2]):
    big, small = -INF, INF
    lr, lc = _filter.shape
    for i in range(lr, H-lr):
      for j in range(lc, W-lc):
        conv = convolv(img, _filter, i, j)
        big, small = max(big, conv), min(small, conv)
        #if img[i][j] != 255: print(img[i][j],conv)
        new_img[i][j] = conv
    for i in range(1, H-1):
      for j in range(1, W-1):
        now = new_img[i][j]
        got = translate(small, big, 0, 255, now)
        #if img[i][j] != 255: print(now, got)
        new_img[i][j] = got
    components.append(new_img)
  return components


#img = np.float(img)/255.0
#gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=1)
#gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=1)
#mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)

if __name__ == "__main__":
  for i in range(1,4):
    img = cv2.imread(f'object-detection/img_{i}.jpg')
    H, W = img.shape[0], img.shape[1]
    print(f"Height : {H}, Width: {W}")
    #B, G, R = img[:,:,0], img[:,:,1], img[:,:,2]
    B, G, R = cv2.split(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #img = img.astype('float64') / 255.0
    img = blur(2, gauss_smt, img)
    #img = blur(5, med_smt, img)
    gx, gy = sobel(img)
    cv2.imshow('img', img)
    cv2.imshow('gx', gx)
    cv2.imshow('gy', gy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
