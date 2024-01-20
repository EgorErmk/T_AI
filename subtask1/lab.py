import cv2 as cv
import numpy as np

def togreyscale(bgr_normalized_pixel):
    mask = (bgr_normalized_pixel < 0.04045)
    bgr_normalized_pixel = (np.logical_not(mask)*bgr_normalized_pixel) + ((mask*bgr_normalized_pixel)/12.92)
    bgr_normalized_pixel = (mask*bgr_normalized_pixel) + np.power((((np.logical_not(mask)*bgr_normalized_pixel) + 0.055)/1.055),2.4)
    bgr_normalized_pixel = bgr_normalized_pixel.dot(np.array([0.114,0.587,0.299]))
    return bgr_normalized_pixel

def fun(Image,prepfun):
    for i in range(Image.shape[0]):
        for j in range(Image.shape[1]):
            Image[i][j] = prepfun(Image[i][j])
    return Image

def threshold_proc(greyscaled_pixel):
    if (greyscaled_pixel[0] > 128):
        greyscaled_pixel[0] = np.array([0])
    else:
        greyscaled_pixel = np.array([255])
    return greyscaled_pixel
print('Reading image...')
cv.samples.addSamplesDataSearchPath(r"/opt/images/")
img = cv.imread(cv.samples.findFile("IMG_1836_3840.png"))
print('Grayscale converting...')
img_grey = fun(img.astype(float)/255,togreyscale)
img_grey = (img_grey*255).astype(np.uint8)
print('Saving image...')
cv.imwrite('greyscale.jpg',img_grey)
print('Threshold processing...')
img_bin = np.empty((img_grey.shape[0],img_grey.shape[1],1),dtype="uint8")
for i in range(img_grey.shape[0]):
    for j in range(img_grey.shape[1]):
        img_bin[i][j] = img_grey[i][j][0]  

img_th = fun(img_bin,threshold_proc)
print('Saving image...')
cv.imwrite('threshold.jpg',img_th)