import cv2, math
import numpy as np
from numpy.fft import fft2, fftshift, ifft2, ifftshift
def gaussianFilter(img, sigma, highPass, name):
	numRows, numCols, _ = img.shape
	centerX = int(numRows/2)
	centerY = int(numCols/2)
	filter = np.array([[math.exp(-((i - centerX)**2 + (j - centerY)**2) / (2 * sigma**2)) for j in range(numCols)] for i in range(numRows)])
	filter = 1 - filter if highPass else filter 
	cv2.imwrite("result/"+name+".bmp", filter*255.0)
	return filter

def convolveImages(filter, image):
	product = np.zeros(image.shape) + 0.j
	for i in range(image.shape[2]):
		freqImage = fftshift(fft2(image[:,:,i]))
		product[:,:,i] = (ifft2(ifftshift(freqImage * filter)))	
	return np.real(product)

img1 = cv2.imread("bird.bmp")
img2 = cv2.imread("plane.bmp")
#print(img1.shape, img2.shape)
highPassGaussian = gaussianFilter(img1, 20, True, "highPassGaussian")
lowPassGaussian = gaussianFilter(img2, 30, False, "lowPassGaussian")

highPassImg = convolveImages(highPassGaussian, img1)
lowPassImg = convolveImages(lowPassGaussian, img2)

cv2.imwrite("result/high.bmp", highPassImg)
cv2.imwrite("result/low.bmp", lowPassImg)

hybridImg = highPassImg + lowPassImg
cv2.imwrite("results/hybridImg_bird_plane_20_30.bmp", hybridImg)