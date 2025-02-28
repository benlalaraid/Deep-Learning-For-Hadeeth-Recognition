import numpy as np
from PIL import Image


def otsu_threshold(image):
    # Calculate histogram
    hist, bins = np.histogram(image.ravel(), 256, [0,256])

    # Calculate total number of pixels in image
    total_pixels = image.shape[0] * image.shape[1]

    # Initialize variables
    max_var = 0
    threshold = 0

    # Loop through all possible threshold values
    for i in range(256):
        # Calculate class probabilities
        w0 = np.sum(hist[:i]) / total_pixels
        w1 = np.sum(hist[i:]) / total_pixels

        # Calculate class means
        u0 = np.sum(np.arange(i) * hist[:i]) / (w0 * total_pixels)
        u1 = np.sum(np.arange(i, 256) * hist[i:]) / (w1 * total_pixels)

        # Calculate between-class variance
        var = w0 * w1 * ((u0 - u1) ** 2)

        # Check if current threshold produces maximum between-class variance
        if var > max_var:
            max_var = var
            threshold = i

    # Apply threshold to image
    binary_img = np.zeros_like(image)
    binary_img[image >= threshold] = 255

    return binary_img

def crop_on_borders(img) :   
    gray = otsu_threshold(np.array(img.convert("L")))
    #gray= np.array([[255,255,255],[255,255,255],[255,0,255],[255,255,255],[255,255,255],[255,0,255],[255,255,255]])
    img_gray = Image.fromarray(gray)
    lg = len(gray)
    lgT = len(gray.T)
    ci1,ci2 = lg,lg
    cj1,cj2 = lgT,lgT
    

    for i in range(lg):
            if np.min(gray[i])!=255:
                ci1 = i-1
                break
    for i in reversed(range(lg)):
            if np.min(gray[i])!=255:
                ci2 = i+1
                break


    for i in range(lgT):
            if np.min(gray.T[i])!=255:
                cj1 = i-1
                break
    for i in reversed(range(lgT)):
            if np.min(gray.T[i])!=255:
                cj2 = i+1
                break
                
    return (cj1,ci1,cj2,ci2)