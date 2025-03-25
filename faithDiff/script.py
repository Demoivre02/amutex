import cv2
import numpy as np
import os

def enhance_image(input_image):
    """
    Enhances an input image and returns the enhanced image.
    """
    
    I = input_image.astype("float64") / 255

    
    def dark_channel(im, sz):
        b, g, r = cv2.split(im)
        dc = cv2.min(cv2.min(r, g), b)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (sz, sz))
        dark = cv2.erode(dc, kernel)
        return dark

    
    def get_atmosphere(im, darkch, p=0.001):
        M, N = darkch.shape
        flatI = im.reshape(M * N, 3)
        flatdark = darkch.ravel()
        searchidx = (-flatdark).argsort()[:int(M * N * p)]
        A = np.mean(flatI.take(searchidx, axis=0), axis=0)
        return A

    
    def transmission_estimate(im, A, sz):
        omega = 0.95
        im3 = np.empty(im.shape, im.dtype)
        for ind in range(0, 3):
            im3[:, :, ind] = im[:, :, ind] / A[ind]
        transmission = 1 - omega * dark_channel(im3, sz)
        return transmission

    
    def transmission_refine(im, et):
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        gray = np.float64(gray) / 255
        r = 30
        eps = 0.0001
        mean_I = cv2.boxFilter(gray, cv2.CV_64F, (r, r))
        mean_p = cv2.boxFilter(et, cv2.CV_64F, (r, r))
        mean_Ip = cv2.boxFilter(gray * et, cv2.CV_64F, (r, r))
        cov_Ip = mean_Ip - mean_I * mean_p
        mean_II = cv2.boxFilter(gray * gray, cv2.CV_64F, (r, r))
        var_I = mean_II - mean_I * mean_I
        a = cov_Ip / (var_I + eps)
        b = mean_p - a * mean_I
        mean_a = cv2.boxFilter(a, cv2.CV_64F, (r, r))
        mean_b = cv2.boxFilter(b, cv2.CV_64F, (r, r))
        q = mean_a * gray + mean_b
        return q

    
    def recover(im, t, A, tx=0.1):
        res = np.empty(im.shape, im.dtype)
        t = cv2.max(t, tx)
        for ind in range(0, 3):
            res[:, :, ind] = (im[:, :, ind] - A[ind]) / t + A[ind]
        return res

    
    dark = dark_channel(I, 15)  
    A = get_atmosphere(I, dark)  
    te = transmission_estimate(I, A, 15)  
    t = transmission_refine(input_image, te)  
    J = recover(I, t, A, 0.1)  

    
    enhanced_image = np.clip(J * 255, 0, 255).astype("uint8")
    return enhanced_image


if __name__ == "__main__":
    
    input_image_path = "faithDiff/test/newyork.png"  
    output_dir = "faithDiff/output"  

    
    input_image = cv2.imread(input_image_path)
    if input_image is None:
        raise ValueError("Image not found or unable to load.")

    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    
    enhanced_image = enhance_image(input_image)

    
    output_image_path = os.path.join(output_dir, "enhanced_image.jpg")
    cv2.imwrite(output_image_path, enhanced_image)

    print(f"Enhanced image saved to: {output_image_path}")

    
    cv2.imshow("Original Image", input_image)
    cv2.imshow("Enhanced Image", enhanced_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()