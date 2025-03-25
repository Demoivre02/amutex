import cv2
import numpy as np
import os

def enhance_image(image_path, output_dir):
    """
    Enhances the input image and saves the output to the specified directory.
    Returns the path to the enhanced image.
    """
    
    hazy_img = cv2.imread(image_path)
    if hazy_img is None:
        raise ValueError("Image not found or unable to load.")

    
    I = hazy_img.astype("float64") / 255

    
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

    
    def enhance_hsv(hazy_img, A, R=3):
        hazy_hsv = cv2.cvtColor(hazy_img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hazy_hsv)

        
        t_s = transmission_estimate(I, A, 3)
        translation_ratio = 1 / (R * t_s) - 1

        
        s_inters_img = np.full_like(s, A[1] * 255)
        s = s.astype("float64")
        enh_s = s + translation_ratio * (s - s_inters_img)
        enh_s = np.clip(enh_s, 0, 255).astype("uint8")

        v_inters_img = np.full_like(v, A[2] * 255)
        v = v.astype("float64")
        enh_v = v + translation_ratio * (v - v_inters_img)
        enh_v = np.clip(enh_v, 0, 255).astype("uint8")

        
        new_hsv = cv2.merge([h, enh_s, enh_v])
        return cv2.cvtColor(new_hsv, cv2.COLOR_HSV2BGR)

    
    dark = dark_channel(I, 15)  
    A = get_atmosphere(I, dark)  
    te = transmission_estimate(I, A, 15)  
    t = transmission_refine(hazy_img, te)  
    J = recover(I, t, A, 0.1)  

    
    enhanced_hsv = enhance_hsv(hazy_img, A)

    
    enhanced_image = np.clip(J * 255, 0, 255).astype("uint8")
    final_image = cv2.addWeighted(enhanced_image, 0.7, enhanced_hsv, 0.3, 0)

    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    
    input_filename = os.path.basename(image_path)
    output_filename = f"enhanced_{input_filename}"
    output_path = os.path.join(output_dir, output_filename)

    
    cv2.imwrite(output_path, final_image)
    print(f"Enhanced image saved to: {output_path}")

    return output_path


if __name__ == '__main__':
    
    input_image_path = 'rsvt/test/tiananmen1.png'  
    output_directory = 'rsvt/output'  

    
    output_image_path = enhance_image(input_image_path, output_directory)

    
    original_image = cv2.imread(input_image_path)
    enhanced_image = cv2.imread(output_image_path)
    cv2.imshow('Original Image', original_image)
    cv2.imshow('Enhanced Image', enhanced_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()