import argparse
import os
import cv2
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import time

# settings
parser = argparse.ArgumentParser(description='Performance')
parser.add_argument('--input_dir', default='results/')
parser.add_argument('--reference_dir', default='dataset')

opt = parser.parse_args()
print(opt)

im_path = opt.input_dir
re_path = opt.reference_dir
avg_psnr = 0
avg_ssim = 0
n = 0

for filename in os.listdir(im_path):
    print(im_path + '/' + filename)
    t0 = time.time()
    n = n + 1
    im1 = cv2.imread(im_path + '/' + filename)
    im2 = cv2.imread(re_path + '/' + filename)

    h, w, c = im2.shape
    im1 = cv2.resize(im1, (w, h))

    score_psnr = psnr(im1, im2)

    gray_im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    gray_im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    score_ssim = ssim(gray_im1, gray_im2, multichannel=True)

    avg_psnr += score_psnr
    avg_ssim += score_ssim
    t1 = time.time()

    print("===> PSNR: %.4f dB || SSIM: %.4f || Timer: %.4f sec." % (score_psnr, score_ssim, (t1 - t0)))


avg_psnr = avg_psnr / n
avg_ssim = avg_ssim / n
print("===> Avg.PSNR: {:.4f} dB ".format(avg_psnr))
print("===> Avg.SSIM: {:.4f} ".format(avg_ssim))




