"""indexes"""
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from functools import partial
import numpy as np
from skvideo.measure import strred
from skvideo.utils import rgb2gray


def raw2gray(bayer_images):
    """RGBG -> linear RGB"""
    # T H W C
    lin_rgb = np.stack([
        bayer_images[..., 0], 
        np.mean(bayer_images[..., [1,3]], axis=3), 
        bayer_images[...,2]], axis=3)

    lin_gray = rgb2gray(lin_rgb)

    return lin_gray


class Framewise(object):
    def __init__(self, index_fn):
        self.index_fn = index_fn

    def __call__(self, X, Y):
        T = X.shape[0]
        bwindex = []
        for t in range(T):
            x = X[t, ...]
            y = Y[t, ...]
            index = self.index_fn(x, y)
            bwindex.append(index)
        
        return bwindex


compare_psnr_video = Framewise(partial(peak_signal_noise_ratio, data_range=255))
compare_ssim_video = Framewise(partial(structural_similarity, data_range=255, multichannel=True))


def compare_ncc(x, y):
    return np.mean((x-np.mean(x)) * (y-np.mean(y))) / (np.std(x) * np.std(y)) 


def ssq_error(correct, estimate):
    """Compute the sum-squared-error for an image, where the estimate is
    multiplied by a scalar which minimizes the error. Sums over all pixels
    where mask is True. If the inputs are color, each color channel can be
    rescaled independently."""
    assert correct.ndim == 2
    if np.sum(estimate**2) > 1e-5:
        alpha = np.sum(correct * estimate) / np.sum(estimate**2)
    else:
        alpha = 0.
    return np.sum((correct - alpha*estimate) ** 2)


def local_error(correct, estimate, window_size, window_shift):
    """Returns the sum of the local sum-squared-errors, where the estimate may
    be rescaled within each local region to minimize the error. The windows are
    window_size x window_size, and they are spaced by window_shift."""
    M, N, C = correct.shape
    ssq = total = 0.
    for c in range(C):
        for i in range(0, M - window_size + 1, window_shift):
            for j in range(0, N - window_size + 1, window_shift):
                correct_curr = correct[i:i+window_size, j:j+window_size, c]
                estimate_curr = estimate[i:i+window_size, j:j+window_size, c]
                ssq += ssq_error(correct_curr, estimate_curr)
                total += np.sum(correct_curr**2)
    # assert np.isnan(ssq/total)
    return ssq / total


def quality_assess(X, Y, data_range=255):
    # Y: correct; X: estimate
    if X.ndim == 3:  # image
        psnr = peak_signal_noise_ratio(Y, X, data_range=data_range)
        ssim = structural_similarity(Y, X, data_range=data_range, multichannel=True)
        return {'PSNR':psnr, 'SSIM': ssim}

    elif X.ndim == 4:  # video clip
        vpsnr = np.mean(compare_psnr_video(Y/data_range*255, X/data_range*255))
        vssim = np.mean(compare_ssim_video(Y/data_range*255, X/data_range*255))

        if X.shape[0] != 1:
            _, _strred, _strredsn = strred(raw2gray(Y)/data_range, raw2gray(X)/data_range)
        else:
            _strred = 0
            _strredsn = 0

        return {'PSNR': vpsnr, 'SSIM': vssim, 'STRRED': _strred, 'STRREDSN':_strredsn}
    else:
        raise NotImplementedError
