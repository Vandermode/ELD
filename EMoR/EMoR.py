import os
import numpy as np
from os.path import join
import matplotlib.pyplot as plt
import rawpy
import cv2
import colour
from scipy import interpolate
from sklearn.metrics import mean_squared_error


def read_dorf(address):
    with open(address) as f:
        lines = f.readlines()
        curve_names = lines[0::6]
        Es = lines[3::6]
        Bs = lines[5::6]

        Es = [np.array(E.strip().split()).astype(np.float32) for E in Es]
        Bs = [np.array(B.strip().split()).astype(np.float32) for B in Bs]

    return curve_names, Es, Bs


def read_emor(address):    
    def _read_curve(lst):
        curve = [l.strip() for l in lst]
        curve = ' '.join(curve)        
        curve = np.array(curve.split()).astype(np.float32)
        return curve

    with open(address) as f:
        lines = f.readlines()
        k = 1
        E = _read_curve(lines[k:k+256])        
        k += 257
        f0 = _read_curve(lines[k:k+256])
        hs = []
        for _ in range(25):
            k += 257
            hs.append(_read_curve(lines[k:k+256]))

        hs = np.array(hs)

        return E, f0, hs        


def process_color_checker(datadir, name):
    # name = 'DSC01034'
    # datadir = 'D:\\Academic\\Data\\ELD-Sony-original\\Radiometric Calibration'

    imgpath = join(datadir, '{}.ARW'.format(name))
    raw = rawpy.imread(imgpath)    

    img = raw.postprocess(use_camera_wb=True, gamma=(1, 1), no_auto_bright=True, output_bps=16)[700:2500, 700:2000, :] / 65535.
    
    masks_y = np.linspace(210, 210+250*5, 6).astype(np.long)
    masks_x = np.linspace(210, 210+250*3, 4).astype(np.long)

    masks = np.zeros(img.shape)
    color_samples = np.zeros((24,3))

    k = 0
    for y in masks_y:
        for x in masks_x:
            masks[y:y+100, x:x+100, ...] = k / 24
            color_samples[k, :] = img[y:y+100, x:x+100, :].mean(axis=(0,1))
            k += 1
    
    img = np.clip(img + masks * 0.25, 0, 1)

    plt.imshow(img)
    plt.show()

    np.savetxt(name+'_raw.txt', color_samples)

    ################################################
    imgpath = join(datadir, '{}.JPG'.format(name))
    
    img = cv2.imread(imgpath, cv2.IMREAD_UNCHANGED)[700:2500,700:2000,::-1] / 255.

    masks_y = np.linspace(210, 210+250*5, 6).astype(np.long)
    masks_x = np.linspace(210, 210+250*3, 4).astype(np.long)

    masks = np.zeros(img.shape)
    color_samples = np.zeros((24,3))

    k = 0
    for y in masks_y:
        for x in masks_x:
            masks[y:y+100, x:x+100, ...] = k / 24
            color_samples[k, :] = img[y:y+100, x:x+100, :].mean(axis=(0,1))
            k += 1
    
    img = np.clip(img + masks * 0.25, 0, 1)

    plt.imshow(img)
    plt.show()

    np.savetxt(name+'_rgb.txt', color_samples)


if __name__ == '__main__':
    datadir = 'D:\\Academic\\Code\\EMOR'

    E, f0, hs = read_emor(join(datadir, 'emor.txt'))
    
    num_params = 5
    F0 = interpolate.interp1d(E, f0)
    H = interpolate.interp1d(E, hs[:num_params, :])
    
    # for i in range(num_params):
    #     plt.plot(E, hs[i, :])
    # plt.show()

    ############

    curve_names, xs, ys = read_dorf(join(datadir, 'dorfCurves.txt'))

    x = xs[0]
    # # print(len(x) / 4)
    y = ys[0]

    # for i in range(100):
    #     plt.plot(x, ys[i], label=str(i))
    # #     # plt.plot(x, ys[i], label=curve_names[i])
    # # # plt.title(curve_names[0])
    # plt.legend()
    # plt.show()

    ############

    # name = 'DSC01034'
    datadir = 'D:\\Academic\\Data\\ELD-Sony-original\\Radiometric Calibration'
    fns = sorted(set(fn[:-4] for fn in os.listdir(datadir)))
    
    # ipdb.set_trace()

    # for name in fns:
        # process_color_checker(datadir, name)

    ############
    # name = 'DSC01034'
    
    # fns = ['DSC01038', 'DSC01039']
    color_samples_raw = np.array([[0, 0, 0], [1, 1, 1]])
    color_samples_rgb = np.array([[0, 0, 0], [1, 1, 1]])

    for name in fns:
        color_samples_raw = np.concatenate([color_samples_raw, np.loadtxt(join('color_samples', name+'_raw.txt'))])
        color_samples_rgb = np.concatenate([color_samples_rgb, np.loadtxt(join('color_samples', name+'_rgb.txt'))])

        # gamma = color_samples_raw ** (1 / 2.2)
        # plt.scatter(color_samples_raw[:, 2], color_samples_rgb[:, 2])
        # plt.scatter(color_samples_raw[:, 2], gamma[:, 2])
        # plt.title(name)

    for i in range(color_samples_rgb.shape[0]):
        if 1 in color_samples_rgb[i, :]:
            color_samples_rgb[i, :] = 1
            color_samples_raw[i, :] = 1
        if 0 in color_samples_rgb[i, :]:
            color_samples_rgb[i, :] = 0
            color_samples_raw[i, :] = 0
    
    color_samples_hsv = colour.RGB_to_HSV(color_samples_rgb)
    ind = color_samples_hsv[:, 1] < 0.5
    
    color_samples_raw = color_samples_raw[ind, :]
    color_samples_rgb = color_samples_rgb[ind, :]

    color_map = ['r', 'g', 'b']
    fs = []

    for channel in range(3):
        ind = np.argsort(color_samples_raw[:, channel])

        ind = ind[np.arange(0, 101, 1)]
        x = color_samples_raw[ind, channel]
        y = color_samples_rgb[ind, channel]

        Y = interpolate.interp1d(x, y)

        ind = np.arange(0, 1024, 50)
        x = E[ind]
        # y = ys[51][ind]
        y = Y(x)

        coef = np.matmul(H(x), (y - F0(x))) / len(ind) * 1024
        # np.savetxt('coeff_SonyA7S2.txt', coef)
        print(coef)

        f_est = f0 + np.matmul(coef, hs[:num_params, :])
        
        fs.append(f_est)

        F_est = interpolate.interp1d(E, f_est)
        
        # plt.plot(E, f0, label='f_0')
        # plt.plot(E, np.matmul(coef, hs[:num_params, :]))
        plt.plot(E, f_est, label='f_{}'.format(color_map[channel]), color=color_map[channel])

        # ipdb.set_trace()
        # plt.scatter(color_samples_raw[:, 0], color_samples_rgb[:, 0])
        # plt.scatter(color_samples_raw[:, 1], color_samples_rgb[:, 1])
        plt.scatter(color_samples_raw[:, channel], color_samples_rgb[:, channel], color=color_map[channel])
        # plt.scatter(color_samples_raw[ind, 2], color_samples_rgb[ind, 2])
        # plt.scatter(x, y)

        rmse = mean_squared_error(color_samples_rgb[:, channel], F_est(color_samples_raw[:, channel]))
        print(rmse)

        # ipdb.set_trace()
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.xlabel('Irradiance')
        plt.ylabel('Brightness')
        plt.legend()

    plt.show()

    fs = np.stack(fs)
    Fs = interpolate.interp1d(E, fs)

    np.savetxt('CRF_SonyA7S2_10.txt', fs)
    pass
