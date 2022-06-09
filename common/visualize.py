import matplotlib.pyplot as plt
import cv2
from skimage.metrics import structural_similarity as compare_ssim
import numpy as np

def save_result(heatmap, save_path):
    plt.rcParams['savefig.dpi'] = 300
    plt.imsave(save_path, heatmap)


def compute_ssim(img_path, base_path):
    '''
    :param ref_path:
    :param dist_path:
    :return: SSIM score of each image, each layer
    [[image1 - layer30(original),layer28 ... ],
    [image2 - layer30(original), layer28 ...]
    ..]
    '''

    x = [30,28,26,24,21,19,17,14,12,10,7,5,2,0]
    labels = ['original', 'conv5-3', 'conv5-2', 'conv5-1', 'conv4-3', 'conv4-2', 'conv4-1', 'conv3-3', 'conv3-2', 'conv3-1', 'conv2-2', 'conv2-1', 'conv1-2', 'conv1-1']
    dummy = [30 * 13 / 13, 30 * 12 / 13, 30 * 11 / 13, 30 * 10 / 13, 30 * 9 / 13, 30 * 8 / 13, 30 * 7 / 13, 30 * 6 / 13, 30 * 5 / 13, 30 * 4 / 13, 30 * 3 / 13, 30 * 2 / 13, 30 * 1 / 13, 0]

    ssim_ls = []
    for path in img_path:
        ref_path = base_path + path.split('/')[-1].split('_')[2].split('.')[0] + '/30.jpeg'
        ref = cv2.imread(ref_path)
        ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY) / 255

        ssim_individual = []
        for layer in x: #for vgg16
            dist_path = base_path + path.split('/')[-1].split('_')[2].split('.')[0] + '/' +str(layer) + '.jpeg'
            dist = cv2.imread(dist_path)
            dist = cv2.cvtColor(dist, cv2.COLOR_BGR2GRAY) / 255

            score, diff = compare_ssim(ref, dist, full=True)
            diff = (diff * 255).astype("uint8")
            ssim_individual.append(score)

        ssim_ls.append(ssim_individual)

        plt.title(dist_path.split('/')[6])
        plt.plot(x, ssim_individual, marker='D')
        plt.xticks(dummy, labels, rotation=40)
        plt.yticks([1.0, 0.9, 0.8, 0.7, 0.6])
        plt.gca().invert_xaxis()
        plt.savefig('/'.join(dist_path.split('/')[:7]) + '/ssim.jpeg', bbox_inches = 'tight')
        plt.show()



    return ssim_ls


def plot_ssim(ssim_ls, method, save_path):
    x = ['30','28','26','24','21','19','17','14','12','10','7','5','2','0']
    labels = ['original', 'conv5-3', 'conv5-2', 'conv5-1', 'conv4-3', 'conv4-2', 'conv4-1', 'conv3-3', 'conv3-2',
              'conv3-1', 'conv2-2', 'conv2-1', 'conv1-2', 'conv1-1']
    dummy = [30 * 13 / 13, 30 * 12 / 13, 30 * 11 / 13, 30 * 10 / 13, 30 * 9 / 13, 30 * 8 / 13, 30 * 7 / 13, 30 * 6 / 13, 30 * 5 / 13, 30 * 4 / 13, 30 * 3 / 13, 30 * 2 / 13, 30 * 1 / 13, 0]

    mean_ssim = np.array(ssim_ls).mean(axis=0)
    dummy_np = np.array(dummy)
    print(mean_ssim)
    plt.plot(dummy_np, mean_ssim, marker='D')
    plt.xticks(dummy, labels, rotation=40)
    plt.yticks([1.0, 0.9, 0.8, 0.7, 0.6])
    plt.title(method)
    plt.ylabel('SSIM score: vgg16')
    plt.xlabel('features randomized')
    plt.gca().invert_xaxis()
    plt.savefig(save_path, bbox_inches = 'tight')

    plt.show()
