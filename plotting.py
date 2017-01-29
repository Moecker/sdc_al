import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import cv2

kOutputFolder = 'output_images/'

def plot_images(image, adopted_image, name, is_gray=False):    
    # First image
    plt.clf()
    plt.subplot(211)
    
    if not is_gray:
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(image, cmap='gray')
        
    plt.xlim([0, image.shape[1]])
    plt.ylim([image.shape[0], 0])

    # Second image
    plt.subplot(212)
    
    if not is_gray:
        plt.imshow(cv2.cvtColor(adopted_image, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(adopted_image, cmap='gray')
        
    plt.xlim([0, adopted_image.shape[1]])
    plt.ylim([adopted_image.shape[0], 0])
        
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    plt.title(name)
    
    plt.savefig(kOutputFolder + name + "_plot.png")
    return plt


def save_plot(plot, name):
    plot.savefig(kOutputFolder + name)