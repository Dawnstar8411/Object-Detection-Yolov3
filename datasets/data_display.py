import os

import numpy as np
from PIL import Image
import csv
import random
from path import Path
from matplotlib.ticker import NullLocator
import matplotlib.pyplot as plt

import matplotlib.patches as patches

if __name__ == "__main__":
    cmap = plt.get_cmap("tab20b")
    colors = [cmap(i) for i in np.linspace(0, 1, 5)]
    bbox_colors = random.sample(colors, 2)
    save_path = Path('/home/deep/chen/Codes/YOLO-v3/annotations')
    save_path.makedirs_p()

    filelist = os.listdir('/home/deep/chen/Datasets/Pineapple/train/images/')
    with open(save_path / 'annotation.csv', 'w') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter='\t')
        csv_writer.writerow(['name','index','label'])
    for f in filelist:
        print(f)
        anno_txt = '/home/deep/chen/Datasets/Pineapple/train/labels/' + f[0:-4] + '.txt'
        img_path= '/home/deep/chen/Datasets/Pineapple/train/images/' + f
        img = Image.open(img_path)
        img_array = np.array(img)
        height, width,channel = np.shape(img_array)

        file = open(anno_txt, 'r')
        lines = file.read().split('\n')
        lines = [x.rstrip().lstrip() for x in lines]  # get rid of fringe whitespaces
        lines = lines[:-1]
        plt.figure()
        fig, ax = plt.subplots(1)
        ax.imshow(img_array)

        for i, anno in enumerate(lines):
            index = i
            temp = anno.split(" ")

            if temp[0] is '0':
                box_class = 'fruit'
            else:
                box_class = 'flower'
            x_center = float(temp[1])
            y_center = float(temp[2])
            box_w = float(temp[3])
            box_h = float(temp[4])
            x1 = (x_center - box_w/2 ) * width
            y1 = (y_center - box_h/2) * height
            box_w = box_w * width
            box_h = box_h * height

            with open(save_path / 'annotation.csv', 'a') as csvfile:
                csv_writer = csv.writer(csvfile, delimiter='\t')
                csv_writer.writerow([f,index,box_class])

            color = bbox_colors[int(temp[0])]
            bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=1, edgecolor=color, facecolor="none")
            ax.add_patch(bbox)
            plt.text(
                x1,
                y1,
                s=index,
                color="white",
                fontsize=6,
                verticalalignment="top",
                bbox={"color": color, "pad": 0},
            )
        plt.axis("off")
        plt.gca().xaxis.set_major_locator(NullLocator())
        plt.savefig(save_path / f, dpi=840, bbox_inches="tight", pad_inches=0.0)  # save image
        plt.close()