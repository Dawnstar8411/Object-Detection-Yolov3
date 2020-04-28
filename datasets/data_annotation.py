import os

import numpy as np
from PIL import Image

if __name__ == "__main__":
    #filelist = os.listdir('/home/deep/chen/Datasets/Pineapple/train/images/')
    filelist = ['G0031952.JPG']
    for f in filelist:
        anno_txt = open('/home/deep/chen/Datasets/Pineapple/train/labels/' + f[0:-4] + '.txt', 'w')
        fruit_path = '/home/deep/chen/Datasets/Pineapple/annotation/' + f[0:-4] + '_Fruit.jpg'
        flower_path = '/home/deep/chen/Datasets/Pineapple/annotation/' + f[0:-4] + '_Flower.jpg'
        print(f)
        for index in range(2):
            if index == 0:
                img_path = fruit_path
            else:
                img_path = flower_path

            if not os.path.exists(img_path):
                continue
            img = Image.open(img_path)
            img_gray = img.convert('L')
            img_array = np.array(img_gray)

            height, width = np.shape(img_array)

            for h in range(height):
                for w in range(width):
                    if img_array[h, w] > 200:
                        img_array[h, w] = 255
                    elif img_array[h, w] < 50:
                        img_array[h, w] = 0

            for w in range(width - 1):
                for h in range(height - 1):
                    if (h == 0 and img_array[h, w] == 0 and img_array[h, w + 1] == 0 and img_array[h + 1, w] == 0 and
                        img_array[h, w - 1] == 255) or (
                            w == 0 and img_array[h, w] == 0 and img_array[h, w + 1] == 0 and img_array[
                        h + 1, w] == 0 and img_array[h - 1, w] == 255) or (
                            img_array[h, w] == 0 and img_array[h, w + 1] == 0 and img_array[h + 1, w] == 0 and
                            img_array[h - 1, w] == 255 and img_array[h, w - 1] == 255):
                        for i in range(width - w + 1):
                            if ((w + i) < width and img_array[h, w + i] == 255) or (
                                    (w + i) == width and img_array[h, w + i - 1] == 0):
                                top_right_x = w + i - 1
                                break
                        for j in range(height - h + 1):
                            if ((h + j) < height and img_array[h + j, w] == 255) or (
                                    (h + j) == height and img_array[h + j - 1, w] == 0):
                                bottom_left_y = h + j - 1
                                break
                        label_idx = str(index)
                        x_center = str((w + top_right_x) / 2 / width)
                        y_center = str((h + bottom_left_y) / 2 / height)
                        wid = str((top_right_x - w) / width)
                        hei = str((bottom_left_y - h) / height)
                        anno_txt.write("{} {} {} {} {}\n".format(label_idx, x_center, y_center, wid, hei))
        anno_txt.close()
