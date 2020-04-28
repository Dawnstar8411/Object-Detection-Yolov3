import csv
import random
import time

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.ticker import NullLocator
from torch.utils.data import DataLoader

from config.args_detect import *
from datasets.data_loader import ImageFolder, ListDataset
from models.models import Darknet
from utils.parse_config import parse_data_config
from utils.utils import *

if __name__ == "__main__":

    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("1. Path to save the output")
    args.save_path = Path(args.save_path)
    args.save_path.makedirs_p()
    print("=> will save everything to {}".format(args.save_path))

    print("2. Data Loading...")
    if args.is_gt:
        data_config = parse_data_config(args.data_config)
        test_path = data_config["test"]
        class_names = load_classes(data_config["names"])  # list of all class names，in the same numbering order as the annotaion
        test_set = ListDataset(test_path, transform=None, img_size=args.img_size, multiscale=False)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.n_cpu,
                                 collate_fn=test_set.collate_fn)
    else:
        test_set = ImageFolder(args.image_folder, img_size=args.img_size)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.n_cpu)

    print("3. Creating Model!")
    model = Darknet(args.model_def).to(device=device)
    if args.weights_path:
        if args.weights_path.endswith(".pth"):
            model.load_state_dict(torch.load(args.weights_path))
        else:
            model.load_darknet_weights(args.weights_path)

    print("4. Create csvfile to save test results. ")

    with open(args.save_path / args.log, 'w') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter='\t')
        csv_writer.writerow('test results!')

    print("5. Start Detection!")

    model.eval()

    classes = load_classes(args.class_path)  # list of class names

    img_path_list = []  # store path of each image
    img_detections = []  # store detection results of each image
    labels = []  #
    sample_metrics = []  #

    prev_time = time.time()
    if not args.is_gt:
        for i, (img_paths, imgs) in enumerate(test_loader):
            imgs = imgs.to(device)

            with torch.no_grad():
                detections = model(imgs)
                detections = non_max_suppression(detections, args.conf_thres, args.nms_thres)

            img_path_list.extend(img_paths)
            img_detections.extend(detections)
            current_time = time.time()
            inference_time = datetime.timedelta(seconds=current_time - prev_time)
            prev_time = current_time
            print("\t+ Batch: {}, Inference Time: {}".format(i, inference_time))
    else:
        for i, (img_paths, imgs, targets) in enumerate(test_loader):
            labels += targets[:,1].tolist()  # label of ground Truth
            targets[:, 2:] = xywh2xyxy(targets[:, 2:])  # transfer (x,y,w,h) to (x1,y1,x2,y2)  top-left, right-bottom
            targets[:, 2:] *= args.img_size  # transfer coordinates to origenal scale

            imgs = imgs.to(device)

            with torch.no_grad():
                detections = model(imgs)
                # matrix, each row: (x1, y1, x2, y2, object_conf, class_score, class_pred)
                detections = non_max_suppression(detections, conf_thres=args.conf_thres, nms_thres=args.nms_thres)

            img_path_list.extend(img_paths)
            img_detections.extend(detections)
            # batch_size, each sample :[true_positives(list), pred_scores(list), pred_labels(list)]
            sample_metrics += get_batch_statistics(detections, targets, iou_threshold=args.iou_thres)
            current_time = time.time()
            inference_time = datetime.timedelta(seconds=current_time - prev_time)
            prev_time = current_time
            print("\t+ Batch: {}, Inference Time: {}".format(i, inference_time))

        # three lists: true_positives, pred_scores, pred_labels
        true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
        # the format of callback is "vector" corresponding to the class order in ap_class
        precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)

        print("Average Precisions:")
        for i, c in enumerate(ap_class):
            print(f"+ Class '{c}' ({class_names[c]}) - AP: {AP[i]}")

        print(f"mAP: {AP.mean()}")
        print(f"precision: {precision.mean()}")
        print(f"recall: {recall.mean()}")
        with open(args.save_path / args.log, 'a') as csvfile:
            csv_writer = csv.writer(csvfile, delimiter='\t')
            csv_writer.writerow(['mAP:',AP.mean()])

    # Bounding-box colors
    cmap = plt.get_cmap("tab20b")
    colors = [cmap(i) for i in np.linspace(0, 1, 20)]
    bbox_colors = [(0.2235294117647059, 0.23137254901960785, 0.4745098039215686, 1.0),
                   (0.9058823529411765, 0.5882352941176471, 0.611764705882353, 1.0)]

    print("5. Saving images！")
    # Iterate through images and save plot of detections
    for i, (path, detections) in enumerate(zip(img_path_list, img_detections)):

        print("{} Image: {}".format(i, path))
        with open(args.save_path / args.log, 'a') as csvfile:
            csv_writer = csv.writer(csvfile, delimiter='\t')
            csv_writer.writerow(['Image:',path])

        # read and display image
        img = np.array(Image.open(path))
        plt.figure()
        fig, ax = plt.subplots(1)
        ax.imshow(img)

        # add rectangles and class names to image
        if detections is not None:
            # resize detection results to original scale
            detections = rescale_boxes(detections, args.img_size, img.shape[:2])

            unique_labels = detections[:, -1].cpu().unique()  # the last colume represents the class label
            #n_cls_preds = len(unique_labels)
            #bbox_colors = random.sample(colors, n_cls_preds)
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                print("\t+ Label: {}, Conf: {}".format(classes[int(cls_pred)], cls_conf.item()))
                with open(args.save_path / args.log, 'a') as csvfile:
                    csv_writer = csv.writer(csvfile, delimiter='\t')
                    csv_writer.writerow(['\t+Label:',classes[int(cls_pred)],'Conf:', cls_conf.item()])
                box_w = x2 - x1  # x1：column of left-up point, x2: column of right-bottom point
                box_h = y2 - y1  # y1: row of left-up point,    y2: row of right-bottom point
                #color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]  # color of rectangle
                color = bbox_colors[int(cls_pred)]
                bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=0.4, edgecolor=color, facecolor="none")
                ax.add_patch(bbox)
                plt.text(
                    x1,
                    y1-35,
                    s=classes[int(cls_pred)],
                    color="white",
                    fontsize=4,
                    verticalalignment="top",
                    bbox={"color": color, "pad": 0},
                )

        # Save generated image with detections
        plt.axis("off")
        plt.gca().xaxis.set_major_locator(NullLocator())
        plt.gca().yaxis.set_major_locator(NullLocator())
        filename = path.split("/")[-1].split(".")[0]  # name of image
        plt.savefig(f"submissions/{filename}.jpg", dpi=840, bbox_inches="tight", pad_inches=0.0)  # save image
        plt.close()
