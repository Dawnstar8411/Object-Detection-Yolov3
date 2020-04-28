import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--image_folder", type=str, default="/home/deep/chen/Datasets/Pineapple/samples/",
                    help="path to dataset")
parser.add_argument("--model_def", type=str, default="config/yolov3-pineapple.cfg",
                    help="path to model definition file")
parser.add_argument("--weights_path", type=str, default="weights/yolov3_pineapple.pth", help="path to weights file")
parser.add_argument("--class_path", type=str, default="/home/deep/chen/Datasets/Pineapple/train/classes.names",
                    help="path to class label file")
parser.add_argument("--data_config", type=str, default="config/pineapple.data", help="path to data config file")
parser.add_argument("--save_path", type=str, default="submissions", help="path to save detection results")
parser.add_argument("--log", type=str, default="test_result.csv")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--n_cpu", type=int, default=1, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_size", type=int, default=1280, help="size of each image dimension")

parser.add_argument("--conf_thres", type=float, default=0.5, help="object confidence threshold")
parser.add_argument("--nms_thres", type=float, default=0.5, help="iou thresshold for non-maximum suppression")
parser.add_argument("--iou_thres", type=float, default=0.5, help="iou threshold required to qualify as detected")

parser.add_argument("--is_gt", type=bool, default=True, help="detection mode")
args = parser.parse_args()
