import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--model_def", type=str, default="config/yolov3-pineapple.cfg",
                    help="path to model definition file")
parser.add_argument("--data_config", type=str, default="config/pineapple.data", help="path to data config file")
parser.add_argument("--pretrained_weights", type=str, default="weights/yolov3_pineapple_34.pth",
                    help="if specified starts from checkpoint model")
parser.add_argument("--log", type=str, default="logs")
parser.add_argument("--dataset_name", type=str, default="Pineapple")
parser.add_argument("--model_name", type=str, default="Darknet")

parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
parser.add_argument("--batch_size", type=int, default=2, help="size of each image batch")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_size", type=int, default=1280, help="size of each image dimension")
parser.add_argument("--img_channel", type=int, default=3, help="size of image channel")

parser.add_argument("--conf_thres", type=float, default=0.5, help="object confidence threshold")
parser.add_argument("--nms_thres", type=float, default=0.5, help="iou thresshold for non-maximum suppression")
parser.add_argument("--iou_thres", type=float, default=0.5, help="iou threshold required to qualify as detected")

parser.add_argument("--multiscale_training", default=False, help="allow for multi-scale training")

parser.add_argument("--is_debug", default=False)
args = parser.parse_args()
