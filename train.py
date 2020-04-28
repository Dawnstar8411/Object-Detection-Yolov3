import csv
import time
import warnings

import torch.optim.lr_scheduler as lr_scheduler
from terminaltables import AsciiTable
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from config.args_train import *
from datasets import custom_transforms
from datasets.data_loader import ListDataset
from models.models import Darknet
from utils.parse_config import parse_data_config
from utils.utils import *

warnings.filterwarnings('ignore')
print("Start Training!")

torch.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("1. Path to save the output.")
save_path = Path(save_path_formatter(args))
args.save_path = 'checkpoints' / save_path
args.save_path.makedirs_p()
print("=> will save everything to {}".format(args.save_path))

print("2. Data Loading...")

data_config = parse_data_config(args.data_config)
train_path = data_config["train"]  # train.txt的路径
valid_path = data_config["valid"]  # valid.txt的路径
class_names = load_classes(data_config["names"])  # list, 包含所有类别的名字，和annotation中的编号顺序一致

train_transform = custom_transforms.Compose([
    custom_transforms.Random_horisontal_flip(0.5),
])

train_set = ListDataset(train_path, transform=train_transform, img_size=args.img_size, multiscale=False)
valid_set = ListDataset(valid_path, transform=None, img_size=args.img_size, multiscale=False)

print('{} samples found in train split'.format(len(train_set)))
print('{} samples found in valid split'.format(len(valid_set)))

train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.n_cpu, pin_memory=True,
                          collate_fn=train_set.collate_fn)
valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, num_workers=args.n_cpu, pin_memory=True,
                          collate_fn=valid_set.collate_fn)

print("3. Creating Model")
# Initiate model
model = Darknet(args.model_def).to(device)
model.apply(weights_init_normal)

# If specified we start from checkpoint

if args.pretrained_weights.endswith(".pth"):
    model.load_state_dict(torch.load(args.pretrained_weights))
else:
    model.load_darknet_weights(args.pretrained_weights)

print("4. Setting Optimization Solver")

optimizer = torch.optim.Adam(model.parameters(), lr=0.0002, weight_decay=1e-4)
exp_lr_scheduler_R = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.2)

print("5. Start Tensorboard")

# tensorboard --logdir=/path_to_log_dir/ --port 6006
writer = SummaryWriter(args.save_path)

print("6. Create csvfile to save log information.")

with open(args.save_path / args.log, 'w') as csvfile:
    csv_writer = csv.writer(csvfile, delimiter='\t')
    csv_writer.writerow(['val_precision', 'val_recall', 'val_map', 'val_f1'])  # 这里需要根据实际记录的内容更新

print("7. Start Training")

def main():
    best_error = -1

    for epoch in range(args.epochs):

        start_time = time.time()

        losses, loss_name = train(model, optimizer)

        precision, recall, AP, f1, ap_class = valid(model, iou_thres=args.iou_thres, conf_thres=args.conf_thres,
                                                    nms_thres=args.nms_thres, img_size=args.img_size)

        # store weights file
        decisive_error = AP.mean()
        if best_error < 0:
            best_error = decisive_error
        is_best = best_error < decisive_error
        best_error = max(best_error, decisive_error)
        torch.save(model.state_dict(), args.save_path / 'yolov3_pineapple_{}.pth'.format(epoch))
        # #
        print(is_best)
        if is_best:
            torch.save(model.state_dict(), args.save_path / 'yolov3_pineapple.pth')

        # write results to csv file
        with open(args.save_path / args.log, 'a') as csvfile:
            csv_writer = csv.writer(csvfile, delimiter='\t')
            csv_writer.writerow([precision.mean(), recall.mean(), AP.mean(), f1.mean()])

        # write results to tensorboard
        writer.add_scalar('Train/loss', losses[0], epoch)
        writer.flush()

        writer.add_scalar("Valid/precision", precision.mean(), epoch)
        writer.add_scalar("Valid/recall", recall.mean(), epoch)
        writer.add_scalar("Valid/mAP", AP.mean(), epoch)
        writer.add_scalar("Valid/f1", f1.mean(), epoch)
        writer.flush()

        # print to Terminal
        print("\n---- [Epoch {}/{}] ----".format(epoch, args.epochs))
        print("Total loss: {}".format(losses[0]))

        ap_table = [["Index", "Class name", "AP"]]
        for i, c in enumerate(ap_class):
            ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
        print(AsciiTable(ap_table).table)
        print(f"---- mAP {AP.mean()}")

        epoch_left = args.epochs - (epoch + 1)
        time_left = datetime.timedelta(seconds=epoch_left * (time.time() - start_time))
        print("---- ETA {}".format(time_left))


def valid(model, iou_thres, conf_thres, nms_thres, img_size):
    model.eval()

    labels = []
    sample_metrics=[]
    metrics = []
    for i, (_, imgs, targets) in enumerate(valid_loader):
        labels += targets[:, 1].tolist()  # label of ground Truth
        targets[:, 2:] = xywh2xyxy(targets[:, 2:])  # transfer (x,y,w,h) to (x1,y1,x2,y2)  top-left, right-bottom
        targets[:, 2:] *= img_size  # transfer coordinates to origenal scale
        imgs = imgs.to(device)
        with torch.no_grad():
            detections = model(imgs)
            # matrix, each row: (x1, y1, x2, y2, object_conf, class_score, class_pred)
            detections = non_max_suppression(detections, conf_thres=conf_thres, nms_thres=nms_thres)
        # batch_size, each sample :[true_positives(list), pred_scores(list), pred_labels(list)]
        sample_metrics += get_batch_statistics(detections, targets, iou_threshold=iou_thres)
    # three lists: true_positives, pred_scores, pred_labels
    true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    # the format of callback is "vector" corresponding to the class order in ap_class
    precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)
    return precision, recall, AP, f1, ap_class


def train(model, optimizer):
    loss_names = ['loss']
    losses = AverageMeter(i=len(loss_names))
    model.train()
    for i, (img_path, imgs, targets) in enumerate(train_loader):
        imgs = imgs.to(device)
        targets = targets.to(device)
        loss, _ = model(imgs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.update([loss.item()], args.batch_size)

    return losses.avg, loss_names


if __name__ == "__main__":
    main()
