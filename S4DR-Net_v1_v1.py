from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from data import S3DIS_DX
from model import S4DRNetv1_v1
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler
from util import cal_loss, IOStream
import sklearn.metrics as metrics

from sklearn.metrics import accuracy_score, recall_score, f1_score #计算其余指标

from tqdm import tqdm
import time


def calculate_metrics(true_labels, pred_labels, num_classes=8): #计算其余指标
    class_acc = []
    class_recall = []
    class_f1 = []

    for i in range(num_classes):  # Assuming there are 9 classes
        class_true = (true_labels == i)
        class_pred = (pred_labels == i)

        # 计算每个类别的准确度
        class_acc.append(accuracy_score(class_true, class_pred))

        # 计算每个类别的召回率
        class_recall.append(recall_score(class_true, class_pred))

        # 计算每个类别的F1分数
        class_f1.append(f1_score(class_true, class_pred))

    return class_acc, class_recall, class_f1 #

def _init_():
    args.exp_name = args.exp_name + time.strftime("_%m_%d_%H_%M")

    if not os.path.exists('outputs'):
        os.makedirs('outputs')
    if not os.path.exists('outputs/' + args.exp_name):
        os.makedirs('outputs/' + args.exp_name)
    if not os.path.exists('outputs/' + args.exp_name + '/' + 'models'):
        os.makedirs('outputs/' + args.exp_name + '/' + 'models')
    os.system('cp train.py outputs' + '/' +
              args.exp_name + '/' + 'train.py.backup')
    os.system('cp model.py outputs' + '/' +
              args.exp_name + '/' + 'model.py.backup')
    os.system('cp util.py outputs' + '/' +
              args.exp_name + '/' + 'util.py.backup')
    os.system('cp part_utils.py outputs' + '/' +
              args.exp_name + '/' + 'part_util.py.backup')
    os.system('cp data.py outputs' + '/' +
              args.exp_name + '/' + 'data.py.backup')


def calculate_sem_IoU(pred_np, seg_np, visual=False):
    I_all = np.zeros(8)
    U_all = np.zeros(8)
    for sem_idx in range(seg_np.shape[0]):
        for sem in range(8):
            I = np.sum(np.logical_and(pred_np[sem_idx] == sem, seg_np[sem_idx] == sem))
            U = np.sum(np.logical_or(pred_np[sem_idx] == sem, seg_np[sem_idx] == sem))
            I_all[sem] += I
            U_all[sem] += U
    if visual:
        for sem in range(8):
            if U_all[sem] == 0:
                I_all[sem] = 1
                U_all[sem] = 1
    return I_all / U_all


def train(args, io):
    train_loader = DataLoader(
        S3DIS_DX(partition='train', num_points=args.num_points, test_area=args.test_area, split_num=args.split_num),
        num_workers=8, batch_size=args.batch_size, shuffle=True, drop_last=True)

    test_loader = DataLoader(
        S3DIS_DX(partition='test', num_points=args.num_points, test_area=args.test_area, split_num=args.split_num),
        num_workers=8, batch_size=args.test_batch_size, shuffle=False, drop_last=False)

    device = torch.device("cuda" if args.cuda else "cpu")

    model = S4DRNetv1_v1(args).to(device)
    print(str(model))
    model = nn.DataParallel(model)
    print("Let's use", torch.cuda.device_count(), "GPUs!")

    if args.pretrain: #第一步
        model.module.freeze_layers_1()
        print("预训练1 冻结分类头")

    # Load pretrained weights if not in pretraining mode
    if not args.pretrain:  # 第二步
        pretrained_dict = torch.load(f'/best_pretrain_model.t7')  #加载预训练模型
        model_dict = model.state_dict()

        # 排除不需要加载的层，例如 conv6, conv7, conv8, conv9
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                           k in model_dict and 'conv6' not in k and 'conv7' not in k and 'conv8' not in k and 'conv9' not in k}

        # 更新模型的权重字典
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print("预训练2 加载预训练后的特征提取层")

    # Freeze parameters if in fine-tuning mode

    # Select optimizer
    if args.use_sgd:
        opt = optim.SGD(model.parameters(), lr=args.lr * 100, momentum=args.momentum, weight_decay=1e-4)
    else:
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # Learning rate scheduler
    if args.scheduler == 'cos':
        scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=1e-3)
    elif args.scheduler == 'step':
        scheduler = StepLR(opt, step_size=20, gamma=0.5)

    best_test_iou = 0
    best_hop_acc = 0
    for epoch in range(args.epochs):
        ####################
        # Train
        ####################
        train_loss = 0.0
        train_loss_hop = 0.0
        train_loss_seg = 0.0
        count = 0.0
        model.train()
        train_true_cls = []
        train_pred_cls = []
        train_true_seg = []
        train_pred_seg = []
        train_pred_hop = []
        train_true_hop = []

        for data, seg, p2v_indices, part_distance, part_distance_sp, part_rand_idx in tqdm(train_loader):
            data, seg = data.to(device), seg.to(device)

            dis = part_distance  #
            dis_sp = part_distance_sp
            dis.long().to(device)
            dis_sp.long().to(device)  #

            p2v_indices = p2v_indices.long().to(device)
            part_num = part_distance.shape[1]
            triu_idx = torch.triu_indices(part_num, part_num)
            part_distance = part_distance[:, triu_idx[0], triu_idx[1]]
            part_distance = part_distance.long().to(device)
            part_rand_idx = part_rand_idx.to(device)
            part_num_sp = part_distance_sp.shape[1]
            triu_idx_sp = torch.triu_indices(part_num_sp, part_num_sp)
            part_distance_sp = part_distance_sp[:, triu_idx_sp[0], triu_idx_sp[1]]
            part_distance_sp = part_distance_sp.long().to(device)

            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            opt.zero_grad()
            seg_pred, hop_logits_list, hop_logits_list_sp = model(data, p2v_indices, part_rand_idx)

            # Adjust segmentation prediction dimensions
            seg_pred = seg_pred.permute(0, 2, 1).contiguous()

            # Calculate segmentation loss
            loss_seg = cal_loss(seg_pred.view(-1, 8), seg.view(-1, 1).squeeze(), smoothing=True)

            # Handle hop_logits_list
            lasthop_logits = hop_logits_list[-1]
            lasthop_logits = (lasthop_logits + lasthop_logits.permute(0, 1, 3, 2)) / 2
            lasthop_logits = lasthop_logits[:, :, triu_idx[0], triu_idx[1]]
            hop_loss = F.cross_entropy(lasthop_logits, part_distance, label_smoothing=0.2)

            if not args.single_hoploss:
                for hop_logits in hop_logits_list[:-1]:
                    hop_logits = (hop_logits + hop_logits.permute(0, 1, 3, 2)) / 2
                    hop_logits = hop_logits[:, :, triu_idx[0], triu_idx[1]]
                    hop_loss += F.cross_entropy(hop_logits, part_distance, label_smoothing=0.2)
                hop_loss /= len(hop_logits_list)

            # Handle hop_logits_list_sp
            lasthop_logits_sp = hop_logits_list_sp[-1]
            lasthop_logits_sp = (lasthop_logits_sp + lasthop_logits_sp.permute(0, 1, 3, 2)) / 2
            lasthop_logits_sp = lasthop_logits_sp[:, :, triu_idx_sp[0], triu_idx_sp[1]]
            hop_loss_sp = F.cross_entropy(lasthop_logits_sp, part_distance_sp, label_smoothing=0.2)

            if not args.single_hoploss:
                for hop_logits_sp in hop_logits_list_sp[:-1]:
                    hop_logits_sp = (hop_logits_sp + hop_logits_sp.permute(0, 1, 3, 2)) / 2
                    hop_logits_sp = hop_logits_sp[:, :, triu_idx_sp[0], triu_idx_sp[1]]
                    hop_loss_sp += F.cross_entropy(hop_logits_sp, part_distance_sp, label_smoothing=0.2)
                hop_loss_sp /= len(hop_logits_list_sp)

            # Combine losses
            if args.pretrain:
                loss = (hop_loss / hop_loss.detach()) + (hop_loss_sp / hop_loss_sp.detach())
            else:
                loss = (loss_seg / loss_seg.detach())

            loss.backward()
            opt.step()

            pred = seg_pred.max(dim=2)[1]
            count += batch_size
            train_loss += loss.item() * batch_size
            train_loss_hop += hop_loss.item() * batch_size
            train_loss_seg += loss_seg.item() * batch_size

            seg_np = seg.cpu().numpy()
            pred_np = pred.detach().cpu().numpy()
            train_true_cls.append(seg_np.reshape(-1))
            train_pred_cls.append(pred_np.reshape(-1))
            train_true_seg.append(seg_np)
            train_pred_seg.append(pred_np)
            train_pred_hop.append(lasthop_logits.max(dim=1)[1].detach().cpu().numpy())
            train_true_hop.append(part_distance.cpu().numpy())

        if args.scheduler == 'cos':
            scheduler.step()
        elif args.scheduler == 'step':
            scheduler.step()

        # Compute metrics
        train_true_cls = np.concatenate(train_true_cls)
        train_pred_cls = np.concatenate(train_pred_cls)
        train_acc = metrics.accuracy_score(train_true_cls, train_pred_cls)
        avg_per_class_acc = metrics.balanced_accuracy_score(train_true_cls, train_pred_cls)
        train_true_seg = np.concatenate(train_true_seg, axis=0)
        train_pred_seg = np.concatenate(train_pred_seg, axis=0)
        train_ious = calculate_sem_IoU(train_pred_seg, train_true_seg)
        train_true_hop = np.concatenate(train_true_hop).reshape(-1)
        train_pred_hop = np.concatenate(train_pred_hop).reshape(-1)
        train_hop_acc = metrics.accuracy_score(train_true_hop, train_pred_hop)

        # Print logs
        outstr = 'Train %d\nsegmentation loss:  %.6f, hop loss:  %.6f, multi-task loss: %.6f\ntrain acc: %.6f, train avg acc: %.6f, train iou: %.6f, hop train acc: %.6f' % (
        epoch,
        train_loss_seg * 1.0 / count,
        train_loss_hop * 1.0 / count,
        train_loss * 1.0 / count,
        train_acc,
        avg_per_class_acc,
        np.mean(train_ious),
        train_hop_acc)
        io.cprint(outstr)

        ####################
        # Test
        ####################
        # Testing code remains unchanged, as the test process will work for both phases.
        # This includes calculating test loss and metrics.
        test_loss = 0.0
        test_loss_seg = 0.0
        test_loss_hop = 0.0
        count = 0.0
        model.eval()

        test_true_cls = []
        test_pred_cls = []
        test_true_seg = []
        test_pred_seg = []
        test_pred_hop = []
        test_pred_hop_sp = []
        test_true_hop = []
        test_true_hop_sp = []


        # Disable gradient calculation during evaluation
        with torch.no_grad():
            for data, seg, p2v_indices, part_distance, part_distance_sp, part_rand_idx in tqdm(test_loader):
                data, seg = data.to(device), seg.to(device)

                dis = part_distance  #
                dis_sp = part_distance_sp
                dis.long().to(device)
                dis_sp.long().to(device)  #

                p2v_indices = p2v_indices.long().to(device)

                part_num = part_distance.shape[1]
                triu_idx = torch.triu_indices(part_num, part_num)
                part_distance = part_distance[:, triu_idx[0], triu_idx[1]]
                part_distance = part_distance.long().to(device)
                part_rand_idx = part_rand_idx.to(device)

                part_num_sp = part_distance_sp.shape[1]
                triu_idx_sp = torch.triu_indices(part_num_sp, part_num_sp)
                part_distance_sp = part_distance_sp[:, triu_idx_sp[0], triu_idx_sp[1]]
                part_distance_sp = part_distance_sp.long().to(device)

                data = data.permute(0, 2, 1)
                batch_size = data.size()[0]

                seg_pred, hop_logits_list, hop_logits_list_sp = model(data, p2v_indices, part_rand_idx)
                seg_pred = seg_pred.permute(0, 2, 1).contiguous()

                # Calculate segmentation loss
                loss_seg = cal_loss(seg_pred.view(-1, 8), seg.view(-1, 1).squeeze())

                # Handle hop_logits_list
                lasthop_logits = hop_logits_list[-1]
                lasthop_logits = (lasthop_logits + lasthop_logits.permute(0, 1, 3, 2)) / 2
                lasthop_logits = lasthop_logits[:, :, triu_idx[0], triu_idx[1]]
                hop_loss = F.cross_entropy(lasthop_logits, part_distance, label_smoothing=0.2)

                if not args.single_hoploss:
                    for hop_logits in hop_logits_list[:-1]:
                        hop_logits = (hop_logits + hop_logits.permute(0, 1, 3, 2)) / 2
                        hop_logits = hop_logits[:, :, triu_idx[0], triu_idx[1]]
                        hop_loss += F.cross_entropy(hop_logits, part_distance, label_smoothing=0.2)
                    hop_loss /= len(hop_logits_list)

                # Handle hop_logits_list_sp
                lasthop_logits_sp = hop_logits_list_sp[-1]
                lasthop_logits_sp = (lasthop_logits_sp + lasthop_logits_sp.permute(0, 1, 3, 2)) / 2
                lasthop_logits_sp = lasthop_logits_sp[:, :, triu_idx_sp[0], triu_idx_sp[1]]
                hop_loss_sp = F.cross_entropy(lasthop_logits_sp, part_distance_sp, label_smoothing=0.2)

                if not args.single_hoploss:
                    for hop_logits_sp in hop_logits_list_sp[:-1]:
                        hop_logits_sp = (hop_logits_sp + hop_logits_sp.permute(0, 1, 3, 2)) / 2
                        hop_logits_sp = hop_logits_sp[:, :, triu_idx_sp[0], triu_idx_sp[1]]
                        hop_loss_sp += F.cross_entropy(hop_logits_sp, part_distance_sp, label_smoothing=0.2)
                    hop_loss_sp /= len(hop_logits_list_sp)

                loss = loss_seg + hop_loss + hop_loss_sp

                pred = seg_pred.max(dim=2)[1]
                count += batch_size
                test_loss += loss.item() * batch_size
                test_loss_seg += loss_seg.item() * batch_size
                test_loss_hop += hop_loss.item() * batch_size

                seg_np = seg.cpu().numpy()
                pred_np = pred.detach().cpu().numpy()
                test_true_cls.append(seg_np.reshape(-1))
                test_pred_cls.append(pred_np.reshape(-1))
                test_true_seg.append(seg_np)
                test_pred_seg.append(pred_np)

                test_pred_hop.append(lasthop_logits.max(dim=1)[1].detach().cpu().numpy())
                test_pred_hop_sp.append(lasthop_logits_sp.max(dim=1)[1].detach().cpu().numpy())

                test_true_hop.append(part_distance.cpu().numpy())
                test_true_hop_sp.append(part_distance_sp.cpu().numpy())

        # Compute metrics
        test_true_cls = np.concatenate(test_true_cls)
        test_pred_cls = np.concatenate(test_pred_cls)
        test_acc = metrics.accuracy_score(test_true_cls, test_pred_cls)
        avg_per_class_acc = metrics.balanced_accuracy_score(test_true_cls, test_pred_cls)
        test_true_seg = np.concatenate(test_true_seg, axis=0)
        test_pred_seg = np.concatenate(test_pred_seg, axis=0)
        test_ious = calculate_sem_IoU(test_pred_seg, test_true_seg)

        class_acc, class_recall, class_f1 = calculate_metrics(test_true_cls, test_pred_cls) #计算其余指标

        for idx, iou in enumerate(test_ious):
            print(f"Class {idx} IoU: {iou:.6f}")
            print(f"Class {idx} Accuracy: {class_acc[idx]:.6f}")  #输出其余指标
            print(f"Class {idx} Recall: {class_recall[idx]:.6f}")
            print(f"Class {idx} F1 Score: {class_f1[idx]:.6f}")

        test_true_hop = np.concatenate(test_true_hop).reshape(-1)
        test_true_hop_sp = np.concatenate(test_true_hop_sp).reshape(-1)

        test_pred_hop = np.concatenate(test_pred_hop).reshape(-1)
        test_pred_hop_sp = np.concatenate(test_pred_hop_sp).reshape(-1)

        test_hop_acc = metrics.accuracy_score(test_true_hop, test_pred_hop)
        test_hop_acc_sp = metrics.accuracy_score(test_true_hop_sp, test_pred_hop_sp)

        test_hop_acc = (test_hop_acc + test_hop_acc_sp) / 2

        # Output results
        outstr = f'Test {epoch}\nsegmentation loss: {test_loss_seg * 1.0 / count:.6f}, hop loss: {test_loss_hop * 1.0 / count:.6f}, multi-task loss: {test_loss * 1.0 / count:.6f}\ntest acc: {test_acc:.6f}, test avg acc: {avg_per_class_acc:.6f}, test iou: {np.mean(test_ious):.6f}, hop test acc: {test_hop_acc:.6f}'
        io.cprint(outstr)
        # Save model with best IoU

        if args.pretrain:
            if test_hop_acc > best_hop_acc:
                best_hop_acc = test_hop_acc
                torch.save(model.state_dict(), f'outputs/{args.exp_name}/models/best_pretrain_model.t7')
                io.cprint(f"Saved best pretrain model with hop acc: {best_hop_acc:.6f}")

        if np.mean(test_ious) >= best_test_iou:
            best_test_iou = np.mean(test_ious)


        outstr = f'Best test IoU: {best_test_iou:.6f}'
        io.cprint(outstr)


if __name__ == "__main__":
    # Add argument for pretraining
    parser = argparse.ArgumentParser(description='Point Cloud Part Segmentation')
    parser.add_argument('--exp_name', default='S4DRNet_train', type=str, metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--pretrain_exp_name', type=str, default='S4DRNet_pretrain', metavar='N',
                        help='Name of the pretraining experiment')
    parser.add_argument('--pretrain', action='store_true', default=False,
                        help='Use self-supervised pretraining')
    parser.add_argument('--dataset', type=str, default='S3DIS', metavar='N',
                        choices=['S3DIS'])
    parser.add_argument('--batch_size', type=int, default=12, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=8, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', type=bool, default=True,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--scheduler', type=str, default='cos', metavar='N',
                        choices=['cos', 'step'],
                        help='Scheduler to use, [cos, step]')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--num_points', type=int, default=4096,
                        help='num of points to use')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use')
    parser.add_argument('--gpu', type=str, default="7", metavar='N',
                        help='Cuda id')

    # Additional arguments for training
    parser.add_argument('--split_num', type=int, default=5, metavar='N',
                        help='Voxel split number')
    parser.add_argument('--single_hoploss', type=bool, default=False, metavar='N',
                        help='if only use the last hop loss')
    parser.add_argument('--sigma2', type=float, default=1.0, metavar='N',
                        help='sigma2 in gauss kernel')
    parser.add_argument('--test_area', type=str, default=5, metavar='N',
                        choices=['1', '2', '3', '4', '5', '6', 'all'])

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    _init_()

    io = IOStream(f'outputs/{args.exp_name}/run.log')
    io.cprint(str(args))

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        io.cprint(
            f'Using GPU: {torch.cuda.current_device()} from {torch.cuda.device_count()} devices')
        torch.cuda.manual_seed(args.seed)
    else:
        io.cprint('Using CPU')

    train(args, io)
