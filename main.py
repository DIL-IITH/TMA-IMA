import argparse

import time

import math

from PIL import Image
import numpy as np

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
import os

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"

torch.set_num_threads(4)
torch.set_num_interop_threads(4)

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
import torchvision.models as models

from clip.custom_clip import getimagematrixadapter, gettextmatrixadapter, getzeroshotclip

from data_utils.cls_to_names import *
from data_utils.fewshot_dataset import fewshot_datasets, build_fewshot_dataset
from data_utils.imagenet_variants import thousand_k_to_200, imagenet_a_mask, imagenet_r_mask, imagenet_v_mask
from data_utils.imagenet_prompts import imagenet_classes

from data_utils.apply_aug import AugMixAugmenter
from data_utils.dataset import CustomImageFolder

from episodic_methods.ima import IMA
from episodic_methods.tma import TMA

import time

def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s

def zero_shot(args):
    normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                     std=[0.26862954, 0.26130258, 0.27577711])
    
    dset = args.test_sets

    base_transform = transforms.Compose([
                transforms.Resize(args.resolution, interpolation=BICUBIC),
                transforms.CenterCrop(args.resolution)])

    preprocess = transforms.Compose([
                transforms.ToTensor(),
                normalize
    ])

    data_transform = AugMixAugmenter(base_transform, preprocess, n_views=args.batch_size-1, use_augmix=True, severity=1)

    batchsize = 1
    if dset.lower() in [d.lower() for d in fewshot_datasets]:
        val_dataset = build_fewshot_dataset(
            set_id=dset, 
            root=args.testdir,
            transform=data_transform,
            mode=args.dataset_mode
        )
    else:
        val_dataset = CustomImageFolder(    
            root=args.testdir,
            transform=data_transform
        )

    log_str = ("Number of test samples: {}".format(len(val_dataset)))
    args.out_file.write(log_str + "\n")
    args.out_file.flush()

    val_loader = DataLoader( #list of tuples where first element is list of tensors with first tensor corresponding to original img and second element is label
        val_dataset,
        batch_size=batchsize,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True
    )

    print("evaluating: {}".format(args.test_sets))

    if len(args.test_sets) > 1: 
        # fine-grained classification datasets
        classnames = eval("{}_classes".format(args.test_sets.lower()))
    else:
        assert args.test_sets in ['A', 'R', 'K', 'V', 'I']
        classnames_all = imagenet_classes
        classnames = []
        if args.test_sets in ['A', 'R', 'V']:
            label_mask = eval("imagenet_{}_mask".format(args.test_sets.lower()))
            if args.test_sets == 'R':
                for i, m in enumerate(label_mask):
                    if m:
                        classnames.append(classnames_all[i])
            else:
                classnames = [classnames_all[i] for i in label_mask]
        else:
            classnames = classnames_all

    model = getzeroshotclip(args, clip_arch=args.arch, classnames=classnames, device=args.gpu, n_ctx=args.n_ctx, ctx_init=args.ctx_init)

    log_str = ("=> Model created: visual backbone {}".format(args.arch))
    args.out_file.write(log_str + "\n")
    args.out_file.flush()

    if not torch.cuda.is_available():
            print('Using CPU, this will be slow')
    else:
        assert args.gpu is not None
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)

    model.eval()

    num_classes = len(classnames)

    classes_seen = torch.zeros(num_classes, dtype=torch.long)
    classes_correct = torch.zeros(num_classes, dtype=torch.long)

    total_seen = 0
    total_correct = 0

    print_freq = 100

    start = time.time()

    for i, (images, target) in enumerate(val_loader):
        assert args.gpu is not None
        target = target.cuda(args.gpu, non_blocking=True)

        # ---------- image handling ----------
        if isinstance(images, list):
            for k in range(len(images)):
                images[k] = images[k].cuda(args.gpu, non_blocking=True)
            image = images[0]
        
        images = torch.cat(images, dim=0)

        # ---------- prediction ----------
        logits = model(image) # (1, C)
        pred = logits.argmax(dim=1)  # (1,)
        gt = target.view(-1) # (1,)

        # ---------- update stats ----------
        cls = gt.item()
        classes_seen[cls] += 1
        total_seen += 1

        if pred.item() == cls:
            classes_correct[cls] += 1
            total_correct += 1

        if ((i + 1) % 5 == 0) or ((i + 1) == len(val_loader)):
            print_log = 'Sample:{}/{}'.format(i + 1, len(val_loader))
            args.out_file.write(print_log + '\n')
            args.out_file.flush()

        # ---------- running accuracy ----------
        if (i + 1) % print_freq == 0:
            running_acc = 100.0 * total_correct / total_seen

            log_str = (
                f"Running Acc: {running_acc:.2f}%"
            )

            args.out_file.write("\n" + log_str + "\n" + "\n")
            args.out_file.flush()

    elapsed = time.time() - start

    log_str = f"\nTime: {time.strftime('%H:%M:%S', time.gmtime(elapsed))}"
    args.out_file.write(log_str + "\n\n")
    args.out_file.flush()
    
    del val_dataset, val_loader

    # ---------- class-wise accuracy ----------
    classwise_acc = torch.zeros(num_classes)
    for c in range(num_classes):
        if classes_seen[c] > 0:
            classwise_acc[c] = (
                100.0 * classes_correct[c] / classes_seen[c]
            )
        else:
            classwise_acc[c] = float("nan")

    args.out_file.write("\n=== Class-wise Accuracy ===\n")

    for c in range(num_classes):
        acc = classwise_acc[c].item()

        if not math.isnan(acc):
            log_str = f"Class {c:04d} ({classnames[c]}): {acc:.2f}%"
        else:
            log_str = f"Class {c:04d} ({classnames[c]}): N/A"

        args.out_file.write(log_str + "\n")


    # ---------- final accuracy ----------
    final_acc = 100.0 * total_correct / total_seen
    log_str = (f"Final Accuracy: {final_acc:.2f}%")

    args.out_file.write("\n" + log_str + "\n")
    args.out_file.flush()
    
    results = {
        "final_accuracy": final_acc,
        "classwise_accuracy": classwise_acc,
        "classes_seen": classes_seen,
        "classes_correct": classes_correct,
        "classnames": classnames,
    }

    torch.save(results, os.path.join(args.output_dir, "accuracy_results.pt"))



def main():
    args = parser.parse_args()

    name_suffix = ''

    if args.seed == -1:
        import random
        args.seed = random.randint(0, 2**32 - 1)
        
    set_random_seed(args.seed)

    args.output_dir = os.path.join(args.output_dir, args.algorithm, args.test_sets, 'seed_'+str(args.seed), args.arch)

    if os.path.exists(args.output_dir):
        items = os.listdir(args.output_dir)
        
        # Filter for directories that match the pattern "exp_<number>"
        exp_folders = []
        for item in items:
            item_path = os.path.join(args.output_dir, item)
            if os.path.isdir(item_path) and item.startswith('exp_'):
                try:
                    # Extract the number from folder name "exp_<number>"
                    exp_num = int(item.split('_')[1])
                    exp_folders.append(exp_num)
                except (IndexError, ValueError):
                    # Skip if folder name doesn't follow the pattern
                    continue
        
        # Find the next available experiment number
        if exp_folders:
            next_exp_num = max(exp_folders) + 1
        else:
            next_exp_num = 0
    else:
        # If the directory doesn't exist, create parent directories and start with exp_0
        os.makedirs(args.output_dir, exist_ok=True)
        next_exp_num = 0

    new_exp_folder = f"exp_{next_exp_num}"
    args.output_dir = os.path.join(args.output_dir, new_exp_folder)

    os.makedirs(args.output_dir, exist_ok=True)

    args.out_file = open(os.path.join(args.output_dir, 'output'+name_suffix+'.log'), 'w')

    args.out_file.write(print_args(args)+'\n')
    args.out_file.flush()



    assert args.gpu is not None

    set_random_seed(args.seed)

    log_str = ("Use GPU: {} for training".format(args.gpu))
    args.out_file.write(log_str + "\n")
    args.out_file.flush()

    cudnn.benchmark = True

    if args.algorithm == 'zs':
        zero_shot(args)
        return None

    normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                     std=[0.26862954, 0.26130258, 0.27577711])
    
    dset = args.test_sets



    if True:
        if args.algorithm in ['ima/all', 'ima/fil',
                              'tma/all', 'tma/fil']:
            # ##########  DataSet  ##########
            base_transform = transforms.Compose([
                transforms.Resize(args.resolution, interpolation=BICUBIC),
                transforms.CenterCrop(args.resolution)])

            preprocess = transforms.Compose([
                transforms.ToTensor(),
                normalize
                ])

            data_transform = AugMixAugmenter(base_transform, preprocess, n_views=args.batch_size-1, use_augmix=True, severity=1)

            batchsize = 1
            if dset.lower() in [d.lower() for d in fewshot_datasets]:
                val_dataset = build_fewshot_dataset(
                    set_id=dset, 
                    root=args.testdir,
                    transform=data_transform,
                    mode=args.dataset_mode
                )
            else:
                val_dataset = CustomImageFolder(    
                    root=args.testdir,
                    transform=data_transform
                )

            log_str = ("Number of test samples: {}".format(len(val_dataset)))
            args.out_file.write(log_str + "\n")
            args.out_file.flush()

            val_loader = DataLoader(
                val_dataset,
                batch_size=batchsize,
                shuffle=False,
                num_workers=args.workers,
                pin_memory=True
            )
        else:
            raise NotImplementedError



        print("evaluating: {}".format(args.test_sets))

        if len(args.test_sets) > 1: 
            # fine-grained classification datasets
            classnames = eval("{}_classes".format(args.test_sets.lower()))
        else:
            assert args.test_sets in ['A', 'R', 'K', 'V', 'I']
            classnames_all = imagenet_classes
            classnames = []
            if args.test_sets in ['A', 'R', 'V']:
                label_mask = eval("imagenet_{}_mask".format(args.test_sets.lower()))
                if args.test_sets == 'R':
                    for i, m in enumerate(label_mask):
                        if m:
                            classnames.append(classnames_all[i])
                else:
                    classnames = [classnames_all[i] for i in label_mask]
            else:
                classnames = classnames_all



        # ##########  Model  ##########
        if True:
            elif args.algorithm.startswith('ima'):
                model = getimagematrixadapter(args, clip_arch=args.arch, classnames=classnames, device=args.gpu, n_ctx=args.n_ctx, ctx_init=args.ctx_init)
            elif args.algorithm.startswith('tma'):
                model = gettextmatrixadapter(args, clip_arch=args.arch, classnames=classnames, device=args.gpu, n_ctx=args.n_ctx, ctx_init=args.ctx_init)

        log_str = ("=> Model created: visual backbone {}".format(args.arch))
        args.out_file.write(log_str + "\n")
        args.out_file.flush()



        if not torch.cuda.is_available():
            print('Using CPU, this will be slow')
        else:
            assert args.gpu is not None
            torch.cuda.set_device(args.gpu)
            model = model.cuda(args.gpu)

        

        if args.algorithm.startswith('ima'):
            tta_trainer = IMA(model, args.gpu)
        elif args.algorithm.startswith('tma'):
            tta_trainer = TMA(model, args.gpu)
        else:
            raise NotImplementedError



        tta_trainer.prepare_model_and_optimization(args)

        tta_trainer.model.eval()

        num_classes = len(classnames)

        classes_seen = torch.zeros(num_classes, dtype=torch.long)
        classes_correct = torch.zeros(num_classes, dtype=torch.long)

        total_seen = 0
        total_correct = 0

        print_freq = 100

        log_str = f"Start"
        args.out_file.write(log_str + "\n\n")

        for i, (images, target) in enumerate(val_loader):
            assert args.gpu is not None
            target = target.cuda(args.gpu, non_blocking=True)

            # ---------- image handling ----------
            if isinstance(images, list):
                for k in range(len(images)):
                    images[k] = images[k].cuda(args.gpu, non_blocking=True)
                image = images[0]
            
            images = torch.cat(images, dim=0)

            # ---------- episodic TTA ----------
            tta_trainer.pre_adaptation()

            return_dict = tta_trainer.adaptation_process(image, images, args)

            # ---------- prediction ----------
            logits = return_dict["output"] # (1, C)
            pred = logits.argmax(dim=1)  # (1,)
            gt = target.view(-1) # (1,)

            # ---------- update stats ----------
            cls = gt.item()
            classes_seen[cls] += 1
            total_seen += 1

            if pred.item() == cls:
                classes_correct[cls] += 1
                total_correct += 1

            if ((i + 1) % 5 == 0) or ((i + 1) == len(val_loader)):
                print_log = 'Sample:{}/{}'.format(i + 1, len(val_loader))
                args.out_file.write(print_log + '\n')
                args.out_file.flush()
        
        del val_dataset, val_loader

        # ---------- class-wise accuracy ----------
        classwise_acc = torch.zeros(num_classes)
        for c in range(num_classes):
            if classes_seen[c] > 0:
                classwise_acc[c] = (
                    100.0 * classes_correct[c] / classes_seen[c]
                )
            else:
                classwise_acc[c] = float("nan")

        args.out_file.write("\n=== Class-wise Accuracy ===\n")

        for c in range(num_classes):
            acc = classwise_acc[c].item()

            if not math.isnan(acc):
                log_str = f"Class {c:04d} ({classnames[c]}): {acc:.2f}%"
            else:
                log_str = f"Class {c:04d} ({classnames[c]}): N/A"

            args.out_file.write(log_str + "\n")


        # ---------- final accuracy ----------
        final_acc = 100.0 * total_correct / total_seen
        log_str = (f"Final Accuracy: {final_acc:.2f}%")
    
        args.out_file.write("\n" + log_str + "\n")
        args.out_file.flush()
        
        results = {
            "final_accuracy": final_acc,
            "classwise_accuracy": classwise_acc,
            "classes_seen": classes_seen,
            "classes_correct": classes_correct,
            "classnames": classnames,
        }

        torch.save(results, os.path.join(args.output_dir, "accuracy_results.pt"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Episodic TTA')

    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')

    parser.add_argument('--seed', type=int, default=-1)
    parser.add_argument('--gpu', default=0, type=int, help='GPU id to use.')

    parser.add_argument('--output_dir', type=str, default='output_results/temp')
    parser.add_argument("--testdir", type=str, default="./data/imagenet/A")
    parser.add_argument('--dataset_mode', type=str, default='test', help='which split to use: train/val/test')

    parser.add_argument('--resolution', default=224, type=int, help='CLIP image resolution')
    parser.add_argument('-b', '--batch-size', default=64, type=int, metavar='N')

    parser.add_argument('-a', '--arch', metavar='ARCH', default='RN50')
    parser.add_argument('--test_sets', type=str, default='A/R/V/K/I', help='test dataset (multiple datasets split by slash)')

    parser.add_argument('--algorithm', type=str, default='zs', choices=['ima/all', 'ima/fil',
                                                                         'tma/all', 'tma/fil',
                                                                         'zs',
                                                                        ])

    parser.add_argument('--transform_mode', default=None, type=str, help='Same or Different matrices')

    parser.add_argument('--lr', '--learning-rate', default=5e-3, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')

    parser.add_argument('--selection_p', default=0.1, type=float, help='confidence selection percentile')
    parser.add_argument('--tta_steps', default=1, type=int, help='test-time-adapt steps')

    parser.add_argument('--n_ctx', default=4, type=int, help='number of tunable tokens')
    parser.add_argument('--ctx_init', default=None, type=str, help='init tunable prompts')

    parser.add_argument('--ensemble', default=False, type=bool, help='multiple prompt templates')

    parser.add_argument("--coop_init", action="store_true", help="Use CoOp prompt initialization")
    parser.add_argument("--coop_ckpt", type=str, default=None, help="Path to CoOp checkpoint (model.pth.tar-50)")

    main()