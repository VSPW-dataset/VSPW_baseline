# System libs
import os
import time
import json
# import math
import random
import argparse
from utils import Evaluator
from distutils.version import LooseVersion
# Numerical libs
import torch
import torch.nn as nn
# Our libs
from config import cfg
from dataset import BaseDataset, BaseDataset_clip,TestDataset_clip,BaseDataset_longclip
from models import ModelBuilder, Clip_PSP,ClipOCRNet
from utils import AverageMeter, parse_devices, setup_logger
import numpy as np
import torch.multiprocessing as mp
import torch.distributed as dist
import datetime as datetime
from torch.cuda.amp import autocast, GradScaler


# train one epoch
def train(segmentation_module, data_loader, optimizers, history, epoch, cfg,args,gpu):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    ave_total_loss = AverageMeter()
    ave_acc = AverageMeter()

    segmentation_module.train()

    epoch_iters = len(data_loader)
    max_iters = epoch_iters * cfg.TRAIN.num_epoch
    data_loader.sampler.set_epoch(epoch)



    # main loop
    tic = time.time()
    it_=0
    for i,data in enumerate(data_loader):
        it_+=1
        #continue
        # load a batch of data
        
        clip_imgs, clip_gts = data
        clip_imgs = [imgs.cuda(gpu) for imgs in clip_imgs]
        clip_gts = [gts.cuda(gpu) for gts in clip_gts]
        batch_data ={}
        if args.clip_num%2==0:
            idx = args.clip_num/2
        else:
            idx = (args.clip_num-1)/2
        if args.method=='clip_psp' or args.method=='clip_ocr':
            batch_data['img_data']= clip_imgs[0]
            batch_data['seg_label'] = clip_gts[0]
            batch_data['clipimgs_data'] = clip_imgs[1:]
            batch_data['cliplabels_data'] = clip_gts[1:]
    
        else:
            raise(NotImplementedError)
        batch_data['step']=it_
        
        data_time.update(time.time() - tic)

        optimizers.zero_grad()

        # adjust learning rate
        cur_iter = i + (epoch - 1) * epoch_iters
        batch_data['cur_step']=cur_iter

        adjust_learning_rate(optimizers, cur_iter, cfg,max_iters,args)

        # forward pass


        loss, acc = segmentation_module(batch_data)
        loss = loss.mean()
        acc = acc.mean()
    
        # Backward
        loss.backward()
    #    for optimizer in optimizers:
        optimizers.step()

        # measure elapsed time
        batch_time.update(time.time() - tic)
        tic = time.time()

        # update average loss and acc
        ave_total_loss.update(loss.data.item())
        ave_acc.update(acc.data.item()*100)

        # calculate accuracy, and display
        if dist.get_rank()==0:
            print('Epoch: [{}][{}/{}], Time: {:.2f}, Data: {:.2f}, '
                  'lr_encoder: {:.6f}, '
                  'Accuracy: {:4.2f}, Loss: {:.6f}'
                  .format(epoch, i, epoch_iters,
                          batch_time.average(), data_time.average(),
                          cfg.TRAIN.running_lr_encoder, 
                          ave_acc.average(), ave_total_loss.average()))
#

def test(segmentation_module, args=None):

    label_num_ = args.num_class
    segmentation_module.eval()
    evaluator = Evaluator(label_num_)

    print('validation')
    
    with open(os.path.join(args.dataroot,'val.txt'),'r') as f:
        lines=f.readlines()
        videolists = [line[:-1] for line in lines]

    for video in videolists:
        test_dataset = TestDataset_clip(args.dataroot,video,args,is_train=True)
        loader = torch.utils.data.DataLoader(test_dataset,batch_size=args.batchsize,shuffle=False,num_workers=args.workers,drop_last=False)

        for i,data in enumerate(loader):
            # process data
            print('[{}]/[{}]'.format(i,len(loader)))
            imgs, gts,clip_imgs,_,_ = data
            imgs = imgs.cuda(gpu)
            gts = gts.cuda(gpu)
            clip_imgs = [img.cuda(gpu) for img in clip_imgs]
            batch_data ={}
            batch_data['img_data']= imgs
            batch_data['seg_label'] = gts
            batch_data['clipimgs_data']=clip_imgs
            segSize = (imgs.size(2),
                       imgs.size(3))
    
            with torch.no_grad():
                 
                scores = segmentation_module(batch_data, segSize=segSize)
                pred = torch.argmax(scores, dim=1)
                pred = pred.data.cpu().numpy()
                target = gts.squeeze(1).cpu().numpy()
                 
                # Add batch sample into evaluator
                evaluator.add_batch(target, pred)
    Acc = evaluator.Pixel_Accuracy()
    Acc_class = evaluator.Pixel_Accuracy_Class()
    mIoU =evaluator.Mean_Intersection_over_Union()
    FWIoU = evaluator.Frequency_Weighted_Intersection_over_Union()
    print('Validation:')
    print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))






def checkpoint(opt,nets, history, args, epoch):
    print('Saving checkpoints...')

    if not os.path.exists(args.saveroot):
        os.makedirs(args.saveroot)
    torch.save(
        nets.state_dict(),
        '{}/model_epoch_{}.pth'.format(args.saveroot, epoch))
    torch.save(
        opt.state_dict(),
        '{}/opt_epoch_{}.pth'.format(args.saveroot, epoch))
def get_trainable_params(model, base_lr, weight_decay, beta_wd=True):
    params = []
    memo = set()
    for key, value in model.named_parameters():
        if not value.requires_grad or value in memo:
            continue
        memo.add(value)
        wd = weight_decay
        if 'beta' in key and not beta_wd:
            wd = 0.
        if 'bias' in key:
            wd = 0.
        params += [{"params": [value], "lr": base_lr, "weight_decay": wd, "name": key}]
    return params


def create_optimizers(model, cfg,args):

    train_params = get_trainable_params(model,args.lr,cfg.TRAIN.weight_decay)
    
    optimizer = torch.optim.SGD(
        train_params,
        lr=args.lr,
        momentum=cfg.TRAIN.beta1,
        weight_decay=cfg.TRAIN.weight_decay)
    return optimizer


def adjust_learning_rate(optimizer, cur_iter, cfg,max_iters,args):
    scale_running_lr = ((1. - float(cur_iter) / max_iters) ** cfg.TRAIN.lr_pow)
    cfg.TRAIN.running_lr_encoder = args.lr * scale_running_lr

    now_lr = cfg.TRAIN.running_lr_encoder


    for param_group in optimizer.param_groups:
        if 'encoder' in param_group['name']:
            param_group['lr'] = now_lr * 0.1
        else:
            param_group['lr'] = now_lr







def main(gpu,cfg,args):
    # Network Builders

    load_gpu = gpu+args.start_gpu
    rank = gpu
    torch.cuda.set_device(load_gpu)
    dist.init_process_group(
        backend='nccl',
        init_method='tcp://127.0.0.1:{}'.format(args.port),
        world_size=args.gpu_num,
        rank=rank,
        timeout=datetime.timedelta(seconds=300))

    label_num_=args.num_class

    net_encoder = ModelBuilder.build_encoder(
        arch=cfg.MODEL.arch_encoder.lower(),
        fc_dim=cfg.MODEL.fc_dim,
        weights=cfg.MODEL.weights_encoder,args=args)



    crit = nn.NLLLoss(ignore_index=255)

    
    if args.method =='clip_psp':
        segmentation_module=Clip_PSP(net_encoder,crit,args,deep_sup_scale=0.4)
    elif args.method =='clip_ocr':
        segmentation_module=ClipOCRNet(net_encoder,crit,args,deep_sup_scale=0.4)

    else:
        raise(NotImplementedError)

    # Dataset and Loader
    if args.method=='clip_psp' or args.method=='clip_ocr':
        dataset_train = BaseDataset_longclip(args,'train')
     
    else:
        dataset_train = BaseDataset_clip(
             args,
            'train'
            )

    sampler_train =torch.utils.data.distributed.DistributedSampler(dataset_train)
    loader_train = torch.utils.data.DataLoader(dataset_train,  batch_size=args.batchsize,  shuffle=False,sampler=sampler_train,   pin_memory=True,
                                    num_workers=args.workers)


    

    # load nets into gpu
    
    segmentation_module=nn.SyncBatchNorm.convert_sync_batchnorm(segmentation_module).cuda(load_gpu)
    if args.resume_epoch!=0:
        to_load = torch.load(os.path.join('./resume','model_epoch_{}.pth'.format(args.resume_epoch)),map_location=torch.device("cuda:"+str(load_gpu)))
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in to_load.items():
            name = k[7:] # remove `module.`，表面从第7个key值字符取到最后一个字符，正好去掉了module.
            new_state_dict[name] = v #新字典的key值对应的value为一一对应的值。
        cfg.TRAIN.start_epoch=args.resume_epoch
        segmentation_module.load_state_dict(new_state_dict)

    segmentation_module= torch.nn.parallel.DistributedDataParallel(
                    segmentation_module,
                device_ids=[load_gpu],
                find_unused_parameters=True)

    optimizer = create_optimizers(segmentation_module, cfg,args)
    if args.resume_epoch!=0:
        optimizer.load_state_dict(torch.load(os.path.join('./resume','opt_epoch_{}.pth'.format(args.resume_epoch)),map_location=torch.device("cuda:"+str(load_gpu))))
        print('resume from epoch {}'.format(args.resume_epoch))

    # Set up optimizers

    # Main loop
    history = {'train': {'epoch': [], 'loss': [], 'acc': []}}

    

    for epoch in range(cfg.TRAIN.start_epoch, cfg.TRAIN.num_epoch):
        print('Epoch {}'.format(epoch))
        if dist.get_rank() == 0 and epoch==0:
            checkpoint(optimizer,segmentation_module, history, args, epoch+1)
        train(segmentation_module, loader_train, optimizer, history, epoch+1, cfg,args,load_gpu)

###################        # checkpointing
        if (epoch+1)%20==0:
            if dist.get_rank() == 0:
                checkpoint(optimizer,segmentation_module, history, args, epoch+1)
            if args.validation:
                test(segmentation_module,args)
#
    print('Training Done!')


if __name__ == '__main__':
    def str2bool(v):
        if isinstance(v, bool):
           return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
    assert LooseVersion(torch.__version__) >= LooseVersion('0.4.0'), \
        'PyTorch>=0.4.0 is required'

    parser = argparse.ArgumentParser(
        description="PyTorch Semantic Segmentation Training"
    )
    parser.add_argument(
        "--cfg",
        default="config/ade20k-resnet50dilated-ppm_deepsup.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--gpus",
        default="0-3",
        help="gpus to use, e.g. 0-3 or 0,1,2,3"
    )
    parser.add_argument(
    "--predir",
    default= '../../ade20k-hrnetv2-c1'
    )
    parser.add_argument("--num_class",type=int,default=124)
    parser.add_argument("--batchsize",type=int,default=16)
    parser.add_argument("--workers",type=int,default=0)
    parser.add_argument("--start_gpu",type=int,default=0)
    parser.add_argument("--gpu_num",type=int,default=1)
    parser.add_argument("--dataroot",type=str,default='')
    parser.add_argument("--trainfps",type=int,default=1)
    parser.add_argument("--lr",type=float,default=0.02)
    parser.add_argument("--multi_scale",type=str2bool,default=False)
    parser.add_argument("--saveroot",type=str,default='')
    parser.add_argument("--totalepoch",type=int,default=30)
    parser.add_argument("--dataroot2",type=str,default='')
    parser.add_argument("--usetwodata",type=str2bool,default=False)
    parser.add_argument("--cropsize",type=int,default=531)
    parser.add_argument("--validation",type=str2bool,default=True)
    parser.add_argument("--lesslabel",type=str2bool,default=False)
    parser.add_argument("--clip_num",type=int,default=5)
    parser.add_argument("--dilation_num",type=int,default=3)
    
    parser.add_argument("--clip_up",type=str2bool,default=False)
    parser.add_argument("--clip_middle",type=str2bool,default=False)

    parser.add_argument("--fix",type=str2bool,default=False)
    parser.add_argument("--othergt",type=str2bool,default=False)
    parser.add_argument("--propclip2",type=str2bool,default=False)
    parser.add_argument("--early_usecat",type=str2bool,default=False)
    parser.add_argument("--earlyfuse",type=str2bool,default=False)

    parser.add_argument("--weight_decay",type=float,default=1e-4)
    #####

    ####
    parser.add_argument("--allsup",type=str2bool,default=False)
    parser.add_argument("--allsup_scale",type=float,default=0.3)
    parser.add_argument("--deepsup_scale",type=float,default=0.4)
    parser.add_argument("--linear_combine",type=str2bool,default=False)
    parser.add_argument("--distsoftmax",type=str2bool,default=False)
    parser.add_argument("--distnearest",type=str2bool,default=False)
    parser.add_argument("--temp",type=float,default=3)
    parser.add_argument("--max_distances",type=str,default='10')

    
    parser.add_argument("--pre_enc",type=str,default='')
    parser.add_argument("--pre_dec",type=str,default='')

    ####
    parser.add_argument("--method",type=str,default='',choices=['clip_psp','clip_ocr'])
    
    parser.add_argument("--dilation2",type=str,default="3,6,9")
    parser.add_argument("--resume_epoch",type=int,default=0)


    parser.add_argument("--clipocr_all",type=str2bool,default=False)
    parser.add_argument("--use_memory",type=str2bool,default=False)
    parser.add_argument("--memory_num",type=int,default=8)
    parser.add_argument("--st_weight",type=float,default=0.1)
    parser.add_argument("--psp_weight",type=str2bool,default=False)
    parser.add_argument("--port",type=int,default=45321)
    parser.add_argument("--split",type=str,default="train")


    #####
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()

    args.max_distances = args.max_distances.split(',')
    args.max_distances = [int(dd) for dd in args.max_distances]
 


    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)
    # cfg.freeze()


    # Output directory
    if not os.path.isdir(cfg.DIR):
        os.makedirs(cfg.DIR)
    with open(os.path.join(cfg.DIR, 'config.yaml'), 'w') as f:
        f.write("{}".format(cfg))

    cfg.MODEL.weights_encoder = args.pre_enc
    cfg.MODEL.weights_decoder = args.pre_dec

    # Parse gpu ids
    cfg.TRAIN.num_epoch = args.totalepoch

    cfg.TRAIN.max_iters = cfg.TRAIN.epoch_iters * cfg.TRAIN.num_epoch

    cfg.TRAIN.weight_decay=args.weight_decay

    cfg.TRAIN.lr_encoder = args.lr
    cfg.TRAIN.lr_decoder = args.lr
    cfg.TRAIN.running_lr_encoder = cfg.TRAIN.lr_encoder

    print(args)



    mp.spawn(main, nprocs=args.gpu_num, args=(cfg,args,))
