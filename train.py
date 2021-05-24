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
from dataset import BaseDataset, BaseDataset_longclip
from models import ModelBuilder, SegmentationModule
from utils import AverageMeter, parse_devices, setup_logger
import numpy as np
import torch.multiprocessing as mp
import torch.distributed as dist
import datetime as datetime

# train one epoch
def train(segmentation_module, data_loader, optimizers, history, epoch, cfg,args,gpu,scaler=None,autocast=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    ave_total_loss = AverageMeter()
    ave_acc = AverageMeter()

    segmentation_module.train(not cfg.TRAIN.fix_bn)

    epoch_iters = len(data_loader)
    max_iters = epoch_iters * cfg.TRAIN.num_epoch

    data_loader.sampler.set_epoch(epoch)

    # main loop
    tic = time.time()
    it_=0
    for i,data in enumerate(data_loader):
        it_+=1
        # load a batch of data
        if args.use_clipdataset:
            clip_imgs, clip_gts = data
            input_imgs = torch.cat(clip_imgs,dim=0)
            gt_label =  torch.cat(clip_gts,dim=0)
            
            input_imgs = input_imgs.cuda(gpu)
            gt_label = gt_label.cuda(gpu)
            batch_data ={}
            batch_data['img_data']= input_imgs
            batch_data['seg_label'] = gt_label

        else:
            imgs, gts = data
            imgs = imgs.cuda(gpu)
            gts = gts.cuda(gpu)
            batch_data ={}
            batch_data['img_data']= imgs
            batch_data['seg_label'] = gts
        batch_data['step']=it_
        
        data_time.update(time.time() - tic)
        optimizers.zero_grad()

        # adjust learning rate
        cur_iter = i + (epoch - 1) * epoch_iters
#        print(cur_iter)
#        print(max_iters)
        adjust_learning_rate(optimizers, cur_iter, cfg, max_iters)

        # forward pass
        if args.use_float16:
            with autocast():
                loss, acc = segmentation_module(batch_data)
                loss = loss.mean()
                acc = acc.mean()
    
            # Backward
            scaler.scale(loss).backward()
           # loss.backward()
            scaler.step(optimizers)
#                optimizer.step()
            scaler.update()
       
        else:
            loss, acc = segmentation_module(batch_data)
            #print(loss)
            #exit()
            loss = loss.mean()
            acc = acc.mean()
    
            # Backward
            loss.backward()
            optimizers.step()
#        loss_to_show = dist.all_reduce(loss)
#        acc_to_show = dist.all_reduce(acc)

        # measure elapsed time
        batch_time.update(time.time() - tic)
        tic = time.time()

        # update average loss and acc
        ave_total_loss.update(loss.item())
        ave_acc.update(acc.item()*100)

        # calculate accuracy, and display
        if dist.get_rank()==0:
            print('Epoch: [{}][{}/{}], Time: {:.2f}, Data: {:.2f}, '
                  'lr_encoder: {:.6f}, '
                  'Accuracy: {:4.2f}, Loss: {:.6f}'
                  .format(epoch, i, epoch_iters,
                          batch_time.average(), data_time.average(),
                          cfg.TRAIN.running_lr_encoder, 
                          ave_acc.average(), ave_total_loss.average()))

        fractional_epoch = epoch - 1 + 1. * i / epoch_iters
        history['train']['epoch'].append(fractional_epoch)
        history['train']['loss'].append(loss.data.item())
        history['train']['acc'].append(acc.data.item())

def test(segmentation_module, loader,gpu,args=None):

    if args.lesslabel:
        label_num_ = 42
    else:
        label_num_ =args.num_class 
    segmentation_module.eval()
    evaluator = Evaluator(label_num_)

    print('validation')
    
    for i,data in enumerate(loader):
        # process data
        print('[{}]/[{}]'.format(i,len(loader)))
        imgs, gts = data
        imgs = imgs.cuda(gpu)
        gts = gts.cuda(gpu)
        batch_data ={}
        batch_data['img_data']= imgs
        batch_data['seg_label'] = gts
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
    #if self.args.tensorboard:
    #    self.writer.add_scalar('val/total_loss_epoch', test_loss, epoch)
    #    self.writer.add_scalar('val/mIoU', mIoU, epoch)
    #    self.writer.add_scalar('val/Acc', Acc, epoch)
    #    self.writer.add_scalar('val/Acc_class', Acc_class, epoch)
    #    self.writer.add_scalar('val/fwIoU', FWIoU, epoch)
    print('Validation:')
    #print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
    print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
#    print('Loss: %.3f' % test_loss)






def checkpoint(nets,optimizers,history, args, epoch):
    print('Saving checkpoints...')
    net_encoder = nets

    dict_encoder = net_encoder.state_dict()

    if not os.path.exists(args.saveroot):
        os.makedirs(args.saveroot)
#    torch.save(
#        history,
#        '{}/history_epoch_{}.pth'.format(args.saveroot, epoch))
    torch.save(
        dict_encoder,
        '{}/model_epoch_{}.pth'.format(args.saveroot, epoch))

    optimizer_encoder=optimizers
    torch.save(optimizer_encoder.state_dict(),'{}/opt_epoch_{}.pth'.format(args.saveroot, epoch))
#    torch.save(optimizer_decoder.state_dict(),'{}/opt_decoder_epoch_{}.pth'.format(args.saveroot, epoch))


def group_weight(module):
    group_decay = []
    group_no_decay = []
    for m in module.modules():
        if isinstance(m, nn.Linear):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.modules.conv._ConvNd):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.modules.batchnorm._BatchNorm):
            if m.weight is not None:
                group_no_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)

    assert len(list(module.parameters())) == len(group_decay) + len(group_no_decay)
    groups = [dict(params=group_decay), dict(params=group_no_decay, weight_decay=.0)]
    return groups


def create_optimizers(nets, cfg):
#    (net_encoder, net_decoder, crit) = nets
    optimizer_encoder = torch.optim.SGD(
        group_weight(nets),
        lr=cfg.TRAIN.lr_encoder,
        momentum=cfg.TRAIN.beta1,
        weight_decay=cfg.TRAIN.weight_decay)
#    optimizer_decoder = torch.optim.SGD(
#        group_weight(net_decoder),
#        lr=cfg.TRAIN.lr_decoder,
#        momentum=cfg.TRAIN.beta1,
#        weight_decay=cfg.TRAIN.weight_decay)
    return optimizer_encoder


def adjust_learning_rate(optimizers, cur_iter, cfg,max_iters):
    scale_running_lr = ((1. - float(cur_iter) / max_iters) ** cfg.TRAIN.lr_pow)
    cfg.TRAIN.running_lr_encoder = cfg.TRAIN.lr_encoder * scale_running_lr

    optimizer_encoder = optimizers
    for param_group in optimizer_encoder.param_groups:
        param_group['lr'] = cfg.TRAIN.running_lr_encoder


def main( gpu,cfg,args):
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
            # self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model).cuda(self.gpu)


    if args.use_float16:
        from torch.cuda.amp import autocast as autocast, GradScaler
        scaler = GradScaler()
    else:
        scaler = None
        autocast = None

    label_num_=args.num_class
    net_encoder = ModelBuilder.build_encoder(
        arch=cfg.MODEL.arch_encoder.lower(),
        fc_dim=cfg.MODEL.fc_dim,
        weights=cfg.MODEL.weights_encoder)
    net_decoder = ModelBuilder.build_decoder(
        arch=cfg.MODEL.arch_decoder.lower(),
        fc_dim=cfg.MODEL.fc_dim,
        num_class=label_num_,
        weights=cfg.MODEL.weights_decoder)

    crit = nn.NLLLoss(ignore_index=255)

    if cfg.MODEL.arch_decoder.endswith('deepsup'):
        segmentation_module = SegmentationModule(
            net_encoder, net_decoder, crit, cfg.TRAIN.deep_sup_scale)
    else:
        segmentation_module = SegmentationModule(
            net_encoder, net_decoder, crit)

    if args.use_clipdataset:
        dataset_train = BaseDataset_longclip(args,'train')
    else:
        dataset_train = BaseDataset(
            args,
            'train'
            )

    sampler_train =torch.utils.data.distributed.DistributedSampler(dataset_train)
    loader_train = torch.utils.data.DataLoader(dataset_train,  batch_size=args.batchsize,  shuffle=False,sampler=sampler_train,   pin_memory=True,
                                    num_workers=args.workers)


    print('1 Epoch = {} iters'.format(cfg.TRAIN.epoch_iters))

    dataset_val = BaseDataset(
        args,
        'val'
        )
    sampler_val =torch.utils.data.distributed.DistributedSampler(dataset_val)
    loader_val = torch.utils.data.DataLoader(dataset_val,  batch_size=args.batchsize,  shuffle=False,sampler=sampler_val,   pin_memory=True,
                                    num_workers=args.workers)
#    loader_val = torch.utils.data.DataLoader(dataset_val,batch_size=args.batchsize,shuffle=False,num_workers=args.workers)
    # create loader iterator
    

    # load nets into gpu

    segmentation_module = segmentation_module.cuda(load_gpu)

    segmentation_module= nn.SyncBatchNorm.convert_sync_batchnorm(segmentation_module)

    if args.resume_epoch!=0:
#        if dist.get_rank() == 0:
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

    # Set up optimizers
#    nets = (net_encoder, net_decoder, crit)
    nets = segmentation_module
    optimizers = create_optimizers(segmentation_module, cfg)
    if args.resume_epoch!=0:
#        if dist.get_rank() == 0:
        optimizers.load_state_dict(torch.load(os.path.join('./resume','opt_epoch_{}.pth'.format(args.resume_epoch)),map_location=torch.device("cuda:"+str(load_gpu))))
        print('resume from epoch {}'.format(args.resume_epoch))

    # Main loop
    history = {'train': {'epoch': [], 'loss': [], 'acc': []}}

#    test(segmentation_module,loader_val,args)
    for epoch in range(cfg.TRAIN.start_epoch, cfg.TRAIN.num_epoch):
        if dist.get_rank() == 0 and epoch==0:
            checkpoint(nets,optimizers, history, args, epoch+1)
        print('Epoch {}'.format(epoch))
        train(segmentation_module, loader_train, optimizers, history, epoch+1, cfg,args,load_gpu,scaler=scaler,autocast=autocast)

###################        # checkpointing
        if dist.get_rank() == 0 and (epoch+1)%10==0:
            checkpoint(segmentation_module,optimizers, history, args, epoch+1)
        if args.validation:
            test(segmentation_module,loader_val,args)

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
    parser.add_argument("--multi_scale",type=str2bool,default=True)
    parser.add_argument("--saveroot",type=str,default='')
    parser.add_argument("--totalepoch",type=int,default=30)
    parser.add_argument("--dataroot2",type=str,default='')
    parser.add_argument("--usetwodata",type=str2bool,default=False)
    parser.add_argument("--cropsize",type=int,default=531)
    parser.add_argument("--validation",type=str2bool,default=True)
    parser.add_argument("--lesslabel",type=str2bool,default=False)
    parser.add_argument("--train_filter",type=str2bool,default=False)
    parser.add_argument("--weight_decay",type=float,default=1e-4)

    ####
    parser.add_argument("--use_clipdataset",type=str2bool,default=False)
    parser.add_argument("--dilation2",type=str,default="3,6,9")
    parser.add_argument("--clip_num",type=int,default=4)
    parser.add_argument("--dilation_num",type=int,default=0)
    parser.add_argument("--resume_epoch",type=int,default=0)
    ###

    parser.add_argument("--use_float16",type=str2bool,default=False)
    parser.add_argument("--port",type=int,default=45321)
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()

    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)
    # cfg.freeze()

#    logger = setup_logger(distributed_rank=0)   # TODO
#    logger.info("Loaded configuration file {}".format(args.cfg))
#    logger.info("Running with config:\n{}".format(cfg))
#    logger.info("Running with config:\n{}".format(args))

    # Output directory
    if not os.path.isdir(cfg.DIR):
        os.makedirs(cfg.DIR)
    #logger.info("Outputing checkpoints to: {}".format(cfg.DIR))
    #with open(os.path.join(cfg.DIR, 'config.yaml'), 'w') as f:
    #    f.write("{}".format(cfg))

    cfg.MODEL.weights_encoder = args.predir
    cfg.MODEL.weights_decoder = ''
#    cfg.MODEL.weights_encoder = ''
#    cfg.MODEL.weights_decoder = ''
    # Start from checkpoint
#    cfg.MODEL.weights_encoder = os.path.join(
#        args.predir, 'encoder_epoch_{}.pth'.format(cfg.TRAIN.num_epoch))
#    cfg.MODEL.weights_decoder = os.path.join(
#        args.predir, 'decoder_epoch_{}.pth'.format(cfg.TRAIN.num_epoch))
#    assert os.path.exists(cfg.MODEL.weights_encoder) and \
#            os.path.exists(cfg.MODEL.weights_decoder), "checkpoint does not exitst!"

    # Parse gpu ids
    gpus = parse_devices(args.gpus)
    gpus = [x.replace('gpu', '') for x in gpus]
    gpus = [int(x) for x in gpus]
    num_gpus = len(gpus)
#    cfg.TRAIN.batch_size = num_gpus * cfg.TRAIN.batch_size_per_gpu
    cfg.TRAIN.num_epoch = args.totalepoch

    cfg.TRAIN.max_iters = cfg.TRAIN.epoch_iters * cfg.TRAIN.num_epoch

    cfg.TRAIN.weight_decay=args.weight_decay

    cfg.TRAIN.lr_encoder = args.lr
    cfg.TRAIN.running_lr_encoder = cfg.TRAIN.lr_encoder

    print(args)

    mp.spawn(main, nprocs=args.gpu_num, args=(cfg,args,))
