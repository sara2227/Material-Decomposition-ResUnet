import warnings
from torch import nn
import sys
# from core.modules import VGG_Feature_Extractor_16
warnings.simplefilter("ignore", (UserWarning, FutureWarning))
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import metrics
from dataloader import ImageDataset
from core.res_unet_plus import ResUnetPlusPlus
import logging
import torch
import argparse
import os
from torchvision.utils import save_image
import logging
import torch.nn.parallel
import torch.distributed as dist
import torch.multiprocessing as mp
import matplotlib.pyplot as plt
from torch.autograd import Variable
from core.movedim import MoveDimTransform
# from torch.utils.tensorboard import SummaryWriter


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def main(gpu,num_gpus,args):

    torch.cuda.set_device(gpu)

    torch.distributed.init_process_group(backend='nccl', rank=gpu,
                    world_size=num_gpus, init_method='env://')
    
    logging.basicConfig(filename='./logs/ResUnet_training.log', level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    # log_dir = "./logs"
    # writer = SummaryWriter(log_dir)
    checkpoint_dir = args.checkpoints
    os.makedirs(checkpoint_dir, exist_ok=True)

    model = ResUnetPlusPlus(6).cuda(gpu)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])

    # set up binary cross entropy and dice loss
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion_mse = torch.nn.MSELoss().cuda(gpu)
    criterion_l1 = torch.nn.L1Loss().cuda(gpu)
    # criterion = metrics.BCEDiceLoss()
    # decay LR
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    # starting params
    best_loss = 999
    start_epoch = 0
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            file = args.resume
            checkpoint = torch.load(file)
            start_epoch = int(checkpoint["epoch"]) + 1
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}'".format(args.resume))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))


    # Define data set
    dataset = ImageDataset('/proj/kth_deep_pcct/users/x_sarsa/data/')
    # print(len(dataset),'**********************')
    train_size = int(0.9 * len(dataset))
    valid_size = len(dataset) - train_size
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size])
    valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset)
    train_sampler =  torch.utils.data.distributed.DistributedSampler(train_dataset)

    # Create DataLoaders for training and validation
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False,num_workers=4*num_gpus,pin_memory=True,sampler=train_sampler)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False,num_workers=4*num_gpus,pin_memory=True,sampler=valid_sampler)
    if gpu==0:
        print('Size train data:',len(train_dataset),'\nSize valid data:',len(valid_dataset))
    print('Size train loader:',len(train_loader),'\nSize valid loader:',len(valid_loader))

    for epoch in range(start_epoch, args.epochs):
        # step the learning rate scheduler
        lr_scheduler.step()
        
        train_acc = metrics.MetricTracker()
        train_loss = metrics.MetricTracker()

        for idx, (input,output) in enumerate(train_loader):

            inputs = input.float().cuda(gpu,non_blocking=True)
            labels = output.float().cuda(gpu,non_blocking=True)

            optimizer.zero_grad()

            outputs = model(inputs)

            loss_mse = criterion_mse(outputs,labels)
            loss_l1 = criterion_l1(outputs,labels)
            loss = args.lambda_mse*loss_mse + args.lambda_l1*loss_l1
            
            # backward
            loss.backward()
            optimizer.step()

            # train_acc.update(metrics.dice_coeff(outputs, labels), outputs.size(0))
            train_loss.update(loss.data.item(), outputs.size(0))
            
            # Print log
            # print(gpu,'------ gpu -----')
            # if gpu==0:
            print(
                "\r[Epoch %d/%d] [Batch %d/%d] [MSE loss: %f] [L1 loss: %f] [Train loss: %f]\n"
                % (
                    epoch,
                    args.epochs,
                    idx,
                    len(train_loader),
                    loss_mse.item(),
                    loss_l1.item(),
                    train_loss.avg,
                    
                )
                )
            # train_acc_sum.append(train_acc.avg)
            if idx % args.logging_step == 0:
                logging.info(f"Epoch {epoch}: Training loss = {train_loss.avg} Dice accuracy = {train_acc.avg}")
            # Validatiuon
            if idx!= 0 and idx % args.validation_interval == 0:
                save_path = os.path.join(checkpoint_dir, "evaluate_epoch%d_step%d.pt" % (epoch,idx))
                save_path2 = os.path.join(checkpoint_dir, "resume_epoch%d_step%d.pt" % (epoch,idx))

                torch.save(
                    {
                        "step": idx,
                        "epoch": epoch,
                        "arch": "ResUnet",
                        "state_dict": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                    },
                    save_path2,
                )
                torch.save(model.module.state_dict(),save_path) #change
                print("Saved checkpoint to: %s" % save_path)
    
    torch.distributed.destroy_process_group()


if __name__ == "__main__":

    ngpus_per_node = torch.cuda.device_count()
    print(ngpus_per_node)

    parser = argparse.ArgumentParser(description="Road and Building Extraction")
    
    parser.add_argument(
        "--epochs",
        default=150,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    parser.add_argument(
        "--batch_size",
        default=2,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    parser.add_argument(
        '--lambda_mse',
        type = float,
        default = 1,
        help = 'Weight given to first loss objective',
        )
    parser.add_argument(
        '--lambda_l1',
        type = float,
        default = 1,
        help = 'Weigh given to second loss objective',
        )
    parser.add_argument(
        '--layer',
        type = int,
        default = 9,
        help = 'layer used in vgg as feature extractor (Kim et al. use 23/24 in vgg16 and Yang et al. use 36 in vgg19)')
    parser.add_argument(
        "--resume",
        default="",
        type=str,
        metavar="PATH",
        help="path to latest checkpoint (default: none)",
    )
    parser.add_argument(
        "--lr",
        default=0.0002,
        type=float,
        metavar="N",
        help="learning rate",
    )
    parser.add_argument(
        "--logging_step",
        default=10,
        type=int,
        metavar="N",
        help="save output images",
    )
    parser.add_argument(
        "--checkpoints",
        default="checkpoints",
        type=str,
        metavar="PATH",
        help="path to latest checkpoint (default: none)",
    )
    parser.add_argument(
        "--validation_interval",
        default=30,
        type=int,
        metavar="N",
        help="",
    )
    parser.add_argument(
        "--content_loss",
        default=False,
        type=str,
        metavar="str",
        help="",
    )
    parser.add_argument("--name", default="mutigpu", type=str, help="Experiment name")

    args = parser.parse_args()
    print(args)
    os.environ['MASTER_ADDR'] = 'localhost'   
    os.environ['MASTER_PORT'] = '12355'
    
    mp.spawn(main, nprocs=ngpus_per_node, args=(ngpus_per_node,args,),join = True)
    # main(args, num_epochs=args.epochs, resume=args.resume, name=args.name)
