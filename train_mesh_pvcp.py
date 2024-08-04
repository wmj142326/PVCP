import os
import random
import copy
import time
import sys
import shutil
import argparse
import errno
import math
import numpy as np
from collections import defaultdict, OrderedDict
import tensorboardX
from tqdm import tqdm
from prettytable import PrettyTable
from datetime import datetime  

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR

from lib.utils.tools import *
from lib.model.loss import *
from lib.model.loss_mesh_pvcp import *
from lib.utils.utils_mesh import *
from lib.utils.utils_smpl import *
from lib.utils.utils_data import *
from lib.utils.learning import *
from lib.data.dataset_mesh import MotionSMPL
from lib.model.model_mesh_pvcp import MeshRegressor
from torch.utils.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/pretrain.yaml", help="Path to the config file.")
    parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH', help='checkpoint directory')
    parser.add_argument('-p', '--pretrained', default='checkpoint', type=str, metavar='PATH', help='pretrained checkpoint directory')
    parser.add_argument('-r', '--resume', default='', type=str, metavar='FILENAME', help='checkpoint to resume (file name)')
    parser.add_argument('-e', '--evaluate', default='', type=str, metavar='FILENAME', help='checkpoint to evaluate (file name)')
    parser.add_argument('-freq', '--print_freq', default=100)
    parser.add_argument('-ms', '--selection', default='latest_epoch.bin', type=str, metavar='FILENAME', help='checkpoint to finetune (file name)')
    parser.add_argument('-sd', '--seed', default=0, type=int, help='random seed')
    opts = parser.parse_args()
    return opts

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def validate(test_loader, model, criterion, dataset_name='h36m'):
    model.eval()
    print(f'===========> validating {dataset_name}')
    batch_time = AverageMeter()
    losses = AverageMeter()
    losses_dict = {'loss_3d_pos': AverageMeter(), 
                   'loss_3d_scale': AverageMeter(), 
                   'loss_3d_velocity': AverageMeter(),
                #    'loss_3d_acc': AverageMeter(),
                   'loss_lv': AverageMeter(), 
                   'loss_lg': AverageMeter(), 
                   'loss_a': AverageMeter(), 
                   'loss_av': AverageMeter(), 
                   'loss_pose': AverageMeter(), 
                   'loss_shape': AverageMeter(),
                   'loss_norm': AverageMeter(),
                   'loss_class': AverageMeter(),
    }
    mpjpes = AverageMeter()
    mpves = AverageMeter()
    results = defaultdict(list)
    smpl = SMPL(args.data_root, batch_size=1).cuda()
    J_regressor = smpl.J_regressor_h36m
    with torch.no_grad():
        end = time.time()
        for idx, (batch_input_2d, batch_input_sensor, batch_gt) in tqdm(enumerate(test_loader)):
            batch_size, clip_len = batch_input_2d.shape[:2]
            if torch.cuda.is_available():
                batch_gt['theta'] = batch_gt['theta'].cuda().float()
                batch_gt['kp_3d'] = batch_gt['kp_3d'].cuda().float()
                batch_gt['verts'] = batch_gt['verts'].cuda().float()
                batch_input_sensor['pose_class'] = batch_input_sensor['pose_class'].cuda().float() # pose_class
                batch_input_2d = batch_input_2d.cuda().float()
            output, out_class = model(batch_input_2d)    
            output_final = output
            if args.flip:
                batch_input_2d_flip = flip_data(batch_input_2d)
                output_flip, out_class = model(batch_input_2d_flip)
                output_flip_pose = output_flip[0]['theta'][:, :, :72]
                output_flip_shape = output_flip[0]['theta'][:, :, 72:]
                output_flip_pose = flip_thetas_batch(output_flip_pose)
                output_flip_pose = output_flip_pose.reshape(-1, 72)
                output_flip_shape = output_flip_shape.reshape(-1, 10)
                output_flip_smpl = smpl(
                    betas=output_flip_shape,
                    body_pose=output_flip_pose[:, 3:],
                    global_orient=output_flip_pose[:, :3],
                    pose2rot=True
                )
                output_flip_verts = output_flip_smpl.vertices.detach()*1000.0
                J_regressor_batch = J_regressor[None, :].expand(output_flip_verts.shape[0], -1, -1).to(output_flip_verts.device)
                output_flip_kp3d = torch.matmul(J_regressor_batch, output_flip_verts)  # (NT,17,3) 
                output_flip_back = [{
                    'theta': torch.cat((output_flip_pose.reshape(batch_size, clip_len, -1), output_flip_shape.reshape(batch_size, clip_len, -1)), dim=-1),
                    'verts': output_flip_verts.reshape(batch_size, clip_len, -1, 3),
                    'kp_3d': output_flip_kp3d.reshape(batch_size, clip_len, -1, 3),
                }]
                output_final = [{}]
                for k, v in output_flip[0].items():
                    output_final[0][k] = (output[0][k] + output_flip_back[0][k])*0.5
                output = output_final
            loss_dict = criterion(output, out_class, batch_gt, batch_input_sensor)
            loss = args.lambda_3d      * loss_dict['loss_3d_pos']      + \
                   args.lambda_scale   * loss_dict['loss_3d_scale']    + \
                   args.lambda_3dv     * loss_dict['loss_3d_velocity'] + \
                   args.lambda_lv      * loss_dict['loss_lv']          + \
                   args.lambda_lg      * loss_dict['loss_lg']          + \
                   args.lambda_a       * loss_dict['loss_a']           + \
                   args.lambda_av      * loss_dict['loss_av']          + \
                   args.lambda_shape   * loss_dict['loss_shape']       + \
                   args.lambda_pose    * loss_dict['loss_pose']        + \
                   args.lambda_norm    * loss_dict['loss_norm']        + \
                   args.lambda_class   * loss_dict['loss_class']       
                #    args.lambda_acc     * loss_dict['loss_3d_acc']      + \

            # update metric
            losses.update(loss.item(), batch_size)
            loss_str = ''
            for k, v in loss_dict.items():
                losses_dict[k].update(v.item(), batch_size)
                loss_str += '{0} {loss.val:.3f} ({loss.avg:.3f})\t'.format(k, loss=losses_dict[k])
            mpjpe, mpve = compute_error(output, batch_gt)
            mpjpes.update(mpjpe, batch_size)
            mpves.update(mpve, batch_size)
            
            for keys in output[0].keys():
                output[0][keys] = output[0][keys].detach().cpu().numpy()
                batch_gt[keys] = batch_gt[keys].detach().cpu().numpy()
            for keys in batch_input_sensor.keys():
                batch_input_sensor[keys] = batch_input_sensor[keys].detach().cpu().numpy()
            results['kp_3d'].append(output[0]['kp_3d'])
            results['verts'].append(output[0]['verts'])
            results['kp_3d_gt'].append(batch_gt['kp_3d'])
            results['verts_gt'].append(batch_gt['verts'])
            results['pose_class'].append(batch_input_sensor['pose_class'])  # pose_class

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            table_1 = PrettyTable([ 'Test', 
                                    'Time (Avg)', 
                                    'Loss (Avg)', 
                                    'PVE (Avg)', 
                                    'JPE (Avg)'
                                    ])
            table_2 = PrettyTable([ 'Loss_3d_pos', 'Loss_3d_scale', 'Loss_3d_velocity', 'Loss_lv',   'Loss_lg', 
                                    'Loss_a',      'Loss_av',       'Loss_shape',       'Loss_pose', 'Loss_norm',   'Loss_class'
                                    ])
            if idx % int(len(test_loader)-1) == 0:         
                table_1.add_row([f'{idx+1}/{len(test_loader)}',
                            f'{batch_time.val:.3f}({batch_time.avg:.3f})', 
                            f'{losses.val:.3f}({losses.avg:.3f})',
                            f'{mpves.val:.3f}({mpves.avg:.3f})', 
                            f'{mpjpes.val:.3f}({mpjpes.avg:.3f})'])
                table_2.add_row([f'{losses_dict[k].val:.3f}({losses_dict[k].avg:.3f})' for k,v in losses_dict.items()])
                print("\n")
                print(table_1)
                print(table_2)
                
    print(f'==> start concating results of {dataset_name}')
    for term in results.keys():
        results[term] = np.concatenate(results[term])
    print(f'==> start evaluating {dataset_name}...')
    error_dict = evaluate_mesh(results)  # from lib.utils.utils_mesh import *
    err_str = ''
    for err_key, err_val in error_dict.items():
        err_str += '   {}: {:6.2f}mm'.format(err_key, err_val)
    print("----"*45)
    
    if args.val_pose_class:
        categories = [0, 1, 2, 3]
        for category in categories:
            mask = (results['pose_class'] == category)
            
            results_pc = defaultdict(list)
            for keys in results.keys():
                results_pc[keys] = [results[keys][mask]]
                
            for term in results_pc.keys():
                results_pc[term] = np.concatenate(results_pc[term])
                
            error_dict_pc = evaluate_mesh(results_pc)  # from lib.utils.utils_mesh import *
            err_str_pc = ''
            for err_key_pc, err_val_pc in error_dict_pc.items():
                err_str_pc += '   {}: {:6.2f}mm'.format(err_key_pc, err_val_pc)
            print(f'==================> {dataset_name} Pose_class {np.count_nonzero(mask):4d} pose{category}: ', err_str_pc)
    print(f'==================> {dataset_name} validation done mean : ', err_str)
    print("----"*45)

    return losses.avg, error_dict['mpjpe_14j'], error_dict['pa_mpjpe_14j'], error_dict['mpve'], error_dict['pa_mpve'], losses_dict, err_str

def train_epoch(args, opts, model, train_loader, losses_train, losses_dict, mpjpes, mpves, criterion, optimizer, batch_time, data_time, epoch):
    model.train()
    end = time.time()
    for idx, (batch_input_2d, batch_input_sensor, batch_gt) in tqdm(enumerate(train_loader)):
        data_time.update(time.time() - end)
        batch_size = len(batch_input_2d)

        if torch.cuda.is_available():
            batch_gt['theta'] = batch_gt['theta'].cuda().float()
            batch_gt['kp_3d'] = batch_gt['kp_3d'].cuda().float()
            batch_gt['verts'] = batch_gt['verts'].cuda().float()
            batch_input_2d = batch_input_2d.cuda().float()
        output, out_class = model(batch_input_2d)
        optimizer.zero_grad()
        loss_dict = criterion(output, out_class, batch_gt, batch_input_sensor)
        loss_train = args.lambda_3d      * loss_dict['loss_3d_pos']      + \
                     args.lambda_scale   * loss_dict['loss_3d_scale']    + \
                     args.lambda_3dv     * loss_dict['loss_3d_velocity'] + \
                     args.lambda_lv      * loss_dict['loss_lv']          + \
                     args.lambda_lg      * loss_dict['loss_lg']          + \
                     args.lambda_a       * loss_dict['loss_a']           + \
                     args.lambda_av      * loss_dict['loss_av']          + \
                     args.lambda_shape   * loss_dict['loss_shape']       + \
                     args.lambda_pose    * loss_dict['loss_pose']        + \
                     args.lambda_norm    * loss_dict['loss_norm']        + \
                     args.lambda_class   * loss_dict['loss_class']
                    # args.lambda_acc     * loss_dict['loss_3d_acc']      + \

                           
        losses_train.update(loss_train.item(), batch_size)
        loss_str = ''
        for k, v in loss_dict.items():
            losses_dict[k].update(v.item(), batch_size)
            loss_str += '{0} {loss.val:.3f} ({loss.avg:.3f})\t'.format(k, loss=losses_dict[k])
        
        mpjpe, mpve = compute_error(output, batch_gt)
        mpjpes.update(mpjpe, batch_size)
        mpves.update(mpve, batch_size)
        
        loss_train.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()
        table_1 = PrettyTable(['Epoch', 'Batch/Total', 
                             'Batch Time (Avg)', 
                             'Data Time (Avg)', 
                             'Loss (Avg)', 
                             'PVE (Avg)', 
                             'JPE (Avg)'
                             ])
        table_2 = PrettyTable(['Loss_3d_pos', 'Loss_3d_scale', 'Loss_3d_velocity', 'Loss_lv',   'Loss_lg', 
                               'Loss_a',      'Loss_av',       'Loss_shape',       'Loss_pose', 'Loss_norm',   'Loss_class'
                               ])
        if idx % int(opts.print_freq) == 0:
            table_1.add_row([f'{epoch}/{args.epochs}', f'{idx + 1}/{len(train_loader)}',
                           f'{batch_time.val:.3f}({batch_time.avg:.3f})', 
                           f'{data_time.val:.3f}({data_time.avg:.3f})', 
                           f'{losses_train.val:.3f}({losses_train.avg:.3f})',
                           f'{mpves.val:.3f}({mpves.avg:.3f})', 
                           f'{mpjpes.val:.3f}({mpjpes.avg:.3f})'])

            table_2.add_row([f'{losses_dict[k].val:.3f}({losses_dict[k].avg:.3f})' for k,v in losses_dict.items()])
            print("\n")
            print(table_1)
            print(table_2)

            # print('Train: [{0}][{1}/{2}]\t'
            #     'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            #     'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
            #     'loss {loss.val:.3f} ({loss.avg:.3f})\t'
            #     '{3}'
            #     'PVE {mpves.val:.3f} ({mpves.avg:.3f})\t'
            #     'JPE {mpjpes.val:.3f} ({mpjpes.avg:.3f})'.format(
            #     epoch, idx + 1, len(train_loader), loss_str, batch_time=batch_time,
            #     data_time=data_time, loss=losses_train, mpves=mpves, mpjpes=mpjpes))
            sys.stdout.flush()

def train_with_config(args, opts):
    print(args)
    try:
        os.makedirs(opts.checkpoint)
        shutil.copy(opts.config, opts.checkpoint)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise RuntimeError('Unable to create checkpoint directory:', opts.checkpoint)
    train_writer = tensorboardX.SummaryWriter(os.path.join(opts.checkpoint, "logs"))
    model_backbone = load_backbone(args)
    if args.finetune:
        if opts.resume or opts.evaluate:
            pass
        else:
            chk_filename = os.path.join(opts.pretrained, opts.selection)
            print('Loading backbone', chk_filename)
            checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)['model_pos']
            model_backbone = load_pretrained_weights(model_backbone, checkpoint)
    if args.partial_train:
        model_backbone = partial_train_layers(model_backbone, args.partial_train)
    model = MeshRegressor(args, backbone=model_backbone, dim_rep=args.dim_rep, hidden_dim=args.hidden_dim, dropout_ratio=args.dropout, num_joints=args.num_joints)
    criterion = MeshLoss(loss_type = args.loss_type)
    best_jpe = 9999.0
    model_params = 0
    for parameter in model.parameters():
        if parameter.requires_grad == True:
            model_params = model_params + parameter.numel()
    print('INFO: Trainable parameter count:', model_params)
    print('Loading dataset...')
    trainloader_params = {
          'batch_size': args.batch_size,
          'shuffle': True,
          'num_workers': 8,
          'pin_memory': True,
          'prefetch_factor': 4,
          'persistent_workers': True,
          'drop_last':args.drop_last

    }
    testloader_params = {
          'batch_size': args.batch_size,
          'shuffle': False,
          'num_workers': 8,
          'pin_memory': True,
          'prefetch_factor': 4,
          'persistent_workers': True,
          'drop_last':args.drop_last

    }
    if hasattr(args, "dt_file_h36m"):
        mesh_train = MotionSMPL(args, data_split='train', dataset="h36m")
        mesh_val = MotionSMPL(args, data_split='test', dataset="h36m")
        train_loader = DataLoader(mesh_train, **trainloader_params)
        test_loader = DataLoader(mesh_val, **testloader_params)
        print('INFO: Training on {} batches (h36m)'.format(len(train_loader)))

    if hasattr(args, "dt_file_pw3d"):
        if args.train_pw3d:
            mesh_train_pw3d = MotionSMPL(args, data_split='train', dataset="pw3d")
            train_loader_pw3d = DataLoader(mesh_train_pw3d, **trainloader_params)
            print('INFO: Training on {} batches (pw3d)'.format(len(train_loader_pw3d)))
        mesh_val_pw3d = MotionSMPL(args, data_split='test', dataset="pw3d")
        test_loader_pw3d = DataLoader(mesh_val_pw3d, **testloader_params)

    if hasattr(args, "dt_file_pedx"):
        if args.train_pedx:
            mesh_train_pedx = MotionSMPL(args, data_split='train', dataset="pedx")
            train_loader_pedx = DataLoader(mesh_train_pedx, **trainloader_params)
            print('INFO: Training on {} batches (pedx)'.format(len(train_loader_pedx)))
        # mesh_val_pedx = MotionSMPL(args, data_split='test', dataset="pedx")
        # test_loader_pedx = DataLoader(mesh_val_pedx, **testloader_params)
        
    if hasattr(args, "dt_file_pvcp"):
        if args.train_pvcp:
            mesh_train_pvcp = MotionSMPL(args, data_split='train', dataset="pvcp")
            train_loader_pvcp = DataLoader(mesh_train_pvcp, **trainloader_params)
            print('INFO: Training on {} batches (pvcp)'.format(len(train_loader_pvcp)))
        mesh_val_pvcp = MotionSMPL(args, data_split='test', dataset="pvcp")
        test_loader_pvcp = DataLoader(mesh_val_pvcp, **testloader_params) 
    
    trainloader_img_params = {
            'batch_size': args.batch_size_img,
            'shuffle': True,
            'num_workers': 8,
            'pin_memory': True,
            'prefetch_factor': 4,
            'persistent_workers': True,
            'drop_last':args.drop_last

        }
    testloader_img_params = {
            'batch_size': args.batch_size_img,
            'shuffle': False,
            'num_workers': 8,
            'pin_memory': True,
            'prefetch_factor': 4,
            'persistent_workers': True,
            'drop_last':args.drop_last
        }
    
    if hasattr(args, "dt_file_coco"):
        mesh_train_coco = MotionSMPL(args, data_split='train', dataset="coco")
        mesh_val_coco = MotionSMPL(args, data_split='test', dataset="coco")
        train_loader_coco = DataLoader(mesh_train_coco, **trainloader_img_params)
        test_loader_coco = DataLoader(mesh_val_coco, **testloader_img_params)
        print('INFO: Training on {} batches (coco)'.format(len(train_loader_coco)))

    if torch.cuda.is_available():
        model = nn.DataParallel(model)
        model = model.cuda()
        
    chk_filename = os.path.join(opts.checkpoint, "latest_epoch.bin")
    if os.path.exists(chk_filename):
        opts.resume = chk_filename
    if opts.resume or opts.evaluate:
        chk_filename = opts.evaluate if opts.evaluate else opts.resume
        print('Loading checkpoint', chk_filename)
        checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['model'], strict=True)
    if not opts.evaluate:
        optimizer = optim.AdamW(
                [     {"params": filter(lambda p: p.requires_grad, model.module.backbone.parameters()), "lr": args.lr_backbone},
                      {"params": filter(lambda p: p.requires_grad, model.module.head.parameters()), "lr": args.lr_head},
                ],      lr=args.lr_backbone, 
                        weight_decay=args.weight_decay
        )
        scheduler = StepLR(optimizer, step_size=1, gamma=args.lr_decay)
        st = 0
        if opts.resume:
            st = checkpoint['epoch']
            if 'optimizer' in checkpoint and checkpoint['optimizer'] is not None:
                optimizer.load_state_dict(checkpoint['optimizer'])
            else:
                print('WARNING: this checkpoint does not contain an optimizer state. The optimizer will be reinitialized.')
            lr = checkpoint['lr']
            if 'best_jpe' in checkpoint and checkpoint['best_jpe'] is not None:
                best_jpe = checkpoint['best_jpe']
        
        # Training
        for epoch in range(st, args.epochs):
            print("\n", "\n", '###'*25,'Training epoch %d.' % epoch, '###'*25,)
            losses_train = AverageMeter()
            losses_dict = {
                'loss_3d_pos': AverageMeter(), 
                'loss_3d_scale': AverageMeter(), 
                'loss_3d_velocity': AverageMeter(),
                # 'loss_3d_acc': AverageMeter(),
                'loss_lv': AverageMeter(), 
                'loss_lg': AverageMeter(), 
                'loss_a': AverageMeter(), 
                'loss_av': AverageMeter(), 
                'loss_pose': AverageMeter(), 
                'loss_shape': AverageMeter(),
                'loss_norm': AverageMeter(),
                'loss_class': AverageMeter(),
            }
            mpjpes = AverageMeter()
            mpves = AverageMeter()
            batch_time = AverageMeter()
            data_time = AverageMeter()
            
            if hasattr(args, "dt_file_h36m") and epoch < args.warmup_h36m:
                print("===========>training h36m..." )
                train_epoch(args, opts, model, train_loader, losses_train, losses_dict, mpjpes, mpves, criterion, optimizer, batch_time, data_time, epoch)
                test_loss, test_mpjpe, test_pa_mpjpe, test_mpve, test_pa_mpve, test_losses_dict, test_err_str_h36m = validate(test_loader, model, criterion, 'h36m')
                for k, v in test_losses_dict.items():
                    train_writer.add_scalar('test_loss/'+k, v.avg, epoch + 1)
                train_writer.add_scalar('test_loss', test_loss, epoch + 1)
                train_writer.add_scalar('test_mpjpe', test_mpjpe, epoch + 1)
                train_writer.add_scalar('test_pa_mpjpe', test_pa_mpjpe, epoch + 1)
                train_writer.add_scalar('test_mpve', test_mpve, epoch + 1)
                train_writer.add_scalar('test_pa_mpve', test_pa_mpve, epoch + 1)

                with open(os.path.join(opts.checkpoint, f'result_h36m.txt'), 'a', encoding='utf-8') as f: 
                    if epoch==0 or epoch==args.epochs-1:
                        f.write("---"*20 + f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}' + "---"*20 + "\n")
                    f.write(str([f"epoch{epoch:3d}"]+[test_err_str_h36m]) + "\n")
                    
            if hasattr(args, "dt_file_coco") and epoch < args.warmup_coco:
                print("===========>training  coco..." )
                train_epoch(args, opts, model, train_loader_coco, losses_train, losses_dict, mpjpes, mpves, criterion, optimizer, batch_time, data_time, epoch)
                test_loss, test_mpjpe, test_pa_mpjpe, test_mpve, test_pa_mpve, test_losses_dict, test_err_str_coco = validate(test_loader, model, criterion, 'coco')
                with open(os.path.join(opts.checkpoint, f'result_coco.txt'), 'a', encoding='utf-8') as f:  
                    if epoch==0 or epoch==args.epochs-1:
                        f.write("---"*20 + f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}' + "---"*20 + "\n")
                    f.write(str([f"epoch{epoch:3d}"]+[test_err_str_coco]) + "\n")
                    
            if hasattr(args, "dt_file_pw3d"):
                if args.train_pw3d:
                    print("===========>training  pw3d..." )
                    train_epoch(args, opts, model, train_loader_pw3d, losses_train, losses_dict, mpjpes, mpves, criterion, optimizer, batch_time, data_time, epoch)    
                test_loss_pw3d, test_mpjpe_pw3d, test_pa_mpjpe_pw3d, test_mpve_pw3d, test_pa_mpve_pw3d, test_losses_dict_pw3d, test_err_str_pw3d = validate(test_loader_pw3d, model, criterion, 'pw3d')  
                for k, v in test_losses_dict_pw3d.items():
                    train_writer.add_scalar('test_loss_pw3d/'+k, v.avg, epoch + 1)
                train_writer.add_scalar('test_loss_pw3d', test_loss_pw3d, epoch + 1)
                train_writer.add_scalar('test_mpjpe_pw3d', test_mpjpe_pw3d, epoch + 1)
                train_writer.add_scalar('test_pa_mpjpe_pw3d', test_pa_mpjpe_pw3d, epoch + 1)
                train_writer.add_scalar('test_mpve_pw3d', test_mpve_pw3d, epoch + 1)
                train_writer.add_scalar('test_pa_mpve_pw3d', test_pa_mpve_pw3d, epoch + 1)

                with open(os.path.join(opts.checkpoint, f'result_pw3d.txt'), 'a', encoding='utf-8') as f: 
                    if epoch==0 or epoch==args.epochs-1:
                        f.write("---"*20 + f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}' + "---"*20 + "\n")
                    f.write(str([f"epoch{epoch:3d}"]+[test_err_str_pw3d]) + "\n")
                
            if hasattr(args, "dt_file_pedx"):
                if args.train_pvcp:
                    print("===========>training  pedx..." )
                    train_epoch(args, opts, model, train_loader_pedx, losses_train, losses_dict, mpjpes, mpves, criterion, optimizer, batch_time, data_time, epoch)    
            
            if hasattr(args, "dt_file_pvcp"):
                if args.train_pvcp:
                    print("===========>training  pvcp..." )
                    train_epoch(args, opts, model, train_loader_pvcp, losses_train, losses_dict, mpjpes, mpves, criterion, optimizer, batch_time, data_time, epoch)    
                test_loss_pvcp, test_mpjpe_pvcp, test_pa_mpjpe_pvcp, test_mpve_pvcp, test_pa_mpve_pvcp, test_losses_dict_pvcp, test_err_str_pvcp = validate(test_loader_pvcp, model, criterion, 'pvcp')  
                for k, v in test_losses_dict_pvcp.items():
                    train_writer.add_scalar('test_loss_pvcp/'+k, v.avg, epoch + 1)
                train_writer.add_scalar('test_loss_pvcp', test_loss_pvcp, epoch + 1)
                train_writer.add_scalar('test_mpjpe_pvcp', test_mpjpe_pvcp, epoch + 1)
                train_writer.add_scalar('test_pa_mpjpe_pvcp', test_pa_mpjpe_pvcp, epoch + 1)
                train_writer.add_scalar('test_mpve_pvcp', test_mpve_pvcp, epoch + 1)
                train_writer.add_scalar('test_pa_mpve_pvcp', test_pa_mpve_pvcp, epoch + 1)

                with open(os.path.join(opts.checkpoint, f'result_pvcp.txt'), 'a', encoding='utf-8') as f:  
                    if epoch==0 or epoch==args.epochs-1:
                        f.write("---"*20 + f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}' + "---"*20 + "\n")
                    f.write(str([f"epoch{epoch:3d}"]+[test_err_str_pvcp]) + "\n")
                    
            for k, v in losses_dict.items():
                train_writer.add_scalar('train_loss/'+k, v.avg, epoch + 1)
            train_writer.add_scalar('train_loss', losses_train.avg, epoch + 1)
            train_writer.add_scalar('train_mpjpe', mpjpes.avg, epoch + 1)
            train_writer.add_scalar('train_mpve', mpves.avg, epoch + 1)

                
            # Decay learning rate exponentially
            scheduler.step()
            # Save latest checkpoint.
            chk_path = os.path.join(opts.checkpoint, 'latest_epoch.bin')
            print('Saving checkpoint to', chk_path)
            torch.save({
                'epoch': epoch+1,
                'lr': scheduler.get_last_lr(),
                'optimizer': optimizer.state_dict(),
                'model': model.state_dict(),
                'best_jpe' : best_jpe
            }, chk_path)
            
            # Save checkpoint if necessary.
            if (epoch+1) % args.checkpoint_frequency == 0:
                chk_path = os.path.join(opts.checkpoint, 'epoch_{}.bin'.format(epoch))
                print('Saving checkpoint to', chk_path)
            torch.save({
                'epoch': epoch+1,
                'lr': scheduler.get_last_lr(),
                'optimizer': optimizer.state_dict(),
                'model': model.state_dict(),
                'best_jpe' : best_jpe
            }, chk_path)

            # if hasattr(args, "dt_file_pw3d"):
            #     best_jpe_cur = test_mpjpe_pw3d
            # else:
            #     best_jpe_cur = test_mpjpe
                
            if hasattr(args, "dt_file_pvcp"):
                best_jpe_cur = test_mpve_pvcp
            else:
                best_jpe_cur = test_mpve
                    
            # Save best checkpoint.
            best_chk_path = os.path.join(opts.checkpoint, 'best_epoch.bin'.format(epoch))
            if best_jpe_cur < best_jpe:
                best_jpe = best_jpe_cur
                print("save best checkpoint")
                torch.save({
                'epoch': epoch+1,
                'lr': scheduler.get_last_lr(),
                'optimizer': optimizer.state_dict(),
                'model': model.state_dict(),
                'best_jpe' : best_jpe
                }, best_chk_path)
    
    if opts.evaluate:
        if hasattr(args, "dt_file_h36m"):
            test_loss, test_mpjpe, test_pa_mpjpe, test_mpve, test_pa_mpve, _, test_err_str = validate(test_loader, model, criterion, 'h36m')
        if hasattr(args, "dt_file_pw3d"):
            test_loss_pw3d, test_mpjpe_pw3d, test_pa_mpjpe_pw3d, test_mpve_pw3d, test_pa_mpve_pw3d, _, test_err_str = validate(test_loader_pw3d, model, criterion, 'pw3d')
        if hasattr(args, "dt_file_pvcp"):
            test_loss_pvcp, test_mpjpe_pvcp, test_pa_mpjpe_pvcp, test_mpve_pvcp, test_pa_mpve_pvcp, _, test_err_str = validate(test_loader_pvcp, model, criterion, 'pvcp')

if __name__ == "__main__":
    opts = parse_args()
    set_random_seed(opts.seed)
    args = get_config(opts.config)
    train_with_config(args, opts)
