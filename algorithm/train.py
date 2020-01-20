import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import time
from tensorboardX import SummaryWriter
import numpy as np
import sys
sys.path.append('../')
from algorithm.models import Color2Normal
# from algorithm.dataset import REALDataset  # TODO: Dataset class
import collections


def compute_angle_error(pred_normal, target_normal, gt_mask):
    cosin_sim = F.cosine_similarity(pred_normal, target_normal, dim=1, eps=1e-8)

    angular = gt_mask[:, 0, :, :] * torch.acos(torch.clamp(cosin_sim, -1, 1))  # [N, H, W]
    mae = torch.sum(angular, dim=(1, 2)) / torch.sum(gt_mask[:, 0, :, :], dim=(1, 2))
    mae = 180 * mae / 3.1415  # radius to degree
    return mae


def write_tensorboard(writer, results, epoch, mode):
    writer.add_scalar('%s/normal_loss' % mode, results['scalar/normal_loss'], epoch)
    writer.add_scalar('%s/category_loss' % mode, results['scalar/category_loss'], epoch)
    writer.add_scalar('%s/loss' % mode, results['scalar/loss'], epoch)
    writer.add_scalar('%s/mae' % mode, results['scalar/mae'], epoch)

    pred_mask = torch.stack(results['image/pred_mask'], dim=0)
    target_mask = torch.stack(results['image/target_mask'], dim=0)  # [N, 1, H, W]
    color_img = torch.stack(results['image/color'], dim=0)  # [N, 3, H, W]

    pred_normal = torch.stack(results['image/pred_normal'], dim=0)  # [N, 3, H, W]
    norm = torch.norm(pred_normal, p=2, dim=1, keepdim=True)  # [N, 1, H, W]
    pred_normal = pred_mask * pred_normal / norm

    target_normal = target_mask * torch.stack(results['image/target_normal'], dim=0)  # [N, 3, H, W]

    writer.add_images('%s/target_mask' % mode, color_img * target_mask, epoch)
    writer.add_images('%s/pred_mask' % mode, color_img * pred_mask, epoch)
    writer.add_images('%s/target_normal' % mode, (target_normal + 1) / 2, epoch)
    writer.add_images('%s/pred_normal' % mode, (pred_normal + 1) / 2, epoch)


def train_normal(args):
    assert args.task == 'polar2normal'
    # GPU or CPU configuration
    use_cuda = torch.cuda.is_available()
    torch.cuda.set_device("cuda:%s" % args.gpu_id if use_cuda else "cpu")
    device = torch.device("cuda:%s" % args.gpu_id if use_cuda else "cpu")
    dtype = torch.float32

    # set dataset
    # TODO: dataset

    # set model
    print('[task] color2normal')
    model = Color2Normal(temperature=args.temperature)

    # set optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr_start)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_step, gamma=args.lr_decay_rate)
    model = model.to(device=device)  # move the model parameters to CPU/GPU
    model_dir = '%s/model_color2normal.pkl' % args.result_dir
    writer = SummaryWriter('%s/%s/%s' % (args.result_dir, args.log_dir,
                                         time.strftime("%Y-%m-%d-%H-%M", time.localtime())))
    print('[tensorboard] %s/%s/%s' %
          (args.result_dir, args.log_dir, time.strftime("%Y-%m-%d-%H-%M", time.localtime())))

    # training
    best_loss = 1e5
    for epoch in range(args.epochs + 1):
        print('====================================== Epoch %i ========================================' % (epoch + 1))

        if epoch == 0:
            print('[status] update calssification')
            for name, param in model.named_parameters():
                if 'polar2normal.encoder1' in name or 'polar2normal.decoder1' in name:  # TODO: change parameter name
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        if epoch == int(args.epochs / 3):
            print('[status] update normal')
            for name, param in model.named_parameters():
                if 'polar2normal.encoder2' in name or 'polar2normal.decoder2' in name:  # TODO: change parameter name
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        if epoch == int(2 * args.epochs / 3):
            print('[status] fine-tune classification and normal')
            for name, param in model.named_parameters():  # TODO: change parameter name
                param.requires_grad = True

        # validation
        print('------------------------------------- Validation ------------------------------------')
        start_time = time.time()
        model.eval()  # dropout layers will not work in eval mode
        results = collections.defaultdict(list)
        with torch.set_grad_enabled(False):  # deactivate autograd to reduce memory usage
            for idx, batch_data in enumerate(val_generator):
                # batch_data: {polar_img, ambiguity_normal, normal, mask, joint_uvd, joint_xyz, smpl_param, info}
                color_img = batch_data['color_img'].to(device=device, dtype=dtype)  # [N, 3, H, W]
                normal = batch_data['normal'].to(device=device, dtype=dtype)  # [N, 3, H, W] compute loss
                mask = batch_data['mask'].to(device=device, dtype=dtype)  # [N, 1, H, W] used in tensorboard
                category = batch_data['category'].to(device=device, dtype=torch.long)  # [N, 1, H, W] compute loss

                # pred_category, pred_normal, pred_mask, beta, theta, joint, normal_mask
                pred_category, pred_normal, pred_mask = model(color_img)

                # normal loss
                norm = torch.norm(pred_normal, p=2, dim=1, keepdim=False)  # [N, H, W]
                cosin_sim = F.cosine_similarity(pred_normal, normal, dim=1, eps=1e-8)
                # tmp = huber_loss_func(cosin_sim, 1, args.normal_huber_weight) + args.norm_weight * (norm - 1).pow(2)
                tmp = torch.abs(1 - cosin_sim) + args.norm_weight * torch.abs(norm - 1)
                normal_loss = torch.sum(mask[:, 0, :, :] * tmp, (1, 2)) / torch.sum(mask, (1, 2, 3))

                # category loss
                category_loss = torch.mean(
                    F.cross_entropy(pred_category, category[:, 0, :, :],
                                    reduction='none'), (1, 2))

                mae = compute_angle_error(pred_normal, normal, mask)

                # total loss for polar2normal
                val_loss = args.normal_loss_weight * normal_loss + args.category_loss_weight * category_loss

                # collect results
                results['scalar/normal_loss'].append(normal_loss)
                results['scalar/category_loss'].append(category_loss)
                results['scalar/loss'].append(val_loss)
                results['scalar/mae'].append(mae)
                if idx % 60 == 0:
                    results['image/pred_normal'].append(pred_normal[0])
                    results['image/target_normal'].append(normal[0])
                    results['image/pred_mask'].append(pred_mask[0])
                    results['image/target_mask'].append(mask[0])
                    results['image/color'].append(color_img[0])

            results['scalar/normal_loss'] = torch.mean(torch.cat(results['scalar/normal_loss'], dim=0))
            results['scalar/category_loss'] = torch.mean(torch.cat(results['scalar/category_loss'], dim=0))
            results['scalar/loss'] = torch.mean(torch.cat(results['scalar/loss'], dim=0))
            results['scalar/mae'] = torch.mean(torch.cat(results['scalar/mae'], dim=0))
            write_tensorboard(writer, results, epoch, 'val')

            if best_loss > results['scalar/loss']:
                torch.save(model.state_dict(), model_dir)
                best_loss = results['scalar/loss']
                print('>>> Model saved as {}... best loss {:.4f}'.format(model_dir, best_loss))

            end_time = time.time()
            print('>>> Validation loss: {:.4f} (best loss {:.4f})\n'
                  '               normal loss: {:.4f}\n'
                  '               category loss: {:.4f}\n'
                  '               mean angular error: {:.4f}\n'
                  '               time used: {:.2f} min'
                  .format(results['scalar/loss'], best_loss, results['scalar/normal_loss'],
                          results['scalar/category_loss'], results['scalar/mae'], (end_time - start_time) / 60.))

        if epoch == args.epochs:
            break
        # train
        print('------------------------------------- Training ------------------------------------')
        model.train()
        results = collections.defaultdict(list)
        start_time = time.time()
        for idx, batch_data in enumerate(train_generator):
            # batch_data: {polar_img, ambiguity_normal, normal, mask, joint_uvd, joint_xyz, smpl_param, info}
            color_img = batch_data['color_img'].to(device=device, dtype=dtype)  # [N, 3, H, W]
            normal = batch_data['normal'].to(device=device, dtype=dtype)  # [N, 3, H, W] compute loss
            mask = batch_data['mask'].to(device=device, dtype=dtype)  # [N, 1, H, W] used in tensorboard
            category = batch_data['category'].to(device=device, dtype=torch.long)  # [N, 1, H, W] compute loss

            optimizer.zero_grad()
            # pred_category, pred_normal, pred_mask
            pred_category, pred_normal, pred_mask = model(color_img)

            # normal loss
            norm = torch.norm(pred_normal, p=2, dim=1, keepdim=False)  # [N, H, W]
            cosin_sim = F.cosine_similarity(pred_normal, normal, dim=1, eps=1e-8)
            # tmp = huber_loss_func(cosin_sim, 1, args.normal_huber_weight) + args.norm_weight * (norm - 1).pow(2)
            tmp = torch.abs(1 - cosin_sim) + args.norm_weight * torch.abs(norm - 1)
            normal_loss = torch.sum(mask[:, 0, :, :] * tmp, (1, 2)) / torch.sum(mask, (1, 2, 3))

            # category loss
            category_loss = torch.mean(
                F.cross_entropy(pred_category, category[:, 0, :, :],
                                reduction='none'), (1, 2))

            mae = compute_angle_error(pred_normal, normal, mask)

            # total loss for polar2normal
            train_loss = torch.mean(args.normal_loss_weight * normal_loss + args.category_loss_weight * category_loss)

            train_loss.backward()
            optimizer.step()

            # collect results
            results['scalar/normal_loss'].append(torch.mean(normal_loss.detach()))
            results['scalar/category_loss'].append(torch.mean(category_loss.detach()))
            results['scalar/loss'].append(train_loss.detach())
            results['scalar/mae'].append(torch.mean(mae.detach()))

            if (idx + 1) % 1000 == 0:
                results['image/pred_normal'].append(pred_normal[0].detach())
                results['image/target_normal'].append(normal[0])
                results['image/pred_mask'].append(pred_mask[0].detach())
                results['image/target_mask'].append(mask[0])
                results['image/color'].append(color_img[0])

                print('>>> [epoch {:2d}/ iter {:6d}]\n'
                      '    loss: {:.4f}, normal loss: {:.4f}, category loss: {:.4f}, mae: {:.4f}.'
                      .format(epoch, idx + 1,
                              torch.mean(torch.stack(results['scalar/loss'], dim=0)),
                              torch.mean(torch.stack(results['scalar/normal_loss'], dim=0)),
                              torch.mean(torch.stack(results['scalar/category_loss'], dim=0)),
                              torch.mean(torch.stack(results['scalar/mae'], dim=0))))

        results['scalar/normal_loss'] = torch.mean(torch.stack(results['scalar/normal_loss'], dim=0))
        results['scalar/category_loss'] = torch.mean(torch.stack(results['scalar/category_loss'], dim=0))
        results['scalar/loss'] = torch.mean(torch.stack(results['scalar/loss'], dim=0))
        results['scalar/mae'] = torch.mean(torch.stack(results['scalar/mae'], dim=0))
        write_tensorboard(writer, results, epoch, args, 'train')
        scheduler.step()
        end_time = time.time()
        print('>>> [epoch {:2d}/ iter {:6d}]\n'
              '    training loss: {:.4f}, normal loss: {:.4f}, category loss: {:.4f}, mae: {:.4f}\n'
              '    lr: {:.6f}, time used: {:.2f} min.'
              .format(epoch, idx + 1, results['scalar/loss'], results['scalar/normal_loss'],
                      results['scalar/category_loss'], results['scalar/mae'],
                      scheduler.get_lr()[0], (end_time - start_time) / 60.))

    writer.close()


def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=str, default='1')
    parser.add_argument('--root_dir', type=str, default='/home/data/data_shihao')
    parser.add_argument('--result_dir', type=str, default='/home/shihao/exp')
    parser.add_argument('--log_dir', type=str, default='log')
    parser.add_argument('--computer', type=int, default=1)

    parser.add_argument('--norm_weight', type=float, default=0.2)
    parser.add_argument('--normal_huber_weight', type=float, default=0.1)
    parser.add_argument('--normal_loss_weight', type=float, default=1)
    parser.add_argument('--category_loss_weight', type=float, default=1)

    parser.add_argument('--temperature', type=float, default=10)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr_start', type=float, default=0.0001)
    parser.add_argument('--lr_decay_rate', type=float, default=0.1)
    parser.add_argument('--lr_decay_step', type=int, default=5)
    args = parser.parse_args()
    util.print_args(args)
    return args


def main():
    args = get_args()
    os.environ['OMP_NUM_THREADS'] = '1'
    torch.set_num_threads(1)
    train_normal(args)


if __name__ == '__main__':
    main()

