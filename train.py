import os
#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:1024"
from config import Config 
opt = Config('training.yml')

# Create log files
log_dir = os.path.join(opt.TRAINING.SAVE_DIR, opt.MODEL.MODE, 'logs', opt.MODEL.SESSION)
os.makedirs(log_dir, exist_ok=True)
psnr_log_path = os.path.join(log_dir, 'psnr_log.txt')
ssim_log_path = os.path.join(log_dir, 'ssim_log.txt')
train_log_path = os.path.join(log_dir, 'train_log.txt')
gradient_log_path = os.path.join(log_dir, 'gradient_log.txt')

gpus = ','.join([str(i) for i in opt.GPU])
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpus

import torch
torch.backends.cudnn.benchmark = True

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from pytorch_msssim import ssim  # Add SSIM calculation library

import random
import time
import numpy as np

import utils
from data_RGB import get_training_data, get_validation_data
from PMUNet import PMUNet
import losses
from warmup_scheduler import GradualWarmupScheduler
from tqdm import tqdm
from pdb import set_trace as stx

class GradientMonitor:
    def __init__(self):
        self.grad_norms = []
        self.clipped_counts = 0
        self.total_counts = 0
    
    def update(self, grad_norm, is_clipped):
        self.grad_norms.append(grad_norm)
        self.total_counts += 1
        if is_clipped:
            self.clipped_counts += 1
    
    def get_stats(self):
        if not self.grad_norms:
            return {}
        
        norms = np.array(self.grad_norms)
        return {
            'mean': np.mean(norms),
            'std': np.std(norms),
            'max': np.max(norms),
            'min': np.min(norms),
            'clipped_ratio': self.clipped_counts / self.total_counts if self.total_counts > 0 else 0
        }
    
    def reset(self):
        self.grad_norms = []
        self.clipped_counts = 0
        self.total_counts = 0

if __name__ == '__main__':

    ######### Set Seeds ###########
    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)

    start_epoch = 1
    mode = opt.MODEL.MODE
    session = opt.MODEL.SESSION

    result_dir = os.path.join(opt.TRAINING.SAVE_DIR, mode, 'results', session)
    model_dir  = os.path.join(opt.TRAINING.SAVE_DIR, mode, 'models',  session)

    utils.mkdir(result_dir)
    utils.mkdir(model_dir)

    train_dir = opt.TRAINING.TRAIN_DIR
    val_dir   = opt.TRAINING.VAL_DIR

    ######### Model ###########
    model_restoration = PMUNet()
    model_restoration.cuda()

    device_ids = [i for i in range(torch.cuda.device_count())]
    if torch.cuda.device_count() > 1:
      print("\n\nLet's use", torch.cuda.device_count(), "GPUs!\n\n")


    # Reduce initial learning rate
    new_lr = opt.OPTIM.LR_INITIAL 
    #* 0.1  # Reduce learning rate to 1/10 of original

    optimizer = optim.AdamW(model_restoration.parameters(), lr=new_lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-4)


    ######### Scheduler ###########
    warmup_epochs = 3
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.OPTIM.NUM_EPOCHS-warmup_epochs, eta_min=opt.OPTIM.LR_MIN)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
    scheduler.step()

    ######### Resume ###########
    if opt.TRAINING.RESUME:
        path_chk_rest = utils.get_last_path(model_dir, '_latest.pth')
        utils.load_checkpoint(model_restoration,path_chk_rest)
        start_epoch = utils.load_start_epoch(path_chk_rest) + 1
        utils.load_optim(optimizer, path_chk_rest)

        for i in range(1, start_epoch):
            scheduler.step()
        new_lr = scheduler.get_lr()[0]
        print('------------------------------------------------------------------------------')
        print("==> Resuming Training with learning rate:", new_lr)
        print('------------------------------------------------------------------------------')

    if len(device_ids)>1:
        model_restoration = nn.DataParallel(model_restoration, device_ids = device_ids)

    ######### Loss - Multi-Domain Loss for HDM ###########
    # Wavelet weight configuration: Stage1(0.0) Stage2(0.05) Stage3(0.00) - conservative and progressive
    criterion_multi_domain = losses.MultiDomainLoss(
        spatial_weight=1.0,    # Spatial domain weight (main loss)
        edge_weight=0.05       # Edge weight
        # Wavelet weight is fixed internally, no external configuration
    )
    
    # Keep the original loss function for comparison (optional)
    criterion_char = losses.CharbonnierLoss()
    criterion_edge = losses.EdgeLoss()

    ######### DataLoaders ###########
    train_dataset = get_training_data(train_dir, {'patch_size':opt.TRAINING.TRAIN_PS})
    train_loader = DataLoader(dataset=train_dataset, batch_size=opt.OPTIM.BATCH_SIZE, shuffle=True, num_workers=16, drop_last=False, pin_memory=True)

    val_dataset = get_validation_data(val_dir, {'patch_size':opt.TRAINING.VAL_PS})
    val_loader = DataLoader(dataset=val_dataset, batch_size=16, shuffle=False, num_workers=8, drop_last=False, pin_memory=True)

    print('===> Start Epoch {} End Epoch {}'.format(start_epoch,opt.OPTIM.NUM_EPOCHS + 1))
    print('===> Loading datasets')

    use_grad_clip = opt.OPTIM.USE_GRAD_CLIP
    grad_clip_norm = opt.OPTIM.GRAD_CLIP_NORM
    log_gradient_stats = opt.OPTIM.LOG_GRADIENT_STATS
    gradient_log_interval = opt.OPTIM.GRADIENT_LOG_INTERVAL

    if use_grad_clip:
        print(f'===> Gradient clipping enabled with max_norm={grad_clip_norm}')
        with open(gradient_log_path, 'w') as f:
            f.write("Gradient Clipping Log\n")
            f.write("=" * 50 + "\n")
            f.write(f"Max Norm: {grad_clip_norm}\n")
            f.write(f"Log Interval: {gradient_log_interval} epochs\n")
            f.write("=" * 50 + "\n")

    gradient_monitor = GradientMonitor()

    best_psnr = 0
    best_epoch = 0

    for epoch in range(start_epoch, opt.OPTIM.NUM_EPOCHS + 1):
        epoch_start_time = time.time()
        epoch_loss = 0
        train_id = 1

        # Initialize multi-domain loss statistics
        epoch_loss_details = {
            'spatial': 0, 'edge': 0, 'amplitude': 0, 
            'phase': 0, 'wavelet': 0, 'total_stage': 0
        }

        gradient_monitor.reset()

        model_restoration.train()
        for i, data in enumerate(tqdm(train_loader), 0):
            # zero_grad
            for param in model_restoration.parameters():
                param.grad = None

            target = data[0].cuda()
            input_ = data[1].cuda()

            restored = model_restoration(input_)
            
            # ===== Use multi-domain fusion loss =====
            total_loss = 0
            batch_loss_details = {
                'spatial': 0, 'edge': 0, 'wavelet': 0, 'total_stage': 0
            }
            
            # Calculate multi-domain loss for each stage - differentiated loss strategy
            # restored order: [stage3, stage2, stage1]
            stage_types = ["stage3", "stage2", "stage1"]  # Corresponding to restored index order
            
            for j in range(len(restored)):
                stage_type = stage_types[j]
                stage_loss, stage_loss_dict = criterion_multi_domain(restored[j], target, stage_type=stage_type)
                total_loss += stage_loss
                
                # Accumulate domain loss statistics
                for key in ['spatial', 'edge', 'wavelet']:
                    batch_loss_details[key] += stage_loss_dict[key].item()
                batch_loss_details['total_stage'] += stage_loss.item()
                
                # Print loss details for each stage (for debugging)
                if i % 50 == 0:  # Print every 50 batches
                    print(f"      {stage_type}: spatial={stage_loss_dict['spatial'].item():.6f}, "
                          f"edge={stage_loss_dict['edge'].item():.6f}, "
                          
                          f"wav={stage_loss_dict['wavelet'].item():.6f}")
            
            # Accumulate to epoch-level statistics
            for key in batch_loss_details:
                epoch_loss_details[key] += batch_loss_details[key]
            
            loss = total_loss

            loss.backward()
            
            if use_grad_clip:
                grad_norm_before = torch.nn.utils.clip_grad_norm_(
                    model_restoration.parameters(), float('inf'), norm_type=2
                )
                
                grad_norm_after = torch.nn.utils.clip_grad_norm_(
                    model_restoration.parameters(), grad_clip_norm, norm_type=2
                )
                
                is_clipped = grad_norm_before > grad_clip_norm
                gradient_monitor.update(grad_norm_before.item(), is_clipped)
                
                if grad_norm_before > grad_clip_norm * 100:
                    print(f"\n⚠️  Warning: Very large gradient norm {grad_norm_before:.6f} at epoch {epoch}, step {i}")
                    with open(gradient_log_path, 'a') as f:
                        f.write(f"[WARNING] Epoch {epoch}, Step {i}: Large gradient norm {grad_norm_before:.6f}\n")
            
            optimizer.step()
            epoch_loss +=loss.item()

        #### Evaluation ####
        if epoch%opt.TRAINING.VAL_AFTER_EVERY == 0:
            model_restoration.eval()
            psnr_val_rgb = []
            ssim_val_rgb = []
            for ii, data_val in enumerate(val_loader, 0):
                target = data_val[0].cuda()
                input_ = data_val[1].cuda()

                with torch.no_grad():
                    restored = model_restoration(input_)
                restored = restored[0]

                for res,tar in zip(restored,target):
                    psnr_val_rgb.append(utils.torchPSNR(res, tar))
                    ssim_val_rgb.append(ssim(res.unsqueeze(0), tar.unsqueeze(0), data_range=1.0))

            psnr_val_rgb  = torch.stack(psnr_val_rgb).mean().item()
            ssim_val_rgb  = torch.stack(ssim_val_rgb).mean().item()

            if psnr_val_rgb > best_psnr:
                best_psnr = psnr_val_rgb
                best_epoch = epoch
                torch.save({'epoch': epoch,
                            'state_dict': model_restoration.state_dict(),
                            'optimizer' : optimizer.state_dict()
                            }, os.path.join(model_dir,"model_best.pth"))

            print("[epoch %d PSNR: %.4f SSIM: %.4f --- best_epoch %d Best_PSNR %.4f]" % (epoch, psnr_val_rgb, ssim_val_rgb, best_epoch, best_psnr))
            # Record PSNR and SSIM information
            with open(psnr_log_path, 'a') as f:
                f.write("[epoch %d PSNR: %.4f]\n" % (epoch, psnr_val_rgb))
            with open(ssim_log_path, 'a') as f:
                f.write("[epoch %d SSIM: %.4f]\n" % (epoch, ssim_val_rgb))

            torch.save({'epoch': epoch,
                        'state_dict': model_restoration.state_dict(),
                        'optimizer' : optimizer.state_dict()
                        }, os.path.join(model_dir,f"model_epoch_{epoch}.pth"))

        scheduler.step()

        if use_grad_clip and log_gradient_stats:
            grad_stats = gradient_monitor.get_stats()
            
            grad_info = f"Epoch {epoch}: Grad_Mean={grad_stats['mean']:.6f}, Grad_Max={grad_stats['max']:.6f}, Clipped={grad_stats['clipped_ratio']:.2%}"
            
            if epoch % gradient_log_interval == 0:
                detailed_info = (
                    f"[EPOCH {epoch}] Gradient Statistics:\n"
                    f"  Mean: {grad_stats['mean']:.6f}\n"
                    f"  Std:  {grad_stats['std']:.6f}\n"
                    f"  Max:  {grad_stats['max']:.6f}\n"
                    f"  Min:  {grad_stats['min']:.6f}\n"
                    f"  Clipped Ratio: {grad_stats['clipped_ratio']:.2%}\n"
                    f"  Learning Rate: {scheduler.get_lr()[0]:.8f}\n"
                    f"  Epoch Loss: {epoch_loss:.6f}\n"
                    f"{'-' * 40}\n"
                )
                
                with open(gradient_log_path, 'a') as f:
                    f.write(detailed_info)
                
                print(f" {grad_info}")
            
            with open(gradient_log_path, 'a') as f:
                f.write(f"{grad_info}\n")

        print("------------------------------------------------------------------")
        print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.8f}".format(epoch, time.time()-epoch_start_time, epoch_loss, scheduler.get_lr()[0]))
        
        # ===== Multi-domain loss detailed statistics =====
        num_samples = len(train_loader)
        avg_losses = {k: v/num_samples for k, v in epoch_loss_details.items()}
        
        print("🔬 Multi-Domain Loss Details:")
        print(f"   Spatial: {avg_losses['spatial']:.6f}")
        print(f"   Edge: {avg_losses['edge']:.6f}")  
        
        print(f"   Wavelet: {avg_losses['wavelet']:.6f}")
        print(f"   Stage Total: {avg_losses['total_stage']:.6f}")
        
        print("------------------------------------------------------------------")
        
        # Record training information
        with open(train_log_path, 'a') as f:
            basic_info = "Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.8f}\n".format(
                epoch, time.time()-epoch_start_time, epoch_loss, scheduler.get_lr()[0])
            f.write(basic_info)
            
            # Record multi-domain loss details
            multi_domain_info = (
                f"Multi-Domain Losses - Spatial: {avg_losses['spatial']:.6f}, "
                            f"Edge: {avg_losses['edge']:.6f}, Wavelet: {avg_losses['wavelet']:.6f}\n"
            )
            f.write(multi_domain_info)

        torch.save({'epoch': epoch,
                    'state_dict': model_restoration.state_dict(),
                    'optimizer' : optimizer.state_dict()
                    }, os.path.join(model_dir,"model_latest.pth"))

    if use_grad_clip and log_gradient_stats:
        final_stats = gradient_monitor.get_stats()
        summary = (
            f"\n{'=' * 60}\n"
            f"TRAINING COMPLETED - GRADIENT CLIPPING SUMMARY\n"
            f"{'=' * 60}\n"
            f"Total Epochs: {opt.OPTIM.NUM_EPOCHS}\n"
            f"Gradient Clip Norm: {grad_clip_norm}\n"
            f"Final Gradient Statistics:\n"
            f"  Mean: {final_stats['mean']:.6f}\n"
            f"  Std:  {final_stats['std']:.6f}\n"
            f"  Max:  {final_stats['max']:.6f}\n"
            f"  Min:  {final_stats['min']:.6f}\n"
            f"  Overall Clipped Ratio: {final_stats['clipped_ratio']:.2%}\n"
            f"{'=' * 60}\n"
        )
        
        with open(gradient_log_path, 'a') as f:
            f.write(summary)
        
        print(summary)

