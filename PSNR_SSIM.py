import argparse
import cv2
import numpy as np
from os import path as osp
import os
from tqdm import tqdm

from basicsr.metrics import calculate_psnr, calculate_ssim
from basicsr.utils import scandir
from basicsr.utils import bgr2ycbcr


def get_image_files(folder_path):
    valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
    image_files = []
    
    for img_path in scandir(folder_path, recursive=True, full_path=True):
        ext = osp.splitext(img_path)[1].lower()
        if ext in valid_extensions:
            image_files.append(img_path)
    
    return sorted(image_files)


def main(args):
    """Calculate PSNR and SSIM for images.
    """
    psnr_all = []
    ssim_all = []
    
    img_list_gt = get_image_files(args.gt)
    img_list_restored = get_image_files(args.restored)
    

    print(f"GT images found: {len(img_list_gt)}")
    print(f"Restored images found: {len(img_list_restored)}")
    

    if len(img_list_gt) != len(img_list_restored):
        print(f"WARNING: File count mismatch! GT: {len(img_list_gt)}, Restored: {len(img_list_restored)}")
        min_count = min(len(img_list_gt), len(img_list_restored))
        print(f"Will process only {min_count} files")
        img_list_gt = img_list_gt[:min_count]
        img_list_restored = img_list_restored[:min_count]


    os.makedirs('results', exist_ok=True)
    

    output_file = 'results/PSNR_SSIM.txt'
    with open(output_file, 'w', encoding='utf-8') as f:

        f.write(f"PSNR and SSIM Evaluation Results\n")
        f.write(f"GT Path: {args.gt}\n")
        f.write(f"Restored Path: {args.restored}\n")
        f.write(f"Test Y Channel: {args.test_y_channel}\n")
        f.write(f"Crop Border: {args.crop_border}\n")
        f.write(f"Total Images: {len(img_list_gt)}\n")
        f.write("="*80 + "\n\n")
        
        if args.test_y_channel:
            print('Testing Y channel.')
            f.write('Testing Y channel.\n\n')
        else:
            print('Testing RGB channels.')
            f.write('Testing RGB channels.\n\n')


        for i, img_path in enumerate(tqdm(img_list_gt, desc="Processing images", ncols=80)):
            basename, ext = osp.splitext(osp.basename(img_path))
            img_gt = cv2.imread(img_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.
            
            if args.suffix == '':
                img_path_restored = img_list_restored[i]
            else:
                img_path_restored = osp.join(args.restored, basename + args.suffix + ext)
            

            if not osp.exists(img_path_restored):
                print(f"WARNING: Restored file not found: {img_path_restored}")
                continue
                
            img_restored = cv2.imread(img_path_restored, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.

            if args.correct_mean_var:
                mean_l = []
                std_l = []
                for j in range(3):
                    mean_l.append(np.mean(img_gt[:, :, j]))
                    std_l.append(np.std(img_gt[:, :, j]))
                for j in range(3):
                    # correct twice
                    mean = np.mean(img_restored[:, :, j])
                    img_restored[:, :, j] = img_restored[:, :, j] - mean + mean_l[j]
                    std = np.std(img_restored[:, :, j])
                    img_restored[:, :, j] = img_restored[:, :, j] / std * std_l[j]

                    mean = np.mean(img_restored[:, :, j])
                    img_restored[:, :, j] = img_restored[:, :, j] - mean + mean_l[j]
                    std = np.std(img_restored[:, :, j])
                    img_restored[:, :, j] = img_restored[:, :, j] / std * std_l[j]

            if args.test_y_channel and img_gt.ndim == 3 and img_gt.shape[2] == 3:
                img_gt = bgr2ycbcr(img_gt, y_only=True)
                img_restored = bgr2ycbcr(img_restored, y_only=True)

            # calculate PSNR and SSIM
            psnr = calculate_psnr(img_gt * 255, img_restored * 255, crop_border=args.crop_border, input_order='HWC')
            ssim = calculate_ssim(img_gt * 255, img_restored * 255, crop_border=args.crop_border, input_order='HWC')
            

            result_line = f'{i+1:3d}: {basename:25}. \tPSNR: {psnr:.6f} dB, \tSSIM: {ssim:.6f}\n'
            f.write(result_line)
            
            psnr_all.append(psnr)
            ssim_all.append(ssim)
        

        avg_psnr = sum(psnr_all) / len(psnr_all)
        avg_ssim = sum(ssim_all) / len(ssim_all)
        

        f.write("\n" + "="*80 + "\n")
        f.write(f'Average: PSNR: {avg_psnr:.6f} dB, SSIM: {avg_ssim:.6f}\n')
        f.write(f'Total processed images: {len(psnr_all)}\n')
    

    print(f'\nEvaluation completed!')
    print(f'GT Path: {args.gt}')
    print(f'Restored Path: {args.restored}')
    print(f'Average: PSNR: {avg_psnr:.6f} dB, SSIM: {avg_ssim:.6f}')
    print(f'Detailed results saved to: {output_file}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt', type=str, default='Datasets/test/Rain100L/target', help='Path to gt (Ground-Truth)')
    parser.add_argument('--restored', type=str, default='results/Rain100L', help='Path to restored images')
    parser.add_argument('--crop_border', type=int, default=0, help='Crop border for each side')
    parser.add_argument('--suffix', type=str, default='', help='Suffix for restored images')
    parser.add_argument(
        '--test_y_channel',
        action='store_true',
        help='If True, test Y channel (In MatLab YCbCr format). If False, test RGB channels.')
    parser.add_argument('--correct_mean_var', action='store_true', help='Correct the mean and var of restored images.')
    args = parser.parse_args()
    main(args)