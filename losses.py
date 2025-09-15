import torch
import torch.nn as nn
import torch.nn.functional as F

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss

class EdgeLoss(nn.Module):
    def __init__(self):
        super(EdgeLoss, self).__init__()
        k = torch.Tensor([[.05, .25, .4, .25, .05]])
        self.kernel = torch.matmul(k.t(),k).unsqueeze(0).repeat(3,1,1,1)
        if torch.cuda.is_available():
            self.kernel = self.kernel.cuda()
        self.loss = CharbonnierLoss()

    def conv_gauss(self, img):
        n_channels, _, kw, kh = self.kernel.shape
        img = F.pad(img, (kw//2, kh//2, kw//2, kh//2), mode='replicate')
        return F.conv2d(img, self.kernel, groups=n_channels)

    def laplacian_kernel(self, current):
        filtered    = self.conv_gauss(current)    # filter
        down        = filtered[:,:,::2,::2]               # downsample
        new_filter  = torch.zeros_like(filtered)
        new_filter[:,:,::2,::2] = down*4                  # upsample
        filtered    = self.conv_gauss(new_filter) # filter
        diff = current - filtered
        return diff

    def forward(self, x, y):
        loss = self.loss(self.laplacian_kernel(x), self.laplacian_kernel(y))
        return loss

##########################################################################
# Multi-domain loss function 
##########################################################################


class WaveletLoss(nn.Module):
    """Wavelet domain loss - Wavelet Domain Loss"""
    def __init__(self):
        super(WaveletLoss, self).__init__()
        self.cri = nn.L1Loss()
        try:
            from pytorch_wavelets import DWTForward
            self.dwt = DWTForward(J=2, mode='zero', wave='haar')
            self.wavelets_available = True
        except ImportError:
            print("pytorch-wavelets not available, WaveletLoss will use identity mapping")
            self.wavelets_available = False

    def forward(self, x, y):
        if not self.wavelets_available:
            # If wavelet transform is not available, fall back to L1 loss
            return self.cri(x, y)
        
        try:
            # Try to move DWT to the correct device
            if hasattr(self.dwt, 'to'):
                self.dwt = self.dwt.to(x.device)
            
            x_yl, x_yh = self.dwt(x)
            y_yl, y_yh = self.dwt(y)
            
            # Calculate low-frequency coefficient loss
            loss_ll = self.cri(x_yl, y_yl)
            
            # Calculate high-frequency coefficient loss
            loss_hh = 0
            for i in range(len(x_yh)):
                for j in range(x_yh[i].shape[2]):  # 3 high-frequency sub-bands
                    loss_hh += self.cri(x_yh[i][:, :, j, :, :], y_yh[i][:, :, j, :, :])
            
            return loss_ll + loss_hh
            
        except Exception as e:
            # If wavelet transform fails, fall back to L1 loss
            print(f"Wavelet transform failed: {e}, using L1 loss instead")
            return self.cri(x, y)

class MultiDomainLoss(nn.Module):
    """
    Multi-domain loss function - supports different stage-based loss strategies, using a conservative wavelet weight configuration
    
    Stage strategy (based on FreqMamba frequency domain loss, total weight controlled at 0.08):
    - Stage1: WaveletStatisticalSpatialAttention → Wavelet weight 0.03 (auxiliary spatial enhancement)
    - Stage2: HybridDomainMamba+WaveletBranch → Wavelet weight 0.05 (core frequency domain modeling)
    - Stage3: ORSNet without frequency domain processing → No wavelet loss
    """
    def __init__(self, spatial_weight=1.0, edge_weight=0.05, wavelet_weight=None):
        super(MultiDomainLoss, self).__init__()
        
        # Basic loss function
        self.spatial_loss = CharbonnierLoss()
        self.edge_loss = EdgeLoss()
        self.wavelet_loss = WaveletLoss()
        
        # Loss weights
        self.spatial_weight = spatial_weight
        self.edge_weight = edge_weight
        
        # Stage-based wavelet weight (based on conservative configuration suggested by user)
        # wavelet_weight parameter is deprecated, using fixed Stage weight configuration
        self.stage_wavelet_weights = {
            "stage1": 0.0,    # 0.03 - WaveletStatisticalSpatialAttention auxiliary enhancement
            "stage2": 0.05,    # 0.05 - WaveletBranch core frequency domain modeling (matching FreqMamba)
            "stage3": 0.0      # 0.00 - ORSNet without frequency domain processing
        }

    def forward(self, pred, target, stage_type="stage2"):
        """
        Args:
            pred: Predicted image [B, C, H, W]  
            target: Target image [B, C, H, W]
            stage_type: "stage1", "stage2", "stage3"
        """
        losses = {}
        
        # All stages calculate spatial domain loss
        spatial_loss = self.spatial_loss(pred, target)
        edge_loss = self.edge_loss(pred, target)
        
        losses['spatial'] = spatial_loss * self.spatial_weight
        losses['edge'] = edge_loss * self.edge_weight
        
        # Differentiated wavelet domain loss strategy
        stage_wavelet_weight = self.stage_wavelet_weights.get(stage_type, 0.0)
        
        if stage_wavelet_weight > 0:
            wavelet_loss = self.wavelet_loss(pred, target)
            losses['wavelet'] = wavelet_loss * stage_wavelet_weight
        else:
            losses['wavelet'] = torch.tensor(0.0, device=pred.device)
        
        # Calculate total loss
        total_loss = sum(losses.values())
        
        return total_loss, losses

