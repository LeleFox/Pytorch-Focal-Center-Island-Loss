#v1.0
import torch
from torch import nn
import torch.nn.functional as F

class CE_Center_Criterion(nn.Module):
    def __init__(self, ce_loss=None, center_loss=None, lambda_center=1e-2):
        super(CE_Center_Criterion, self).__init__()
        self.ce_loss = ce_loss
        self.center_loss = center_loss
        self.lambda_center = lambda_center

    def forward(self, logits, labels, features):
        ce_loss_value = self.ce_loss(logits, labels)
        center_loss_value = self.center_loss(features, labels)
        total_loss = ce_loss_value + self.lambda_center * center_loss_value
        return total_loss
    
class CenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        if torch.cuda.is_available():
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))
    
    def forward(self, x, labels):
        """
        Args:
            x: feature matrix (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        
        # Compute the distance between features and centers
        #?||x - c||^2 = ||x||^2 + ||c||^2 - (2 * x * c)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(mat1=x, mat2=self.centers.t(), beta=1, alpha=-2) #? add and mult (-2 * x * c)
        
        classes = torch.arange(self.num_classes).long()
        if torch.cuda.is_available(): 
            classes = classes.cuda()
            
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size #? sum over batch_size and compute mean
        return loss