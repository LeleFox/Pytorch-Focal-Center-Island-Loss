import torch
import torch.utils.data as data
from utils import utils
from utils import compute_class_weights
import torch.optim as optim
from efficientnet_pytorch import EfficientNet
from torch.optim.lr_scheduler import CosineAnnealingLR
from losses import FocalLoss, CenterLoss, IslandLoss, CE_Center_Criterion, CE_Island_Criterion

device= ("cuda" if torch.cuda.is_available() else "cpu")

#load an efficientnet model
model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=10)
model = model.to(device, non_blocking=True) #?model to GPU

num_epochs = 10
eval_freq = 1
num_classes = 7

#define data loader
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)

#!compute class weights for Weighted Cross Entropy Loss
class_weights = compute_class_weights(train_loader, norm=False).to(device, non_blocking=True)

#! Choose and initialize the loss function criterion
loss_fn = 'CE_Center' #'CE_Center' #'CE_Island' #'CE' #'Focal'
if loss_fn == 'CE': #CrossEntropyLoss 
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights, reduction='mean')
elif loss_fn == 'Focal': #Focal Loss 
    criterion = FocalLoss(alpha=1, gamma=2, reduction='mean')
elif loss_fn == 'CE_Center': #CE Loss + Center Loss 
    CE_loss = torch.nn.CrossEntropyLoss(weight=class_weights, reduction='mean')
    Center_loss = CenterLoss(num_classes, feat_dim=1408) #feature dimension from your network
    lambda_center = 1e-2 #5e-4 #3e-3
    optimizer_centers = torch.optim.SGD(Center_loss.parameters(), lr=0.5)  # optimizer for centers 
    criterion = CE_Center_Criterion(CE_loss, Center_loss, lambda_center)
elif loss_fn == 'CE_Island': #CE Loss + Center Loss + Island Loss
    CE_loss = torch.nn.CrossEntropyLoss(weight=class_weights, reduction='mean')
    lambda_global = 1e-2 
    lambda_island = 10
    Island_loss = IslandLoss(num_classes, feat_dim=1408, lambda_island=lambda_island)
    optimizer_centers = torch.optim.SGD(Island_loss.parameters(), lr=0.5)  # optimizer for centers 
    criterion = CE_Island_Criterion(CE_loss, Island_loss, lambda_global)
    
optim_params = filter(lambda parameter: parameter.requires_grad, model.parameters())
optimizer = torch.optim.AdamW(optim_params,lr=5e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

def zero_grad():
    """Reset the gradient when gradient accumulation is finished."""
    optimizer.zero_grad(set_to_none=True)
    
    if loss_fn == 'CE_Center':
        optimizer_centers.zero_grad(set_to_none=True)
        
    if loss_fn == 'CE_Center' or args.train.loss_fn == 'CE_Island':
        optimizer_centers.zero_grad(set_to_none=True)


def step():
    """This method performs an optimization step and resets both the loss and the accuracy.
    """
    optimizer.step() #Perform the step with the optimizer
    scheduler.step() #Perform the step with the LR scheduler (for model parameter)

    #! If using center loss, perform the step with the center optimizer
    if loss_fn == 'CE_Center':
        #don't want the lambda_center to affect the gradients of the loss with respect to the centers, so rescale back the gradients
        for param in Center_loss.parameters():
            param.grad.data *= (1. / lambda_center) 
            optimizer_centers.step()
    elif loss_fn == 'CE_Island':
        #don't want the lambda_global to affect the gradients of the loss with respect to the centers, so rescale back the gradients
        for param in Island_loss.parameters():
            param.grad.data *= (1. / lambda_global) #lambda_island is not so relevant
            optimizer_centers.step()
                        
    # Reset loss and accuracy tracking
    accuracy.reset()
    loss.reset()

#!accuracy and loss track the accuracy and the training loss
accuracy = utils.Accuracy(topk=(1, 5))
loss = utils.LossMeter()

model.train(True) #? set the model to training mode
model.zero_grad() #?clear any existing gradient

for epoch in range(1, num_epochs+1):
    #!move data,labels to gpu
    label = label.to(device, non_blocking=True) #?labels to GPU
    data.to(device, non_blocking=True) #? data to GPU
    
    logits, features = model.forward(data)
    
    if isinstance(criterion, (CE_Center_Criterion, CE_Island_Criterion)):
            loss = criterion(logits, label, features)
    else:
        loss = criterion(logits, label)

    loss.val.backward(retain_graph=False)
    accuracy.update(logits, label)

    model.step() 
    model.zero_grad() 
    
