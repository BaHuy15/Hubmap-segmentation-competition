import torch
import numpy as np
from utils.loss_function import *
import torch.nn.functional as nn
from data_prepare.dataloader import * 
from model.Module.pranet import *

image_dir='data/train'
mask_dir='data/masks'
#call train/test dataset
train_dataset=Hubmapdataset(image_dir,mask_dir,mode='train')
test_dataset=Hubmapdataset(image_dir,mask_dir,mode='test')
#Create dataloader for training
train_loader=DataLoader(dataset=train_dataset,
                                  batch_size=16,
                                  shuffle=True,
                                  num_workers=4,
                                  pin_memory=True)
#Create dataloader for test
test_loader=DataLoader(dataset=test_dataset,
                                  batch_size=16,
                                  shuffle=False,
                                  num_workers=4,
                                  pin_memory=True)
model=PraNet()
device = torch.device('cuda' if True else 'cpu')
model.to(device)

best_acc=0
num_snapshot=0
scheduler_step = 20 // 5
max_lr=1e-04
# min_lr=2e-04
optimizer = torch.optim.Adam(model.parameters(), max_lr,) #momentum=0.9,
                            #weight_decay=1e-4
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, scheduler_step,max_lr)

def test_pranet(test_loader, model):
    running_loss = 0.0
    predicts = []
    truths = []
    model.eval()
    for inputs, masks in test_loader:
        inputs, masks = inputs.to(device), masks.to(device)
        with torch.set_grad_enabled(False):
            lateral_map_5, lateral_map_4, lateral_map_3, lateral_map_2= model(inputs)
            #loss = lovasz_hinge(outputs.squeeze(1), masks.squeeze(1))
            loss5=symmetric_lovasz(lateral_map_5.squeeze(1), masks.squeeze(1))
            loss4=symmetric_lovasz(lateral_map_4.squeeze(1), masks.squeeze(1))
            loss3=symmetric_lovasz(lateral_map_3.squeeze(1), masks.squeeze(1))
            loss2=symmetric_lovasz(lateral_map_2.squeeze(1), masks.squeeze(1))
            loss=loss5+loss4+loss3+loss2
            outputs=lateral_map_2
        predicts.append(F.sigmoid(outputs).detach().cpu().numpy())
        truths.append(masks.detach().cpu().numpy())
        running_loss += loss.item() * inputs.size(0)

    predicts = np.concatenate(predicts).squeeze()
    truths = np.concatenate(truths).squeeze()
    precision, _, _ = do_kaggle_metric(predicts, truths, 0.5)
    precision = precision.mean()
    epoch_loss = running_loss / test_dataset.__len__()
    return epoch_loss, precision


def train_pranet(train_loader, model):
    running_loss = 0.0
    data_size = train_dataset.__len__()
    model.train()
    # for inputs, masks, labels in progress_bar(train_loader, parent=mb):
    for inputs, masks in train_loader:
        inputs, masks = inputs.to(device), masks.to(device)
        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            lateral_map_5, lateral_map_4, lateral_map_3, lateral_map_2= model(inputs)
            #loss = lovasz_hinge(outputs.squeeze(1), masks.squeeze(1))
            loss5=symmetric_lovasz(lateral_map_5.squeeze(1), masks.squeeze(1))
            loss4=symmetric_lovasz(lateral_map_4.squeeze(1), masks.squeeze(1))
            loss3=symmetric_lovasz(lateral_map_3.squeeze(1), masks.squeeze(1))
            loss2=symmetric_lovasz(lateral_map_2.squeeze(1), masks.squeeze(1))
            loss=loss5+loss4+loss3+loss2
            #https://discuss.pytorch.org/t/what-does-the-backward-function-do/9944/2
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        # mb.child.comment = 'loss: {}'.format(loss.item())
    epoch_loss = running_loss / data_size
    return epoch_loss 

    
for epoch in range(50):
    train_loss = train_pranet(train_loader, model)
    val_loss, accuracy = test_pranet(test_loader, model)
    lr_scheduler.step()

    if accuracy > best_acc:
        best_acc = accuracy
        best_param = model.state_dict()

    if (epoch + 1) % scheduler_step == 0:
#         torch.save(best_param, path)
#         #optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9,
#                                     weight_decay=1e-4)
        optimizer = torch.optim.Adam(model.parameters(), max_lr)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, scheduler_step, max_lr)
        num_snapshot += 1
        best_acc = 0

    # mb.write('epoch: {} train_loss: {:.3f} val_loss: {:.3f} val_accuracy: {:.3f}'.format(epoch + 1, train_loss,
    #                                                                                     val_loss, accuracy))
    print('epoch: {} train_loss: {:.3f} val_loss: {:.3f} val_accuracy: {:.3f}'.format(epoch + 1, train_loss,
                                                                                      val_loss, accuracy))