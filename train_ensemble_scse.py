from pickletools import optimize
from model.Module.Resnet34 import *
from data_prepare.dataloader import * 
from utils.loss_function import *
import torch.nn.functional as  F
image_dir='data/train'
mask_dir='data/masks'


#config
batch_size=16
shuffle=True #true if you want to shuffle data
num_workers=4 # use all core of cpu to train
pin_memory=True

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

#choose device to train GPU and CPU 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#call model and change mode to train
model=Res34Unetv3().to(device)
#config for train
max_lr=1e-04
optimizer = torch.optim.Adam(model.parameters(), max_lr,) #momentum=0.9, #optimizer

#Train function
def train(model,train_loader):
    running_loss=0.0
    predicts=[]
    data_size=train_dataset.__len__()
    model.train()
    for image,gt in train_loader:
        image,mask=image.to(device),gt.to(device)
        optimizer.auto_grad()
        with torch.set_grad_enabled(True):
            output=model(image)
            #Use symetric_lovasz to predict negative case for mask
            loss=symmetric_lovasz(output.squeeze(1),mask.squeeze(1))
            # back propagation
            loss.backward()
            #update learning rate
            optimizer.step()
        running_loss += loss.item() * image.size(0)
        # mb.child.comment = 'loss: {}'.format(loss.item())
        epoch_loss = running_loss / data_size
        return epoch_loss

#test function
def test(test_loader, model):
    running_loss = 0.0
    predicts = []
    truths = []
    model.eval()
    for inputs, masks in test_loader:
        inputs, masks = inputs.to(device), masks.to(device)
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            #loss = lovasz_hinge(outputs.squeeze(1), masks.squeeze(1))
            loss=symmetric_lovasz(outputs.squeeze(1), masks.squeeze(1))

        predicts.append(F.sigmoid(outputs).detach().cpu().numpy())
        truths.append(masks.detach().cpu().numpy())
        running_loss += loss.item() * inputs.size(0)

    predicts = np.concatenate(predicts).squeeze()
    truths = np.concatenate(truths).squeeze()
    precision, _, _ = do_kaggle_metric(predicts, truths, 0.5)
    precision = precision.mean()
    epoch_loss = running_loss / test_dataset.__len__()
    return epoch_loss, precision

num_snapshot=0
scheduler_step =20//4
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, scheduler_step,max_lr)

for epoch in range(100):
    train_loss = train(train_loader, model)
    val_loss, accuracy = test(test_loader, model)
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
        



