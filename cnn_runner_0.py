from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import config as cfg
import models
import utils
from utils import RunningMean, use_gpu
from mistt import FurnitureDataset, preprocess, preprocess_with_augmentation, normalize_05
from tensorboardX import SummaryWriter
BATCH_SIZE = 1
IMAGE_SIZE = 299
import numpy as np

def get_model():
    print('[+] loading model... ', end='', flush=True)
    model = models.inceptionv4_finetune(cfg.CLASSES_num)
    #用inceptionv4分3类
    if use_gpu:
        model.cuda()
    print('done')
    return model
# inceptionv4_finetune = partial(FinetunePretrainedmodels,
     #                          net_cls=pretrainedmodels.inceptionv4,
     #                          net_kwards={'pretrained': 'imagenet+background', 'num_classes': 1001})
# partial函数用于携带部分参数生成一个新函数
def train():
    train_dataset = FurnitureDataset('train', transform=preprocess_with_augmentation(normalize_05, IMAGE_SIZE))
    val_dataset = FurnitureDataset('val', transform=preprocess(normalize_05, IMAGE_SIZE))
    training_data_loader = DataLoader(dataset=train_dataset, num_workers=8,
                                      batch_size=BATCH_SIZE,
                                      shuffle=True)
    validation_data_loader = DataLoader(dataset=val_dataset, num_workers=1,
                                        batch_size=BATCH_SIZE,
                                        shuffle=False)

    model = get_model()

    criterion = nn.CrossEntropyLoss().cuda()
    writer = SummaryWriter(log_dir='logs')
    nb_learnable_params = sum(p.numel() for p in model.fresh_params())
    print('nb learnable params: {}'.format(nb_learnable_params))

    lx, px = utils.predict(model, validation_data_loader)
    min_loss =criterion(Variable(px), Variable(lx)).data[0]

    lr = 0
    patience = 0
    for epoch in range(30):
        print('epoch {}'.format(epoch))
        if epoch == 1:
            lr = 0.00005
            print('set lr={}'.format(lr))
        if patience == 2:
            patience = 0
            model.load_state_dict(torch.load('inception4_lyc.pth'))
            lr = lr / 5
            print('set lr={}'.format(lr))
        if epoch == 0:
            lr = 0.001
            print('set lr={}'.format(lr))
            optimizer = torch.optim.Adam(model.fresh_params(), lr=lr)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0001)

        running_loss = RunningMean()
        running_score = RunningMean()
        step  =0
        all_step = int(len(training_data_loader) /BATCH_SIZE)
        model.train()
        pbar = tqdm(training_data_loader, total=len(training_data_loader))
        for inputs, labels in pbar:
            step += 1
            batch_size = inputs.size(0)

            inputs = Variable(inputs)
            labels = Variable(labels)
            if use_gpu:
                inputs = inputs.cuda()
                labels = labels.cuda()

            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs.data, dim=1)

            loss = criterion(outputs, labels)
            running_loss.update(loss.data[0], 1)
            running_score.update(torch.sum(preds != labels.data), batch_size)
            writer.add_scalar('train_loss', loss, epoch*all_step+step)
            loss.backward()
            optimizer.step()

            pbar.set_description('running_loss.value{:.5f} running_score.value{:.3f}'
                                 .format(running_loss.value,running_score.value))
        print('epoch {}: running_loss.value{:.5f} running_score.value{:.3f}'
              .format(epoch, running_loss.value,running_score.value))

        lx, px = utils.predict(model, validation_data_loader)
        log_loss = criterion(Variable(px), Variable(lx))
        log_loss = log_loss.data[0]
        _, preds = torch.max(px, dim=1)
        accuracy = torch.mean((preds != lx).float())
        print('val: log_loss{:.5f} accuracy{:.3f}'.format(log_loss, accuracy))

        if log_loss < min_loss:
            torch.save(model.state_dict(), 'inception4_lyc.pth')
            print('val score improved from {:.5f} to {:.5f}. Saved!'.format(min_loss, log_loss))
            min_loss = log_loss
            patience = 0
        else:
            patience += 1

        writer.add_scalar('val_loss', log_loss, epoch + 1)
        writer.add_scalar('val_acc', accuracy, epoch+1)
    writer.close()

if __name__ == "__main__":
    train()
