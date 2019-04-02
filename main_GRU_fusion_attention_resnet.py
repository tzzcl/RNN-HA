import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torch.nn.init as init
import scipy.io as sio
from torch.autograd import Variable
import load_vgg_m_1024
import torch.utils.data as data_utils
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import torchvision.transforms as transforms
import torchvision.models as models
# ======================= hyper-parameters ===============================

word_embedding_dim = 2048
hidden_dim = 2048
num_layers = 1
my_batch_size = 8
num_model_class = 228
num_vin_class = 10309

def remove_last(model):
    new_classifer = nn.Sequential(*list(model.children())[:-2])
    return new_classifer
# ======================= define model ===================================
class CarDataset(Dataset):
    """Car dataset."""

    def __init__(self, mat_file, root_dir, transform=None):

        self.mat_file = sio.loadmat(mat_file)
        self.name = self.mat_file['images']['name'][0][0][0]
        self.model = self.mat_file['images']['model'][0][0][0]
        self.label = self.mat_file['images']['class'][0][0][0]
        self.root_dir = root_dir
        self.transform = transform
    def __len__(self):
        return (self.label.shape[0])

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.name[idx][0])
        image = Image.open(img_name).convert('RGB')
        model = (self.model[idx] - 1).astype('int64')
        label = (self.label[idx] - 1).astype('int64')
        sample = {'image': image, 'model':model,'label':label}

        if self.transform:
            sample['image'] = self.transform(sample['image'])

        return sample
class Self_Scale(object):
    """Rescales the input PIL.Image to the given 'size'.
    'size' will be the size of the smaller edge.
    For example, if height > width, then image will be
    rescaled to (size * height / width, size)
    size: size of the smaller edge
    interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        return img.resize((self.size, self.size), self.interpolation)
class Fusion_Net(nn.Module):
    def __init__(self):
        super(Fusion_Net, self).__init__()
        model = models.resnet50(pretrained=False)
        model.fc = nn.Linear(2048, 10309)
        new_state_dict = {}
        a = torch.load('checkpoint_resnet.pth.tar')['state_dict']
        for k, v in a.items():
            name = k[7:]
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
        model = remove_last(model)
        self.resnet = model
        self.first_gap = nn.AvgPool2d(kernel_size=(7,7),stride = (7,7))
        self.second_gap = nn.AvgPool2d(kernel_size=(7,7),stride = (7,7))
        self.relu = nn.ReLU()
        self.attention_activation = nn.Softplus()
        self.epilson = 1e-1
        self.rnn = nn.GRU(word_embedding_dim, hidden_dim, num_layers, bidirectional=False)
        self.W1 = nn.Linear(hidden_dim, num_model_class)
        init.kaiming_normal(self.W1.weight)
        self.W2 = nn.Linear(hidden_dim, num_vin_class)
        init.kaiming_normal(self.W2.weight)
        self.W3 = nn.Linear(hidden_dim, hidden_dim)
        init.kaiming_normal(self.W3.weight)
        self.W4 = nn.Linear(hidden_dim,word_embedding_dim)
        init.kaiming_normal(self.W4.weight)
    def forward(self, im, h0):
        im_feat = self.resnet.forward(im)
        first_im_feat = self.first_gap(im_feat)
        first_im_feat = first_im_feat.view(-1,word_embedding_dim)
        first_im_feat = torch.unsqueeze(first_im_feat,0)
        output, hn = self.rnn(first_im_feat, h0)
        o_model = output[0, :, :]

        o_model_input = self.W3(o_model)
        o_model_input = self.relu(o_model)
        o_model_input = self.W4(o_model_input)
        o_model_input_expand = o_model_input.unsqueeze(2).unsqueeze(2).expand_as(im_feat)

        second_im_score = torch.mul(im_feat,o_model_input_expand)
        second_im_score = torch.sum(second_im_score,dim=1,keepdim = True)
        second_im_score = self.attention_activation(second_im_score)

        second_im_score = second_im_score + self.epilson
        second_im_score_total = torch.sum(second_im_score.view(-1,49),dim=1,keepdim=True)
        second_im_score_normalized = torch.div(second_im_score,second_im_score_total.unsqueeze(2).unsqueeze(2).expand_as(second_im_score))

        second_im_feat = torch.mul(im_feat,second_im_score_normalized.expand_as(im_feat))
        second_im_feat = self.second_gap(second_im_feat)
        second_im_feat = second_im_feat.view(-1,word_embedding_dim).unsqueeze(0)
        output, hn = self.rnn(second_im_feat, hn)

        o_vin = output[0, :, :]

        pred_model = self.W1(o_model)
        pred_vin = self.W2(o_vin)
        return F.log_softmax(pred_model), F.log_softmax(pred_vin)


model = Fusion_Net()
model.cuda()
print 'Model is built ...'

optimizer = optim.RMSprop(model.parameters(), lr=0.001)

# ======================= preparing data ==================================
print 'Loading data ...'
print 'Data is loaded ...'
now_transform = transforms.Compose([
    Self_Scale(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize,
])

train_loader = DataLoader(CarDataset(mat_file='train_list_modify_withModel_welldone.mat',root_dir='/home/zhangcl/VehicleID_V1.0/image',transform=now_transform),batch_size=my_batch_size,shuffle=True,num_workers=1)
print 'Training data is ready ...'

# ======================= define training functions =======================

print 'Training starts ...'
for epoch in range(1, 20):

    # training
    train_loss = 0
    correct_model = 0
    correct_vin = 0

    model.train()
    tr_batch = 1
    for sample in train_loader:
        h0 = 0.0 * torch.randn(num_layers,len(sample['image']), hidden_dim)
        h0 = Variable(h0.cuda())
        vin_label_k = sample['label']
        model_label_k = sample['model']
        rnn_data_k = sample['image']
        rnn_data_k, vin_label_k, model_label_k = rnn_data_k.cuda(), vin_label_k.cuda(), model_label_k.cuda()
        rnn_data_k, vin_label_k, model_label_k = Variable(rnn_data_k), Variable(vin_label_k), Variable(model_label_k)

        optimizer.zero_grad()
        output_model, output_vin = model(rnn_data_k, h0)
        loss_model = F.nll_loss(output_model, model_label_k)
        loss_vin = F.nll_loss(output_vin, vin_label_k)
        pred_model = output_model.data.max(1)[1]  # get the index of the max log-probability
        correct_model += 1. * pred_model.eq(model_label_k.data).cpu().sum()/my_batch_size
        pred_vin = output_vin.data.max(1)[1]  # get the index of the max log-probability
        correct_vin += 1. * pred_vin.eq(vin_label_k.data).cpu().sum() / my_batch_size

        loss = loss_model + loss_vin
        loss.backward()
        optimizer.step()

        train_loss += loss.data[0]/my_batch_size

        print('The {}-th training batch of the {}-th epoch: training loss: {:.4f}, train accuracy: model = {:.3f}%, '
              'vin = {:.3f}%\n'.format(
            tr_batch, epoch, train_loss/tr_batch, 100. * correct_model/tr_batch, 100. * correct_vin/tr_batch))
        tr_batch = tr_batch + 1

    log = open('./log/train_resnet.txt', 'a+')
    log.write("The " + str(epoch) + "-th epoch: model accuracy is " + str(100. * correct_model/(tr_batch-1))
              + ", vin accuracy is " + str(100. * correct_vin/tr_batch) +"\n")

    # saving model
    if epoch % 2 == 0:
        saveName = './model/model_attention_epoch_resnet_672_' + str(epoch)
        torch.save(model.state_dict(), saveName)

