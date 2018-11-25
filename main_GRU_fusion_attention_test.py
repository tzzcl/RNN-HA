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
from dataset import *

# ======================= hyper-parameters ===============================

word_embedding_dim = 512
hidden_dim = 1024
num_layers = 1
my_batch_size = 64
num_model_class = 228
num_vin_class = 10309

def remove_last(model):
    list_layer = []
    final_layer = 15
    for i in range(final_layer):
        list_layer.append(model[i])
    new_model = nn.Sequential(*list_layer)
    print(new_model)
    return new_model
class Fusion_Net(nn.Module):
    def __init__(self):
        super(Fusion_Net, self).__init__()
        model = load_vgg_m_1024.load_vgg_m_1024(preTrained=False)
        new_state_dict = {}
        a = torch.load('./checkpoint.pth.tar')['state_dict']
        for k, v in a.items():
            name = k[7:]
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
        model = remove_last(model)
        self.vgg_m = model
        self.first_gap = nn.AvgPool2d(kernel_size=(6,6),stride = (6,6))
        self.second_gap = nn.AvgPool2d(kernel_size=(6,6),stride = (6,6))
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
        self.activation = nn.ReLU()
        init.kaiming_normal(self.W4.weight)
    def forward(self, im, h0):
        im_feat = self.vgg_m.forward(im)
        first_im_feat = self.first_gap(im_feat)
        first_im_feat = first_im_feat.view(-1,word_embedding_dim)
        first_im_feat = torch.unsqueeze(first_im_feat,0)
        first_im_feat = torch.cat((first_im_feat,first_im_feat),0)
        output, hn = self.rnn(first_im_feat, h0)
        o_model = output[0, :, :]

        o_model_input = self.W3(o_model)
        o_model_input = self.relu(o_model_input)
        o_model_input = self.W4(o_model_input)
        o_model_input_expand = o_model_input.unsqueeze(2).unsqueeze(2).expand_as(im_feat)

        second_im_score = torch.mul(im_feat,o_model_input_expand)
        second_im_score = torch.sum(second_im_score,dim=1,keepdim = True)
        second_im_score = self.attention_activation(second_im_score)

        second_im_score = second_im_score + self.epilson
        second_im_score_total = torch.sum(second_im_score.view(-1,36),dim=1,keepdim=True)
        second_im_score_normalized = torch.div(second_im_score,second_im_score_total.unsqueeze(2).unsqueeze(2).expand_as(second_im_score))

        second_im_feat = torch.mul(im_feat,second_im_score_normalized.expand_as(im_feat))
        second_im_feat = self.second_gap(second_im_feat)
        second_im_feat = second_im_feat.view(-1,word_embedding_dim).unsqueeze(0)
        output, hn = self.rnn(second_im_feat, hn)
        o_vin = output[0, :, :]

        return o_model,o_vin
model = Fusion_Net()
model.load_state_dict(torch.load('model/model_fc_epoch_20'))
model.cuda()
print('Model is built ...')


# ======================= preparing data ==================================
print('Loading data ...')
print('Data is loaded ...')
now_transform = transforms.Compose([
    Self_Scale(224),
    transforms.ToTensor(),
    Invert_Normalize(),
])
dataset = sio.loadmat('test_list_2400.mat')
name = dataset['images']['name'][0][0][0]
length = name.shape[0]
label = dataset['images']['class'][0][0][0]
print(length)
print('Testing data is ready ...')

# ======================= define training functions =======================

print('Testing starts ...')

model.eval()
vin_feat_total = None
model_feat_total = None
attmap_total = None
te_batch = 0
for now_name in name:
    total_name = r'/home/zhangcl/VehicleID_V1.0/image/' + str(now_name[0])
    first_image = Image.open(total_name).convert('RGB')
    final_image = (now_transform(first_image))
    h0 = 0.0 * torch.randn(num_layers,1, hidden_dim)
    h0 = Variable(h0.cuda(),volatile=True)
    final_image = torch.unsqueeze(final_image,0)
    final_image = Variable(final_image.cuda(), volatile=True)
    rnn_data_k = final_image
    output_model, output_vin,att_map = model(rnn_data_k, h0)
    if vin_feat_total is None:
        vin_feat_total = output_vin.data.cpu()
        model_feat_total = output_model.data.cpu()
        attmap_total = att_map.data.cpu()
    else:
        vin_feat_total = torch.cat((vin_feat_total, output_vin.data.cpu()), dim=0)
        model_feat_total = torch.cat((model_feat_total, output_model.data.cpu()),dim=0)
        attmap_total = torch.cat((attmap_total,att_map.data.cpu()),dim = 0)
    te_batch += 1
    print('The {}-th testing batch is done ...'.format(te_batch))

vin_feat_total = vin_feat_total.numpy()
model_feat_total = model_feat_total.numpy()
attmap_total = attmap_total.numpy()
sio.savemat('feat_'+'fusion_attention_vehi_vggm_224_2400'+'.mat', dict(vin_feat_total=vin_feat_total, model_feat_total=model_feat_total,attmap_total=attmap_total,label=label))
print('Feature extraction is finished ...')
