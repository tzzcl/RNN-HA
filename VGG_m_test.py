import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torch.nn.init as init
import scipy.io as sio
from torch.autograd import Variable
import torch.utils.data as data_utils
import load_vgg_m_1024
import torchvision.transforms as transforms
from PIL import Image
import gc
def remove_last(model):
    list_layer = []
    final_layer = 16
    for i in range(final_layer):
        list_layer.append(model[i])
    new_model = nn.Sequential(*list_layer)
    print(new_model)
    return new_model
model = load_vgg_m_1024.load_vgg_m_1024(preTrained=False)
#model = torch.nn.DataParallel(model)
#print(model)
new_state_dict = {}
a = torch.load('checkpoint.pth.tar')['state_dict']
for k,v in a.items():
    name = k[7:]
    new_state_dict[name] = v

model.load_state_dict(new_state_dict)
model = remove_last(model)
model.cuda()
print 'Model is loaded ...'

# optimizer = optim.SGD(model.parameters(), lr=0.001)

# ======================= preparing data ==================================
print 'Loading testing data ...'
dataset = sio.loadmat('test_list_2400.mat')
name = dataset['images']['name'][0][0][0]
length = name.shape[0]
label = dataset['images']['class'][0][0][0]
print 'Data is loaded ...'

test_label = torch.from_numpy(label.astype('int64'))

print 'Testing data is ready ...'

# ======================= define testing functions =======================

print 'Start feature extraction ...'


# testing
model.eval()
te_batch = 1
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
class Invert_Normalize(object):
    def __init__(self, mean = None):
        self.mean = [123.68,116.78,103.94]
        self.s = 1. / 255
    def __call__(self, tensor):
        # TODO: make efficient
        for t, m in zip(tensor, self.mean):
            t.div(self.s).sub_(m)
        return tensor
now_transform = transforms.Compose([
    Self_Scale(224),
    # transforms.CenterCrop(224),
    transforms.ToTensor(),
    # normalize,
    Invert_Normalize(),
])
feat_total = None
temp_input = torch.Tensor(1,3,224,224).cuda()
for now_name in name:
    total_name = r'/home/zhangcl/VehicleID_V1.0/image/' + str(now_name[0])
    first_image = Image.open(total_name).convert('RGB')
    final_image = (now_transform(first_image))
    temp_input[0] = final_image
    final_image = Variable(temp_input,volatile = True)
    temp = model.forward(final_image).data.cpu()
    # print feat_model
    if feat_total is None:
        feat_total = temp
    else:
        feat_total = torch.cat((feat_total,temp),dim = 0)
    #if te_batch == 1:
    #    Feat_model = feat_model
    #    Feat_vin = feat_vin
    #else:
    #    Feat_model = torch.cat((Feat_model, feat_model), 0)
    #    Feat_vin = torch.cat((Feat_vin, feat_vin), 0)

    print('The {}-th testing batch is done ...'.format(te_batch))
    te_batch +=1
feat_total = feat_total.numpy()
sio.savemat('feat_'+'vgg_m_fine_tune'+'.mat', dict(feat_total=feat_total, label=label))
print 'Feature extraction is finished ...'
