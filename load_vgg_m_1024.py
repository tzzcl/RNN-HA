import torch
import torch.nn as nn
import vgg_m_1024_torch
import torch.nn.init as init
final_layer = 15
import math
class LambdaBase(nn.Sequential):
    def __init__(self, fn, *args):
        super(LambdaBase, self).__init__(*args)
        self.lambda_func = fn

    def forward_prepare(self, input):
        output = []
        for module in self._modules.values():
            output.append(module(input))
        return output if output else input
class Lambda(LambdaBase):
    def forward(self, input):
        return self.lambda_func(self.forward_prepare(input))


def _initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            n = m.weight.size(1)
            init.kaiming_normal(m.weight)
            m.bias.data.zero_()
def load_vgg_m_1024(name=None,preTrained = True):
    model = vgg_m_1024_torch.vgg_m_1024_torch
    if preTrained:
        model.load_state_dict(name)
    #new_model = model
    list_layer = []
    for i in range(final_layer):
        list_layer.append(model[i])
    gap_layer = nn.AvgPool2d(kernel_size=(6,6),stride = (6,6))
    list_layer.append(gap_layer)
    view_layer = Lambda(lambda x: x.view(x.size(0), -1))
    list_layer.append(view_layer)
    last_layer = nn.Linear(512,10309)
    _initialize_weights(last_layer)
    list_layer.append(last_layer)
    new_model = nn.Sequential(*list_layer)
    return new_model

if __name__ == '__main__':
    model = load_vgg_m_1024(torch.load('vgg_m_1024_torch.pth'))
    print(model)