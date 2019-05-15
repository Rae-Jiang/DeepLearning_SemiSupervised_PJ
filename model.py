import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
import copy

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        # Architecture
        self.features = copy.deepcopy(models.vgg11_bn(pretrained=False).features)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(3, 3))
        self.classifier = nn.Sequential(
            nn.Linear(in_features=4608, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=1000, bias=True))


        # Load pre-trained model
        self.load_weights('weights.pth')

    def load_weights(self, pretrained_model_path, cuda=True):
        # Load pretrained model
        pretrained_model = torch.load(f=pretrained_model_path, map_location="cuda" if cuda else "cpu")

        # Load pre-trained weights in current model
        with torch.no_grad():
            self.load_state_dict(pretrained_model, strict=True)

        # Debug loading
        print('Parameters found in pretrained model:')
        pretrained_layers = pretrained_model.keys()
        for l in pretrained_layers:
            print('\t' + l)
        print('')

        for name, module in self.state_dict().items():
            if name in pretrained_layers:
                assert torch.equal(pretrained_model[name].cpu(), module.cpu())
                print('{} have been loaded correctly in current model.'.format(name))
            else:
                raise ValueError("state_dict() keys do not match")

    def forward(self, x):
        # TODO
        x = self.features(x)
        x = self.avgpool(x)
        x=x.view(x.size(0),-1)
        x = self.classifier(x)
        return x
        