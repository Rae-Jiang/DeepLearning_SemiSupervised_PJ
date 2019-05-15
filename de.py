import torch
import copy
save_path = '/home/rj1407/pytorch-cpu/final/weights.pth'
model = torch.load('190422_vgg_alladj.pt')

best_model_wts = copy.deepcopy(model.state_dict())
with open(save_path, 'wb') as f:
    torch.save(best_model_wts, f)