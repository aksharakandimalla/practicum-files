import torch
model_ft = torch.load('/home/akandimalla/practicum/te_en_models/s4/run4/model_step_9000.pt')
ct = 0
for child in model_ft.children():
  ct += 1
  if ct < 7:
    for param in child.parameters():
        param.requires_grad = False
torch.save(model_ft, '/home/akandimalla/practicum/te_en_models/s4/run4/model_step_frozen_9000.pt')
