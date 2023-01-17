

from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.losses import DiceLoss, DiceCELoss
import os
import torch
from preporcess import prepare
from utilities import train
print("1")

os.environ['KMP_DUPLICATE_LIB_OK']='True'
data_dir = 'H:\TDSI\cc359_preprocessed\Different'
model_dir = 'H:\TDSI\cc359_preprocessed\modele\Modelenotsame'
data_in = prepare(data_dir, cache=True)

print("2")

device = torch.device("cuda:0")
model = UNet(
    dimensions=3,
    in_channels=1,
    out_channels=2,
    channels=(16, 32, 64, 128, 256), 
    strides=(2, 2, 2, 2),
    num_res_units=2,
    norm=Norm.BATCH,
).to(device)

print("3")

#loss_function = DiceCELoss(to_onehot_y=True, sigmoid=True, squared_pred=True, ce_weight=calculate_weights(1792651250,2510860).to(device))
loss_function = DiceLoss(to_onehot_y=True, sigmoid=True, squared_pred=True)
optimizer = torch.optim.Adam(model.parameters(), 1e-5, weight_decay=1e-5, amsgrad=True)

print("4")
if __name__ == '__main__':
    train(model, data_in, loss_function, optimizer, 400, model_dir)

print("5")