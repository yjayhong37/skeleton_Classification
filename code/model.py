import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ConvLSTM(nn.Module):
    def __init__(self, params):
        super(ConvLSTM, self).__init__()
        
        # Here, input_channel = 1 as there are only 3D coordinates of 25 joints
        # So, each frame is provided as input in the following format: (1, 25, 3)
        # Each segment has N frames and the batch size is B_S, so input size is: (B_S, N, 1, 25, 3)
        self.bcc = params.bcc # Base convolution channels (changeable parameter for the model)
        self.conv1 = nn.Conv2d(1, self.bcc, kernel_size=(2, 1))
        self.conv2 = nn.Conv2d(self.bcc, 2*self.bcc, kernel_size=(2, 1))
        self.conv3 = nn.Conv2d(2*self.bcc, 4*self.bcc, kernel_size=(2, 1))
        self.conv4 = nn.Conv2d(4*self.bcc, 8*self.bcc, kernel_size=(2, 1))
        #self.conv5 = nn.Conv2d(8*self.bcc, 16*self.bcc, kernel_size=(2, 1))
        
        self.relu = nn.ReLU()

        self._to_linear, self._to_lstm = None, None
        x = torch.randn(params.BATCH_SIZE, params.seg_size, params.num_channels, params.num_joints, params.num_coord)
        self.convs(x)
        
        self.lstm = nn.LSTM(input_size=self._to_lstm, hidden_size=1024, num_layers=4, batch_first=True)
        
        self.fc1 = nn.Linear(1024,512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, self.bcc*1)
        self.fc4 = nn.Linear(self.bcc*1, params.num_classes)
        
        
    def convs(self, x):
        batch_size, timesteps, c, h, w = x.size()
        x = x.view(batch_size*timesteps, c, h, w)
        x = self.conv1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.relu(x)
        
        x = self.conv4(x)
        x = self.relu(x)

        # x = self.conv5(x)
        # x = self.relu(x)

        if self._to_linear is None:
            # Only used for a random first pass: done to know what the output of the convnet is
            self._to_linear = int(x[0].shape[0]*x[0].shape[1]*x[0].shape[2])
            r_in = x.view(batch_size, timesteps, -1)
            self._to_lstm = r_in.shape[2]

        return x
    
    
    def forward(self, x):
        batch_size, timesteps, c, h, w = x.size()
        cnn_out = self.convs(x)
        r_in = cnn_out.view(batch_size, timesteps, -1)
        self.lstm.flatten_parameters()
        r_out, (h_n, h_c) = self.lstm(r_in)
        r_out = self.fc1(r_out[:, -1, :])
        x = self.relu(x)
        r_out = self.fc2(r_out)
        x = self.relu(x)
        r_out = self.fc3(r_out)
        x = self.relu(x)
        r_out = self.fc4(r_out)
        x = self.relu(x)
        r_out = F.log_softmax(r_out, dim=1)
        
        return r_out

# x = torch.randn(512,12,1,24,3)
# y = torch.randn(1)
# from easydict import EasyDict as edict
# p = edict({
#     'mode': 'train', 
#     'model_path': '/home/youngjoon/DEV/ntu-skeleton/code/saved_models/', 
#     'kp_shape': [24, 3], 
#     'val_pct': 0.2, 
#     'seg_size': 50, 
#     'data_path': '/home/youngjoon/Desktop/Dataset/humanact12/', 
#     'BATCH_SIZE': 512, 
#     'temporal_aug_k': 3, 
#     'k_fold': 1, 
#     'n_epochs': 10, 
#     'num_classes': 12, 
#     'bcc': 32, 
#     'num_channels': 1, 
#     'num_joints': 24, 
#     'num_coord': 3
# })

# model = ConvLSTM(p)
# print(model(x))
# print(model(x).shape)