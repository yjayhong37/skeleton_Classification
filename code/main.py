import argparse
from easydict import EasyDict as edict

import torch
import torch.nn as nn
from torch import optim
from torch.storage import HAS_NUMPY
from torch.utils.data import DataLoader
from torchvision import transforms

from datetime import datetime
from sklearn.metrics import confusion_matrix, accuracy_score
from tqdm import tqdm

from dataset import CMUMocap,HumanAct12, NTUDataset, get_train_val_set
# from mocap_dataset_demo import CMUMocap
from model import ConvLSTM

import torchcontrib

def get_train_val_loader(params, val_pct=0.2):
    train_samples, val_samples = get_train_val_set(data_path=params.data_path, val_pct=val_pct, temporal_aug_k=params.temporal_aug_k)
    print(f'Train samples: {len(train_samples)} || Validation samples: {len(val_samples)}')
    
    # Apply transform to normalize the data
    # transform = transforms.Normalize((0.5), (0.5))
    
    # Load train and validation dataset
    train_set = CMUMocap(sample_set=train_samples, params=params, transform=None)
    val_set =  CMUMocap(sample_set=val_samples, params=params, transform=None)

    train_loader = DataLoader(train_set, batch_size=params.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=params.BATCH_SIZE, shuffle=True)
    
    return train_loader, val_loader


def save_model(model):
    current_time = datetime.now()
    current_time = current_time.strftime("%m_%d_%Y_%H_%M")
    torch.save(model.state_dict(), f'/home/youngjoon/DEV/ntu-skeleton/code/saved_models/ntu_lstm_{current_time}.pth')
    
    
def build_test_stats(preds, actual, acc, params):
    print(f'Model accuracy: {acc}')
    
    # For confusion matrix
    preds = [int(k) for k in preds]
    actual = [int(k) for k in actual]

    cf = confusion_matrix(actual, preds, labels=list(range(params.num_classes)))
    print(cf)

def train(model, train_loader, loss_function, optimizer, params):
    print('Training...')
    
    for epoch in range(params.n_epochs):
        for batch in tqdm(train_loader):
            inputs = batch[0].float().cuda()
            labels=batch[1]
            # print(inputs.shape)
            # labels=batch[1].cuda()
            # labels = batch[1]
            # print(len(labels))
            # exit()
            tmp = []
            
            # for label in labels:
            #     if label.item() == 6:
            #         tmp.append(0)
            #     elif label.item() == 7:
            #         tmp.append(1)
            #     elif label.item() == 8:
            #         tmp.append(2)
            #     elif label.item() == 9:
            #         tmp.append(3)
            #     elif label.item() == 22:
            #         tmp.append(4)
            #     elif label.item() == 23:
            #         tmp.append(5)
            #     elif label.item() == 24:
            #         tmp.append(6)
            #     elif label.item() == 38:
            #         tmp.append(7)
            #     elif label.item() == 80:
            #         tmp.append(8)
            #     elif label.item() == 93:
            #         tmp.append(9)
            #     elif label.item() == 99:
            #         tmp.append(10)
            #     elif label.item() == 100:
            #         tmp.append(11)
            #     elif label.item() == 102:
            #         tmp.append(12)
            #humanact12
            # for label in labels:
            #     label = int(label)
            #     if label==101:
            #         tmp.append(0)
            #     elif label==102:
            #         tmp.append(1)
            #     elif label==103:
            #         tmp.append(2)
            #     elif label==104:
            #         tmp.append(3)
            #     elif label==105:
            #         tmp.append(4)
            #     elif label==106:
            #         tmp.append(5)
            #     elif label==107:
            #         tmp.append(6)
            #     elif label==201:
            #         tmp.append(7)
            #     elif label==301:
            #         tmp.append(8)
            #     elif label==401:
            #         tmp.append(9)
            #     elif label==402:
            #         tmp.append(10)
            #     elif label==501:
            #         tmp.append(11)
            #     elif label==502:
            #         tmp.append(12)
            #     elif label==503:
            #         tmp.append(13)
            #     elif label==504:
            #         tmp.append(14)
            #     elif label==505:
            #         tmp.append(15)
            #     elif label==601:
            #         tmp.append(16)
            #     elif label==602:
            #         tmp.append(17)
            #     elif label==603:
            #         tmp.append(18)
            #     elif label==604:
            #         tmp.append(19)
            #     elif label==605:
            #         tmp.append(20)
            #     elif label==701:
            #         tmp.append(21)
            #     elif label==801:
            #         tmp.append(22)
            #     elif label==802:
            #         tmp.append(23)
            #     elif label==803:
            #         tmp.append(24)
            #     elif label==901:
            #         tmp.append(25)
            #     elif label==1001:
            #         tmp.append(26)
            #     elif label==1002:
            #         tmp.append(27)
            #     elif label==1101:
            #         tmp.append(28)
            #     elif label==1102:
            #         tmp.append(29)
            #     elif label==1103:
            #         tmp.append(30)
            #     elif label==1104:
            #         tmp.append(31)
            #     elif label==1201:
            #         tmp.append(32)
            #     elif label==1202:
            #         tmp.append(33)
            CMUMocap
            for label in labels:
                label = int(label)
                if label=='Walk':
                    tmp.append(0)
                elif label=='Wash':
                    tmp.append(1)
                elif label=='Run':
                    tmp.append(2)
                elif label=='Jump':
                    tmp.append(3)
                elif label=='Animal Behavior':
                    tmp.append(4)
                elif label=='Dance':
                    tmp.append(5)
                elif label=='Step':
                    tmp.append(6)
                elif label=='Climb':
                    tmp.append(7)
            # print(label)
            # print(tmp)
            # exit()   
               
            # print(tmp)
            # exit()
               
            
            labels = torch.tensor(tmp).cuda()
            
            
            #inputs, labels = batch[0].to(device).float(), batch[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            # print(outputs.shape)
            # print("="*30)
            # print(labels)
            # print("="*30)
            # print(labels.shape)
            # exit()
            loss = loss_function(outputs, labels)
            
            loss.backward()
            
            optimizer.step()

        print(f'Epoch: {epoch} | Loss: {loss}')

    return model

def test(model, test_loader):
    print('Testing...')
    correct = 0
    total = 0

    preds = []
    actual = []

    with torch.no_grad():
        for batch in tqdm(test_loader):
            
            inputs = batch[0].float().cuda()
            # labels = batch[1].cuda()
            labels=batch[1]
            tmp = []
            # for label in labels:
            #     if label.item() == 6:
            #         tmp.append(0)
            #     elif label.item() == 7:
            #         tmp.append(1)
            #     elif label.item() == 8:
            #         tmp.append(2)
            #     elif label.item() == 9:
            #         tmp.append(3)
            #     elif label.item() == 22:
            #         tmp.append(4)
            #     elif label.item() == 23:
            #         tmp.append(5)
            #     elif label.item() == 24:
            #         tmp.append(6)
            #     elif label.item() == 38:
            #         tmp.append(7)
            #     elif label.item() == 80:
            #         tmp.append(8)
            #     elif label.item() == 93:
            #         tmp.append(9)
            #     elif label.item() == 99:
            #         tmp.append(10)
            #     elif label.item() == 100:
            #         tmp.append(11)
            #     elif label.item() == 102:
            #         tmp.append(12)

            #HumanAct12
            # for label in labels:
            #     label = int(label)
            #     if label==101:
            #         tmp.append(0)
            #     elif label==102:
            #         tmp.append(1)
            #     elif label==103:
            #         tmp.append(2)
            #     elif label==104:
            #         tmp.append(3)
            #     elif label==105:
            #         tmp.append(4)
            #     elif label==106:
            #         tmp.append(5)
            #     elif label==107:
            #         tmp.append(6)
            #     elif label==201:
            #         tmp.append(7)
            #     elif label==301:
            #         tmp.append(8)
            #     elif label==401:
            #         tmp.append(9)
            #     elif label==402:
            #         tmp.append(10)
            #     elif label==501:
            #         tmp.append(11)
            #     elif label==502:
            #         tmp.append(12)
            #     elif label==503:
            #         tmp.append(13)
            #     elif label==504:
            #         tmp.append(14)
            #     elif label==505:
            #         tmp.append(15)
            #     elif label==601:
            #         tmp.append(16)
            #     elif label==602:
            #         tmp.append(17)
            #     elif label==603:
            #         tmp.append(18)
            #     elif label==604:
            #         tmp.append(19)
            #     elif label==605:
            #         tmp.append(20)
            #     elif label==701:
            #         tmp.append(21)
            #     elif label==801:
            #         tmp.append(22)
            #     elif label==802:
            #         tmp.append(23)
            #     elif label==803:
            #         tmp.append(24)
            #     elif label==901:
            #         tmp.append(25)
            #     elif label==1001:
            #         tmp.append(26)
            #     elif label==1002:
            #         tmp.append(27)
            #     elif label==1101:
            #         tmp.append(28)
            #     elif label==1102:
            #         tmp.append(29)
            #     elif label==1103:
            #         tmp.append(30)
            #     elif label==1104:
            #         tmp.append(31)
            #     elif label==1201:
            #         tmp.append(32)
            #     elif label==1202:
            #         tmp.append(33)

            CMUMocap 
            for label in labels:
                label = int(label)
                if label=='Walk':
                    tmp.append(0)
                elif label=='Wash':
                    tmp.append(1)
                elif label=='Run':
                    tmp.append(2)
                elif label=='Jump':
                    tmp.append(3)
                elif label=='Animal Behavior':
                    tmp.append(4)
                elif label=='Dance':
                    tmp.append(5)
                elif label=='Step':
                    tmp.append(6)
                elif label=='Climb':
                    tmp.append(7)
                

            labels = torch.tensor(tmp).cuda()
            
            # inputs, labels = batch
            # inputs, labels = inputs.to(device).float(), labels.to(device)

            class_outputs = model(inputs)
            _, class_prediction = torch.max(class_outputs.data, 1)
            total += labels.size(0)
            correct += (class_prediction == labels).sum().item()
            preds.extend(list(class_prediction.to(dtype=torch.int64)))
            actual.extend(list(labels.to(dtype=torch.int64)))

    acc = 100*correct/total
    return preds, actual, acc


def main(params):
    # Initialize some variables to track progress
    accs = []
    
    # Initialize the model
    model = ConvLSTM(params=params).to(device)
    
    # Use parallel computing if available
    if device.type == 'cuda' and n_gpus > 1:
        model = nn.DataParallel(model, list(range(n_gpus)))
       
    # Loss Function and Optimizer (can use weight=class_weights if it is a disbalanced dataset)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # optimizer = optim.AdamW(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01)
    
    # Get train and validation loaders
    train_loader, val_loader = get_train_val_loader(params, val_pct=params.val_pct)
    
    # Train the model
    model = train(model, train_loader, loss_function, optimizer, params)
    save_model(model)

    # model = train_test(model, train_loader, val_loader, loss_function, optimizer, params)
    # save_model(model)
     
    # Get training accuracy
    preds, actual, acc = test(model, train_loader)
    build_test_stats(preds, actual, acc, params)
    
    # Validate the model
    preds, actual, acc = test(model, val_loader)
    build_test_stats(preds, actual, acc, params)
    

## Optional code to load and test a model
def load_test_model(params, model_path):
    model = ConvLSTM(params=params).to(device)
    # Use this to fix keyError in the model when using DataParallel while training
    if device.type == 'cuda' and n_gpus > 1:
        model = nn.DataParallel(model, list(range(n_gpus)))
    train_loader, val_loader = get_train_val_loader(params, val_pct=0.2)
    model.load_state_dict(torch.load(model_path))
    model.eval() # To set dropout and batchnormalization OFF
    preds, actual, acc = test(model, val_loader)
    build_test_stats(preds, actual, acc, params)
    
    
# def train_test(model, train_loader, test_loader, loss_function, optimizer, params):
#     print('Training...')
    
#     for epoch in range(params.n_epochs):
#         for batch in tqdm(train_loader):
#             i = 0
#             inputs = batch[0].float().cuda()
#             labels = batch[1].cuda()
#             tmp = []
#             for label in labels:
#                 if label.item() == 6:
#                     tmp.append(0)
#                 elif label.item() == 7:
#                     tmp.append(1)
#                 elif label.item() == 8:
#                     tmp.append(2)
#                 elif label.item() == 9:
#                     tmp.append(3)
#                 elif label.item() ==  22:
#                     tmp.append(4)
#                 elif label.item() == 23:
#                     tmp.append(5)
#                 elif label.item() == 24:
#                     tmp.append(6)
#                 elif label.item() == 38:
#                     tmp.append(7)
#                 elif label.item() == 80:
#                     tmp.append(8)
#                 elif label.item() == 93:
#                     tmp.append(9)
#                 elif label.item() == 99:
#                     tmp.append(10)
#                 elif label.item() == 100:
#                     tmp.append(11)
#                 elif label.item() == 102:
#                     tmp.append(12)
            
#             labels = torch.tensor(tmp).cuda()
        
            
#             #inputs, labels = batch[0].to(device).float(), batch[1].to(device)
#             optimizer.zero_grad()
#             outputs = model(inputs)
#             loss = loss_function(outputs, labels)            

#             loss.backward()
            
#             optimizer.step()
#     #######################################################################################
#         print('Testing...')
#         correct = 0
#         total = 0

#         preds = []
#         actual = []

#         with torch.no_grad():
#             for batch in tqdm(test_loader):
#                 inputs = batch[0].float().cuda()
#                 labels = batch[1].cuda()
#                 tmp = []
#                 for label in labels:
#                     if label.item() == 6:
#                         tmp.append(0)
#                     elif label.item() == 7:
#                         tmp.append(1)
#                     elif label.item() == 8:
#                         tmp.append(2)
#                     elif label.item() == 9:
#                         tmp.append(3)
#                     elif label.item() ==  22:
#                         tmp.append(4)
#                     elif label.item() == 23:
#                         tmp.append(5)
#                     elif label.item() == 24:
#                         tmp.append(6)
#                     elif label.item() == 38:
#                         tmp.append(7)
#                     elif label.item() == 80:
#                         tmp.append(8)
#                     elif label.item() == 93:
#                         tmp.append(9)
#                     elif label.item() == 99:
#                         tmp.append(10)
#                     elif label.item() == 100:
#                         tmp.append(11)
#                     elif label.item() == 102:
#                         tmp.append(12)
                
#                 labels = torch.tensor(tmp).cuda()
#                 # inputs, labels = batch
#                 # inputs, labels = inputs.to(device).float(), labels.to(device)

#                 class_outputs = model(inputs)
#                 _, class_prediction = torch.max(class_outputs.data, 1)
#                 total += labels.size(0)
#                 correct += (class_prediction == labels).sum().item()
#                 preds.extend(list(class_prediction.to(dtype=torch.int64)))
#                 actual.extend(list(labels.to(dtype=torch.int64)))

#             acc = 100*correct/total
#             print(f'Epoch: {epoch} | Loss: {loss} | test_acc: {acc}')
#     return model, preds, actual, acc

if __name__ == '__main__':
    """
    - kp_shape = (25,3)
    - seg_size = varies based on the action being performed (so, select a minimum segment size among all samples in the dataset)
    - participant_list <= those who are in the train or validation or test set (a list of numbers/codes for the participants)
    - data_path = '/data/zak/graph/ntu/train'
    - BATCH_SIZE <== For the model
    - temporal_aug_k <== Defines number of random samples from one segment (for temporal augmentation)
    """

    parser = argparse.ArgumentParser(description='NTU Activity Recognition with 3D Keypoints')
    parser.add_argument('--mode', type=str, default='train', help='train || inference')
    parser.add_argument('--model_path', type=str, default='/home/youngjoon/DEV/ntu-skeleton/code/saved_models/', help='Enter the path to the saved model')
    parser.add_argument('--data_path', type=str, default='/home/youngjoon/Desktop/Dataset/mocap_3djoints/', help='Dataset path')
    parser.add_argument('--seg_size', type=int, default=50, help='Minimum segment size for each video segment')
    parser.add_argument('--val_pct', type=float, default=0.2, help='Ratio for the validation set')
    parser.add_argument('--kp_shape', type=list, nargs=2, default=[20, 3], help='(n_joints, n_coordinates) -- (25, 3)')
    parser.add_argument('--BATCH_SIZE', type=int, default=512, help='Batch size for the dataset')
    parser.add_argument('--temporal_aug_k', type=int, default=5, help='Number of temporal augmentations for each sample')
    parser.add_argument('--k_fold', type=int, default=1, help='k-Fold validation')
    parser.add_argument('--n_epochs', type=int, default=10, help='Number of Epochs to train the model')
    parser.add_argument('--num_joints', type=int, default=20, help='set joints')


    parsed_input = parser.parse_args()

    params = edict({
        'mode': parsed_input.mode,
        'model_path': parsed_input.model_path,
        'kp_shape': parsed_input.kp_shape,
        'val_pct': parsed_input.val_pct,
        'seg_size': parsed_input.seg_size,
        'data_path': parsed_input.data_path,
        'BATCH_SIZE': parsed_input.BATCH_SIZE,
        'temporal_aug_k': parsed_input.temporal_aug_k,
        'k_fold': parsed_input.k_fold,
        'n_epochs': parsed_input.n_epochs,
        'num_classes': 34,
        'bcc': 8, # base convolution channels
        'num_channels': 1, # channel for each frame
        'num_joints': parsed_input.num_joints, # joints used in each frame
        'num_coord': 3, # number of coordinates (x, y, z)
    })
    if params.data_path == "/home/youngjoon/Desktop/Dataset/ntu_gen_real_npy/":
        params.kp_shape=[18,3]
        params.num_joints=18
    elif params.data_path=="/home/youngjoon/Desktop/Dataset/humanact12/":
        params.kp_shape=[24,3]
        params.num_joints=24
    elif params.data_path=='/home/youngjoon/Desktop/Dataset/mocap_3djoints/':
        params.kp_shape=[20,3]
        params.num_joints=20
    else:
        params.kp_shape=[25,3]
        params.num_joints=25
    # Check for GPUs
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    n_gpus = torch.cuda.device_count()
    print(f'Number of GPUs available: {n_gpus}')
    
    if params.mode == 'train':
        # Run the train/val code
        main(params)
    else:
        # Run as this: python main.py --mode="inference" --model_path="../saved_models/ntu_lstm_01_27_2021_15_53.pth"
        if not params.model_path:
            print('Please enter path to the saved model!: python main.py --mode="inference" --model_path="../saved_models/ntu_lstm_01_27_2021_15_53.pth"')
        else:
            # For loading and testing model
            load_test_model(params, model_path=params.model_path)
    

