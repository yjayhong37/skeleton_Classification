# import numpy as np
# import os
# import random

# import torch
# from torch.utils.data import Dataset, DataLoader
# import param_Util as paramUtil
# from lie.pose_lie import *
# import pandas as pd
# import csv
# import os
# import numpy as np
# import numpy.matlib
# import codecs as cs
# import scipy.io as sio


# class CMUMocap(Dataset):
#     def __init__(self, filename, datapath,sample_set, params,opt, do_offset=True,transform=None):
#         self.clip = pd.read_csv(filename, index_col=False).dropna(how='all').dropna(axis=1, how='all')
#         self.data_path = params.data_path
#         self.sample_set = sample_set
#         self.kp_shape = params.kp_shape
#         self.seg_size = params.seg_size
#         self.num_channels = params.num_channels
#         self.transform = transform
#         self.lengths = []
#         self.data = []
#         self.labels = []
#         self.opt = opt
#         self.mean, self.std = self.get_mean_std()

#     def __len__(self):
#         return len(self.sample_set)

#     def __getitem__(self, index):
#         pose_mat, label = self.data[index]
#         label = self.label_enc[label]
#         return pose_mat, label

#     def read_sample(self,sample_name):
#         sample_path=os.path.join(self.data_path,sample_name)
#         kps=None
#         data=np.load(sample_path)
#         kps=self.augment_kp(data)
#         for i in range(self.clip.shape[0]):
#             action_class=self.clip.iloc[i]['action_type']
            
#         return np.reshape(kps, (self.seg_size, 1, self.kp_shape[0], self.kp_shape[1])), action_class

#     def augment_kp(self, sample_kp):
#         # Temporally augment video segment based on the minimum segment size for the dataset
#         # Randomly take "seg_size" number of frames from the segment (in chronological order)
#         sample_size = sample_kp.shape[0]
#         if sample_size < self.seg_size:
#                 # Pad same frames at the end in order to meet the segment size requirement
#             return self.pad_frames(sample_kp)
#         rand_segments = sorted(random.sample(range(0, sample_size), self.seg_size))
#         sample_kp = np.take(sample_kp, rand_segments, axis=0)
#         return sample_kp

#     def pad_frames(self, sample_kp):
#         # Consider seg_size for the dataset is 40 and the current sample has only 15 frames in the segment
#         # We will need to repeat the frames in order to make it reach 40
#         padded_kp = sample_kp
#         sample_size = sample_kp.shape[0]
#         additional_frames = self.seg_size - sample_size
#         while additional_frames >= sample_size:
#             padded_kp = np.concatenate((padded_kp, sample_kp), axis=0)
#             additional_frames -= sample_size
                
#         padded_kp = np.concatenate((padded_kp, sample_kp[:additional_frames]), axis=0)
#         return padded_kp
    
#     def get_mean_std(self):
#         mean_path = f'../mean_std/mean_{self.seg_size}_{self.kp_shape[0]}_{self.kp_shape[1]}.npy'
#         std_path = f'../mean_std/std_{self.seg_size}_{self.kp_shape[0]}_{self.kp_shape[1]}.npy'
#         try:
#             # Read pickled file for mean and std
#             mean = torch.from_numpy(np.load(mean_path))
#             std = torch.from_numpy(np.load(std_path))
#         except OSError:
#             print('Evaluating mean and std for the training set...')         
#             X = torch.tensor([self.read_sample(sample_name)[0] for sample_name in self.sample_set])
#             X = X.view(-1, self.seg_size, self.num_channels, self.kp_shape[0], self.kp_shape[1])
#             mean = torch.mean(X, axis=0)
#             std = torch.std(X, axis=0)
#             np.save(mean_path, mean.numpy())
#             np.save(std_path, std.numpy())
#             print(f'Mean and Std saved successfully!')
            
#         return mean, std

#     def get_participant_number(file_name):
#         return file_name.split('P')[1][:3]

# def split_participants(data_path, val_pct=0.2):
#     # Returns a random list of participants for the train and validation sets respectively
#     samples = os.listdir(data_path)
#     total_samples = len(samples)
#     # Get all unique participant numbers
#     all_participants = set()
#     for sample in samples:
#         part = get_participant_number(sample)
#         all_participants.add(part)
#     total_participants = len(all_participants)
#     all_participants = list(all_participants)
    
#     # Split into train and val sets
#     val_len = int(total_participants * val_pct)
#     # Randomly shuffle the list
#     random.shuffle(list(all_participants))
#     train_participants = all_participants[val_len:]
#     val_participants = all_participants[:val_len]

#     print(f'Total Video Samples: {len(samples)} || Total Participants: {len(all_participants)} || Train Participants: {len(train_participants)} || Validation Participants: {len(val_participants)}')
#     return train_participants, val_participants

# def get_train_val_set(data_path, val_pct=0.2, temporal_aug_k=3):
#     train_participants, val_participants = split_participants(data_path, val_pct)
    
    
#     train_samples, val_samples = [], []

#     for sample in os.listdir(data_path):
#         participant_number = get_participant_number(sample)
        
#         # Temporary code to check the minimum segment size in the dataset
#         # data = np.load(os.path.join(data_path, sample), allow_pickle=True).item()['skel_body0']
#         # min_seg_size = min(min_seg_size, data.shape[0])
        
#         # Apply data augmentation here ('k' times random temporal augmentation)
#         for _ in range(temporal_aug_k):
#             if participant_number in val_participants:
#                 val_samples.append(sample)
#             else:
#                 train_samples.append(sample)
    
#     # print(f'Minimum segment size in the dataset: {min_seg_size}')
#     return train_samples, val_samples


#         # for i in range(self.clip.shape[0]):
#         #     motion_name = self.clip.iloc[i]['motion']
#         #     action_type = self.clip.iloc[i]['action_type']
#         #     npy_path = os.path.join(datapath, motion_name + '.npy')

#         #     # motion_length, joints_num, 3
#         #     pose_raw = np.load(npy_path)
#         #     # rescale the pose
#         #     pose_raw = pose_raw / 20

#         #     # Locate the root joint of initial pose at origin
#         #     if do_offset:
#         #         # get the offset and return the final pose
#         #         offset_mat = np.tile(pose_raw[0, 0], (pose_raw.shape[1], 1))
#         #         pose_mat = pose_raw - offset_mat
#         #     else:
#         #         pose_mat = pose_raw

#         #     pose_mat = pose_mat.reshape((-1, 20 * 3))

#         #     # not used any more
#         #     if self.opt.no_trajectory:
#         #         # for lie params, just exclude the root translation part
#         #         if self.opt.lie_enforce:
#         #             pose_mat = pose_mat[:, 3:]
#         #         else:
#         #             offset = np.tile(pose_mat[..., :3], (1, int(pose_mat.shape[1] / 3)))
#         #             pose_mat = pose_mat - offset

#         #     self.data.append((pose_mat, action_type))
#         #     if action_type not in self.labels:
#         #         self.labels.append(action_type)
#         #     self.lengths.append(pose_mat.shape[0])
#         # self.cumsum = np.cumsum([0] + self.lengths)
#         # print("Total number of frames {}, videos {}, action types {}".format(self.cumsum[-1], self.clip.shape[0],
#         #                                                                      len(self.labels)))
#         # self.label_enc = dict(zip(self.labels, np.arange(len(self.labels))))
#         # self.label_enc_rev = dict(zip(np.arange(len(self.labels)), self.labels))
#         # with codecs.open(os.path.join(opt.save_root, "label_enc_rev_mocap.txt"), 'w', 'utf-8') as f:
#         #     for item in self.label_enc_rev.items():
#         #         f.write(str(item) + "\n")

 

#     def get_label_reverse(self, enc_label):
#         return self.label_enc_rev.get(enc_label)

   