#!/bin/bash

for epoch in 100:
    do
        echo $epoch
        ! python main.py --data_path='/home/youngjoon/Desktop/ntu_gen_real_npy/' 
        ! python main.py --data_path='/home/youngjoon/Desktop/Dataset/humanact12/'
        ! python main.py --data_path='/home/youngjoon/Desktop/Dataset/mocap_3djoints/' -- kp_shape=(20,3) --num_joints=20
    done
python main.py --data_path='/home/youngjoon/Desktop/ntu_gen_real_npy/' --kp_shape=(18,3) --num_joints=18 --n_epochs=10
python main.py --data_path='/home/youngjoon/Desktop/ntu_raw_npy/' --n_epochs=1
python main.py --data_path='/home/youngjoon/Desktop/Dataset/humanact12/' -- kp_shape=(24,3) --num_joints=24
python main.py --data_path='/home/youngjoon/Desktop/Dataset/mocap_3djoints/' -- kp_shape=(20,3) --num_joints=20