# 4D_fMRI_Transformer
import os
import torch
from torch.utils.data import Dataset, IterableDataset

# import augmentations #commented out because of cv errors
import pandas as pd
from pathlib import Path
import numpy as np
import nibabel as nb
import nilearn
import random

from itertools import cycle
import glob

from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, KBinsDiscretizer

class BaseDataset(Dataset):
    def __init__(self, **kwargs):
        super().__init__()      
        self.register_args(**kwargs)
        self.sample_duration = self.sequence_length * self.stride_within_seq
        self.stride = max(round(self.stride_between_seq * self.sample_duration),1)
        self.data = self._set_data(self.root, self.subject_dict)
    
    def register_args(self,**kwargs):
        for name,value in kwargs.items():
            setattr(self,name,value)
        self.kwargs = kwargs
    
    def load_sequence(self, subject_path, start_frame, sample_duration, num_frames=None): 
        if self.contrastive:
            num_frames = len(os.listdir(subject_path)) - 2
            y = []
            load_fnames = [f'frame_{frame}.pt' for frame in range(start_frame, start_frame+sample_duration,self.stride_within_seq)]
            if self.with_voxel_norm:
                load_fnames += ['voxel_mean.pt', 'voxel_std.pt']

            for fname in load_fnames:
                img_path = os.path.join(subject_path, fname)
                y_loaded = torch.load(img_path).unsqueeze(0)
                y.append(y_loaded)
            y = torch.cat(y, dim=4)
            
            random_y = []
            
            full_range = np.arange(0, num_frames-sample_duration+1)
            # exclude overlapping sub-sequences within a subject
            exclude_range = np.arange(start_frame-sample_duration, start_frame+sample_duration)
            available_choices = np.setdiff1d(full_range, exclude_range)
            random_start_frame = np.random.choice(available_choices, size=1, replace=False)[0]
            load_fnames = [f'frame_{frame}.pt' for frame in range(random_start_frame, random_start_frame+sample_duration,self.stride_within_seq)]
            if self.with_voxel_norm:
                load_fnames += ['voxel_mean.pt', 'voxel_std.pt']
            for fname in load_fnames:
                img_path = os.path.join(subject_path, fname)
                y_loaded = torch.load(img_path).unsqueeze(0)
                random_y.append(y_loaded)
            random_y = torch.cat(random_y, dim=4)
            return (y, random_y)

        else: # without contrastive learning
            y = []
            if self.shuffle_time_sequence: # shuffle whole sequences
                load_fnames = [f'frame_{frame}.pt' for frame in random.sample(list(range(0,num_frames)),sample_duration//self.stride_within_seq)]
            else:
                load_fnames = [f'frame_{frame}.pt' for frame in range(start_frame, start_frame+sample_duration,self.stride_within_seq)]
            
            if self.with_voxel_norm:
                load_fnames += ['voxel_mean.pt', 'voxel_std.pt']
                
            for fname in load_fnames:
                img_path = os.path.join(subject_path, fname)
                y_i = torch.load(img_path).unsqueeze(0)
                y.append(y_i)
            y = torch.cat(y, dim=4)
            return y

    def __len__(self):
        return  len(self.data)

    def __getitem__(self, index):
        raise NotImplementedError("Required function")

    def _set_data(self, root, subject_dict):
        raise NotImplementedError("Required function")

class S1200(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _set_data(self, root, subject_dict):
        data = []
        img_root = os.path.join(root, 'img')
        for i, subject in enumerate(subject_dict):
            sex,target = subject_dict[subject]
            subject_path = os.path.join(img_root, subject)
            num_frames = len(os.listdir(subject_path)) - 2 # voxel mean & std
            session_duration = num_frames - self.sample_duration + 1
            for start_frame in range(0, session_duration, self.stride):
                data_tuple = (i, subject, subject_path, start_frame, self.stride, num_frames, target, sex)
                data.append(data_tuple)
        
        # train dataset
        # for regression tasks
        if self.train: 
            self.target_values = np.array([tup[6] for tup in data]).reshape(-1, 1)
        return data

    def __getitem__(self, index):
        _, subject, subject_path, start_frame, sequence_length, num_frames, target, sex = self.data[index]
        # target = self.label_dict[target] if isinstance(target, str) else target.float()

        if self.contrastive:
            y, rand_y = self.load_sequence(subject_path, start_frame, sequence_length)

            background_value = y.flatten()[0]
            y = y.permute(0,4,1,2,3)
            y = torch.nn.functional.pad(y, (6, 6, 0, 0, 9, 9), value=background_value) # adjust this padding level according to your data
            y = y.permute(0,2,3,4,1)

            background_value = rand_y.flatten()[0]
            rand_y = rand_y.permute(0,4,1,2,3)
            rand_y = torch.nn.functional.pad(rand_y, (6, 6, 0, 0, 9, 9), value=background_value) # adjust this padding level according to your data
            rand_y = rand_y.permute(0,2,3,4,1)

            return {
                "fmri_sequence": (y, rand_y),
                "subject_name": subject,
                "target": target,
                "TR": start_frame,
                "sex": sex
            }

        else:
            y = self.load_sequence(subject_path, start_frame, sequence_length, num_frames)

            background_value = y.flatten()[0]
            y = y.permute(0,4,1,2,3)
            y = torch.nn.functional.pad(y, (6, 6, 0, 0, 9, 9), value=background_value) # adjust this padding level according to your data
            y = y.permute(0,2,3,4,1)

            return {
                "fmri_sequence": y,
                "subject_name": subject,
                "target": target,
                "TR": start_frame,
                "sex": sex,
            } 

class ABCD(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _set_data(self, root, subject_dict):
        data = []
        img_root = os.path.join(root, 'img')

        for i, subject_name in enumerate(subject_dict):
            sex, target = subject_dict[subject_name]
            # subject_name = subject[4:]
            
            subject_path = os.path.join(img_root, 'sub-'+subject_name)

            num_frames = len(os.listdir(subject_path)) - 2 # voxel mean & std
            session_duration = num_frames - self.sample_duration + 1

            for start_frame in range(0, session_duration, self.stride):
                data_tuple = (i, subject_name, subject_path, start_frame, self.stride, num_frames, target, sex)
                data.append(data_tuple)
                        
        
        # train dataset
        # for regression tasks
        if self.train: 
            self.target_values = np.array([tup[6] for tup in data]).reshape(-1, 1)

        return data

    def __getitem__(self, index):
        _, subject_name, subject_path, start_frame, sequence_length, num_frames, target, sex = self.data[index]
        #age = self.label_dict[age] if isinstance(age, str) else age.float()
        
        #contrastive learning
        if self.contrastive:
            y, rand_y = self.load_sequence(subject_path, start_frame, sequence_length)

            background_value = y.flatten()[0]
            y = y.permute(0,4,1,2,3)
            # ABCD image shape: 79, 97, 85
            y = torch.nn.functional.pad(y, (6, 5, 0, 0, 9, 8), value=background_value)[:,:,:,:96,:] # adjust this padding level according to your data
            y = y.permute(0,2,3,4,1)

            background_value = rand_y.flatten()[0]
            rand_y = rand_y.permute(0,4,1,2,3)
            # ABCD image shape: 79, 97, 85
            rand_y = torch.nn.functional.pad(rand_y, (6, 5, 0, 0, 9, 8), value=background_value)[:,:,:,:96,:] # adjust this padding level according to your data
            rand_y = rand_y.permute(0,2,3,4,1)

            return {
                "fmri_sequence": (y, rand_y),
                "subject_name": subject_name,
                "target": target,
                "TR": start_frame,
                "sex": sex
            } 

        # resting or task
        else:   
            y = self.load_sequence(subject_path, start_frame, sequence_length, num_frames)

            background_value = y.flatten()[0]
            y = y.permute(0,4,1,2,3)
            if self.input_type == 'rest':
                # ABCD rest image shape: 79, 97, 85
                # latest version might be 96,96,95
                y = torch.nn.functional.pad(y, (0, 1, 0, 0, 0, 0), value=background_value)[:,:,:,:96,:] # adjust this padding level according to your data
            elif self.input_type == 'task':
                # ABCD task image shape: 96, 96, 95
                # background value = 0
                # minmax scaled in brain (0~1)
                y = torch.nn.functional.pad(y, (0, 1, 0, 0, 0, 0), value=background_value) # adjust this padding level according to your data
            y = y.permute(0,2,3,4,1)

            return {
                "fmri_sequence": y,
                "subject_name": subject_name,
                "target": target,
                "TR": start_frame,
                "sex": sex,
            } 
        

class UKB(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _set_data(self, root, subject_dict):
        data = []
        img_root = os.path.join(root, 'img')
        # subject_list = [subj for subj in os.listdir(img_root) if subj.endswith('20227_2_0')] # only use release 2

        for i, subject_name in enumerate(subject_dict):
            sex, target = subject_dict[subject_name]
            subject20227 = str(subject_name)+'_20227_2_0'
            subject_path = os.path.join(img_root, subject20227)
            num_frames = len(os.listdir(subject_path)) - 2 # voxel mean & std
            session_duration = num_frames - self.sample_duration + 1

            for start_frame in range(0, session_duration, self.stride):
                data_tuple = (i, subject_name, subject_path, start_frame, self.stride, num_frames, target, sex)
                data.append(data_tuple)
        
        # train dataset
        # for regression tasks
        if self.train: 
            self.target_values = np.array([tup[6] for tup in data]).reshape(-1, 1)

        return data

    def __getitem__(self, index):
        _, subject_name, subject_path, start_frame, sequence_length, num_frames, target, sex = self.data[index]
        if self.contrastive:
                y, rand_y = self.load_sequence(subject_path, start_frame, sequence_length)

                background_value = y.flatten()[0]
                y = y.permute(0,4,1,2,3)
                y = torch.nn.functional.pad(y, (3, 2, -7, -6, 3, 2), value=background_value) # adjust this padding level according to your data
                y = y.permute(0,2,3,4,1)

                background_value = rand_y.flatten()[0]
                rand_y = rand_y.permute(0,4,1,2,3)
                rand_y = torch.nn.functional.pad(rand_y, (3, 2, -7, -6, 3, 2), value=background_value) # adjust this padding level according to your data
                rand_y = rand_y.permute(0,2,3,4,1)

                return {
                    "fmri_sequence": (y, rand_y),
                    "subject_name": subject_name,
                    "target": target,
                    "TR": start_frame,
                    "sex": sex
                }
        else:
            y = self.load_sequence(subject_path, start_frame, sequence_length, num_frames)

            background_value = y.flatten()[0]
            y = y.permute(0,4,1,2,3)
            y = torch.nn.functional.pad(y, (3, 2, -7, -6, 3, 2), value=background_value) # adjust this padding level according to your data
            y = y.permute(0,2,3,4,1)
            return {
                        "fmri_sequence": y,
                        "subject_name": subject_name,
                        "target": target,
                        "TR": start_frame,
                        "sex": sex,
                    } 
        
        
class dHCP(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _set_data(self, root, subject_dict):
        data = []
                
        if self.use_ic:
            self.mask = nb.load(self.input_mask_path).get_fdata() 
            self.non_zero_indices = np.nonzero(self.mask.flatten())[0]  
            
            for i, sub_ses in enumerate(subject_dict):
                sex, target = subject_dict[sub_ses]
                comps = os.path.join(str(sub_ses), str(sub_ses)+f'_features_{self.sequence_length}_comps.npy')
                subject_path = os.path.join(self.input_features_path, comps)
                
                start_frame = 0
                data_tuple = (i, sub_ses, subject_path, start_frame, target, sex) 
                data.append(data_tuple)
            # train dataset
            # for regression tasks
            if self.train: 
                self.target_values = np.array([tup[4] for tup in data]).reshape(-1, 1)
        else:
            img_root = os.path.join(root, 'img')

            for i, sub_ses in enumerate(subject_dict):
                sex, target = subject_dict[sub_ses]
                # subject_name = subject[4:]
            
                subject_path = os.path.join(img_root, sub_ses)

                num_frames = len(os.listdir(subject_path)) - 2 # voxel mean & std
                session_duration = num_frames - self.sample_duration + 1
                
                if self.use_first_sequence:
                    data_tuple = (i, sub_ses, subject_path, 0, self.stride, num_frames, target, sex)
                    data.append(data_tuple)
                else:
                    for start_frame in range(0, session_duration, self.stride):
                        data_tuple = (i, sub_ses, subject_path, start_frame, self.stride, num_frames, target, sex)
                        data.append(data_tuple)
                        
        
            # train dataset
            # for regression tasks
            if self.train: 
                self.target_values = np.array([tup[6] for tup in data]).reshape(-1, 1)

        return data

    def __getitem__(self, index):
        
        if self.use_ic:
            _, sub_ses, subject_path, start_frame, target, sex = self.data[index]
            features = np.load(subject_path) # ic * features
            num_ic = features.shape[0]
            final_matrix = np.zeros((num_ic,) + self.mask.shape)
            flat_final_matrix = final_matrix.reshape(num_ic, -1)
            flat_final_matrix[:, self.non_zero_indices] = features
            final_matrix = flat_final_matrix.reshape((num_ic,) + self.mask.shape)
            y = torch.tensor(final_matrix)
            
            #padding
            background_value = y.flatten()[0]
            y = torch.nn.functional.pad(y, (9, 10, 4, 5, 9, 10), value=background_value)
            
            y = y.permute(1,2,3,0).unsqueeze(0).half() # 1, 64, 64, 64, ic
                
            return {
                    "fmri_sequence": y,
                    "subject_name": sub_ses,
                    "target": target,
                    "TR": start_frame,
                    "sex": sex,
                }
        else:
        
            _, sub_ses, subject_path, start_frame, sequence_length, num_frames, target, sex = self.data[index]
            #age = self.label_dict[age] if isinstance(age, str) else age.float()
        
            #contrastive learning
            if self.contrastive:
                y, rand_y = self.load_sequence(subject_path, start_frame, sequence_length)

                background_value = y.flatten()[0]
                y = y.permute(0,4,1,2,3)
                # ABCD image shape: 79, 97, 85
                # y = torch.nn.functional.pad(y, (6, 5, 0, 0, 9, 8), value=background_value)[:,:,:,:96,:] # adjust this padding level according to your data
                # dHCP rest image shape: 68, 67, 45
                # y = torch.nn.functional.pad(y, (26, 25, 15, 14, 14, 14), value=background_value)[:,:,:,:96,:] # adjust this padding level according to your data
                # dHCP rest image shape (registered): 45, 55, 45
                y = torch.nn.functional.pad(y, (10, 9, 5, 4, 10, 9), value=background_value)[:,:,:,:96,:] # adjust this padding level according to your data
                y = y.permute(0,2,3,4,1)

                background_value = rand_y.flatten()[0]
                rand_y = rand_y.permute(0,4,1,2,3)
                # ABCD image shape: 79, 97, 85
                # rand_y = torch.nn.functional.pad(rand_y, (6, 5, 0, 0, 9, 8), value=background_value)[:,:,:,:96,:] # adjust this padding level according to your data
                # dHCP rest image shape: 68, 67, 45
                # y = torch.nn.functional.pad(y, (26, 25, 15, 14, 14, 14), value=background_value)[:,:,:,:96,:] # adjust this padding level according to your data
                # dHCP rest image shape (registered): 45, 55, 45
                y = torch.nn.functional.pad(y, (10, 9, 5, 4, 10, 9), value=background_value)[:,:,:,:96,:] # adjust this padding level according to your data
                rand_y = rand_y.permute(0,2,3,4,1)

                return {
                    "fmri_sequence": (y, rand_y),
                    "subject_name": sub_ses,
                    "target": target,
                    "TR": start_frame,
                    "sex": sex
                } 

            # resting or task
            else:   
                y = self.load_sequence(subject_path, start_frame, sequence_length, num_frames)

                background_value = y.flatten()[0]
                y = y.permute(0,4,1,2,3)
                if self.input_type == 'rest':
                    # ABCD rest image shape: 79, 97, 85
                    # latest version might be 96,96,95
                    # y = torch.nn.functional.pad(y, (6, 5, 0, 0, 9, 8), value=background_value)[:,:,:,:96,:] # adjust this padding level according to your data
                    # dHCP rest image shape: 68, 67, 45
                    # y = torch.nn.functional.pad(y, (26, 25, 15, 14, 14, 14), value=background_value)[:,:,:,:96,:] # adjust this padding level according to your data
                    # dHCP rest image shape (registered): 45, 55, 45
                    y = torch.nn.functional.pad(y, (10, 9, 5, 4, 10, 9), value=background_value)[:,:,:,:96,:] # adjust this padding level according to your data
                elif self.input_type == 'task':
                    # ABCD task image shape: 96, 96, 95
                    # background value = 0
                    # minmax scaled in brain (0~1)
                    # y = torch.nn.functional.pad(y, (0, 1, 0, 0, 0, 0), value=background_value) # adjust this padding level according to your data
                    # dHCP rest image shape: 68, 67, 45
                    # y = torch.nn.functional.pad(y, (26, 25, 15, 14, 14, 14), value=background_value)[:,:,:,:96,:] # adjust this padding level according to your data
                    # dHCP rest image shape (registered): 45, 55, 45
                    y = torch.nn.functional.pad(y, (10, 9, 5, 4, 10, 9), value=background_value)[:,:,:,:96,:] # adjust this padding level according to your data
                y = y.permute(0,2,3,4,1)

                return {
                    "fmri_sequence": y,
                    "subject_name": sub_ses,
                    "target": target,
                    "TR": start_frame,
                    "sex": sex,
                } 
    
class Dummy(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, total_samples=100000)
        

    def _set_data(self, root, subject_dict):
        data = []
        for k in range(0,self.total_samples):
            data.append((k, 'subj'+ str(k), 'path'+ str(k), self.stride))
        
        # train dataset
        # for regression tasks
        if self.train: 
            self.target_values = np.array([val for val in range(len(data))]).reshape(-1, 1)
            
        return data

    def __len__(self):
        return self.total_samples

    def __getitem__(self,idx):
        _, subj, _, sequence_length = self.data[idx]
        # Use simple default if img_size not set, but it should be passed via kwargs -> BaseDataset
        h, w, d = self.img_size[0], self.img_size[1], self.img_size[2]
        y = torch.randn(( 1, h, w, d, sequence_length),dtype=torch.float32) #self.y[seq_idx]
        sex = torch.randint(0,2,(1,)).float()
        target = torch.randint(0,2,(1,)).float()

        if self.contrastive:
            rand_y = torch.randn(( 1, h, w, d, sequence_length),dtype=torch.float32)
            return {
                "fmri_sequence": (y, rand_y),
                "subject_name": subj,
                "target": target,
                "TR": 0,
                }
        else:
            return {
                    "fmri_sequence": y,
                    "subject_name": subj,
                    "target": target,
                    "TR": 0,
                    "sex": sex,
                    } 

class Music(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _set_data(self, root, subject_dict):
        data = []
        img_root = os.path.join(root, 'img')
        
        for i, subject in enumerate(subject_dict):
            # Music dataset might not have target/sex in the same way, adapting generic placeholder
            if subject in subject_dict:
                 sex, target = subject_dict[subject]
            else:
                 sex, target = 0, 0 # Default if not in dict
            
            subject_path = os.path.join(img_root, subject)
            if not os.path.exists(subject_path): continue
            
            num_frames = len(glob.glob(os.path.join(subject_path, 'frame_*.pt')))
            session_duration = num_frames - self.sample_duration + 1
            
            for start_frame in range(0, session_duration, self.stride):
                data_tuple = (i, subject, subject_path, start_frame, self.stride, num_frames, target, sex)
                data.append(data_tuple)
        
        if self.train:
             # Dummy target values if not strictly regression
             self.target_values = np.array([tup[6] for tup in data]).reshape(-1, 1)
        return data

    def __getitem__(self, index):
        # Reusing similar logic to S1200/ABCD but adapted for Music if needed
        _, subject, subject_path, start_frame, sequence_length, num_frames, target, sex = self.data[index]
        
        if self.contrastive:
            y, rand_y = self.load_sequence(subject_path, start_frame, sequence_length)
            # Padding/Permuting logic as per other datasets
            background_value = y.flatten()[0]
            y = y.permute(0,4,1,2,3)
            # Assuming standard MNI or similar size, apply padding if needed. 
            # Checking previous datasets, padding is specific to registration.
            # Using generic padding for now, consistent with others.
            y = torch.nn.functional.pad(y, (6, 6, 0, 0, 9, 9), value=background_value) 
            y = y.permute(0,2,3,4,1)
            
            background_value = rand_y.flatten()[0]
            rand_y = rand_y.permute(0,4,1,2,3)
            rand_y = torch.nn.functional.pad(rand_y, (6, 6, 0, 0, 9, 9), value=background_value)
            rand_y = rand_y.permute(0,2,3,4,1)
            
            return {
                "fmri_sequence": (y, rand_y),
                "subject_name": subject,
                "target": target,
                "TR": start_frame,
                "sex": sex
            }
        else:
            y = self.load_sequence(subject_path, start_frame, sequence_length, num_frames)
            background_value = y.flatten()[0]
            y = y.permute(0,4,1,2,3)
            y = torch.nn.functional.pad(y, (6, 6, 0, 0, 9, 9), value=background_value)
            y = y.permute(0,2,3,4,1)
            
            return {
                "fmri_sequence": y,
                "subject_name": subject,
                "target": target,
                "TR": start_frame,
                "sex": sex,
            }

class Narratives(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _set_data(self, root, subject_dict):
        data = []
        img_root = os.path.join(root, 'img')
        
        for i, subject in enumerate(subject_dict):
            # Music dataset might not have target/sex in the same way, adapting generic placeholder
            if subject in subject_dict:
                 sex, target = subject_dict[subject]
            else:
                 sex, target = 0, 0 # Default if not in dict
            
            subject_path = os.path.join(img_root, subject)
            if not os.path.exists(subject_path): continue
            
            num_frames = len(glob.glob(os.path.join(subject_path, 'frame_*.pt')))
            session_duration = num_frames - self.sample_duration + 1
            
            for start_frame in range(0, session_duration, self.stride):
                data_tuple = (i, subject, subject_path, start_frame, self.stride, num_frames, target, sex)
                data.append(data_tuple)
        
        if self.train:
             self.target_values = np.array([tup[6] for tup in data]).reshape(-1, 1)
        return data

    def __getitem__(self, index):
        _, subject, subject_path, start_frame, sequence_length, num_frames, target, sex = self.data[index]
        
        if self.contrastive:
            y, rand_y = self.load_sequence(subject_path, start_frame, sequence_length)
            background_value = y.flatten()[0]
            y = y.permute(0,4,1,2,3)
            y = torch.nn.functional.pad(y, (6, 6, 0, 0, 9, 9), value=background_value) 
            y = y.permute(0,2,3,4,1)
            
            background_value = rand_y.flatten()[0]
            rand_y = rand_y.permute(0,4,1,2,3)
            rand_y = torch.nn.functional.pad(rand_y, (6, 6, 0, 0, 9, 9), value=background_value)
            rand_y = rand_y.permute(0,2,3,4,1)
            
            return {
                "fmri_sequence": (y, rand_y),
                "subject_name": subject,
                "target": target,
                "TR": start_frame,
                "sex": sex
            }
        else:
            y = self.load_sequence(subject_path, start_frame, sequence_length, num_frames)
            background_value = y.flatten()[0]
            y = y.permute(0,4,1,2,3)
            y = torch.nn.functional.pad(y, (6, 6, 0, 0, 9, 9), value=background_value)
            y = y.permute(0,2,3,4,1)
            
            return {
                "fmri_sequence": y,
                "subject_name": subject,
                "target": target,
                "TR": start_frame,
                "sex": sex,
            } 
