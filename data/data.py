import csv
import os
import numpy as np
import librosa
import random
import pandas as pd
from tqdm import tqdm
import gpuRIR
import random
import numpy as np
import numpy.matlib
import gc
EPS = 1e-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import laion_clap
import soundfile as sf
sector_mapping = {
    0: (-22.5, 22.5, "Front"),
    1: (-67.5, -22.5, "Right Front"),
    2: (-112.5, -67.5, "Right"),
    3: (-157.5, -112.5, "Right Rear"),
    4: (157.5, 202.5, "Rear"),
    5: (112.5, 157.5, "Left Rear"),
    6: (67.5, 112.5, "Left"),
    7: (22.5, 67.5, "Left Front"),
}
sex_mapping = {
    1: "female",
    0: "male",
}
mic_pairs = [(0, 1), (0, 2), (0,3),(1,2),(1,3),(2, 3)]  

# Data directories
data_dirs = {
    "train": "./LibriSpeech/train-clean-100",
    "val": "./LibriSpeech/dev-clean",
    "test": "./LibriSpeech/test-clean"
}
out_dirs={
    "train": "./SS-Libri/train-clean-100",
    "val": "./SS-Libri/dev-clean",
    "test": "./SS-Libri/test-clean"
}

# Speaker metadata file
speaker_file = "./LibriSpeech/speaker.csv"

spk_df = pd.read_csv(speaker_file)

# Output CSV file
csv_filename = './SS-Libri/sex_spa.csv'

# CSV header
header = [
    'set','tpd','main_sector'
    'personid1', 'source1_path', 'source1_snr', 'source1_gender','source1_pos'
    'personid2', 'source2_path', 'source2_snr', 'source2_gender','source2_pos' 'length'
]

# Extract speaker ID and relative path
def get_personid_and_path(file_path):
    parts = file_path.split(os.sep)
    personid = parts[-3]  
    relative_path = os.path.join(*parts[-4:])  
    return personid, relative_path

# Get gender from speaker ID
def get_sex_by_id(file_path):
    speaker_id = os.path.basename(file_path).replace('.flac', '').split('-')[0]
    sex_value = spk_df.loc[spk_df['ID'] == int(speaker_id), 'SEX'].values
    return sex_value[0] if len(sex_value) > 0 else "Unknown"


class RIRmodel(nn.Module):
    def __init__(self):
        super(RIRmodel, self).__init__()
        #room    
        self.room_sz = [6, 6, 3]
        self.circle_radius = 0.05
        self.mic_positions = self._get_mic_positions()
        
        self.abs_weights = [0.9]*5+[0.5] # Absortion coefficient ratios of the walls    
        self.T60 = 0.1
        self.fs = 16000
        self.beta = gpuRIR.beta_SabineEstimation(self.room_sz, self.T60,abs_weights=self.abs_weights)
        self.Tdiff = gpuRIR.att2t_SabineEstimator(12, self.T60)
        self.Tmax = gpuRIR.att2t_SabineEstimator(40, self.T60)
        self.nb_img = gpuRIR.t2n(self.Tdiff, self.room_sz)
        
        self.mic_pattern = "card"
        self.orV_rcv = np.matlib.repmat(np.array([0, 1, 0]), 4, 1)

    def _get_mic_positions(self):
        mic_angles = np.linspace(0, 2 * np.pi, 4, endpoint=False)
        mic_positions = np.array([
            [self.room_sz[0] / 2 + self.circle_radius * np.cos(a),
             self.room_sz[1] / 2 + self.circle_radius * np.sin(a),
             1.7] for a in mic_angles
        ])
        return mic_positions

    def get_source_positions(self,flag):
        margin = 0.3
        min_height, max_height = 1.6, min(1.9, self.room_sz[2] - margin)
        if(flag==0):
            main_sector = random.randint(0, 7)
            possible_values = [i for i in range(8) if i != main_sector]
            second_sector = random.choice(possible_values)

        else:
            main_sector = random.randint(0, 7)
            second_sector = random.randint(0, 7)
        
        angle_step = 45  
        sector_angles = np.array([0, -45, -90, -135, 180, 135, 90, 45])  
        base_angle = np.radians(sector_angles[main_sector] + random.uniform(-22.5, 22.5))
        second_angle = np.radians(sector_angles[second_sector] + random.uniform(-22.5, 22.5))

        def polar_to_cartesian(angle):
            radius = np.random.uniform(0.5, min(self.room_sz[0], self.room_sz[1]) / 2 - margin)
            return [self.room_sz[0] / 2 + radius * np.cos(angle),
                    self.room_sz[1] / 2 + radius * np.sin(angle),
                    np.random.uniform(min_height, max_height)]

        source_positions = np.array([polar_to_cartesian(base_angle), polar_to_cartesian(second_angle)])
        
        assert (source_positions >= 0).all() and (source_positions <= self.room_sz).all(), "Sources are outside the room!"

        return source_positions, main_sector
      
    def compute_tpd_time_domain(self, mic_pairs, region, sound_speed=343.0):  
        mic_positions = np.array(self.mic_positions)  # (M, 3), 直接转为 NumPy 数组  

        tpd_features = []  
        for p1, p2 in mic_pairs:  
            delta_p = mic_positions[p1] - mic_positions[p2]  
            distance = np.linalg.norm(delta_p)  
            tpd = distance / sound_speed  
            tpd_samples = tpd * self.fs  
            tpd_features.append(tpd_samples)  

        tpd_features = np.array(tpd_features).flatten()  # 直接转为 NumPy 数组并展平  
        return tpd_features  
    def forward(self, tgt,int,flag):
        a_tgt, sr = librosa.load(tgt, sr=16000)
        a_int, sr = librosa.load(int, sr=16000)

        source_positions, main = self.get_source_positions(flag)
            
        RIRs1 = gpuRIR.simulateRIR(
                self.room_sz, self.beta, source_positions[0].reshape(1, 3), self.mic_positions, 
                self.nb_img, self.Tmax, self.fs, Tdiff=self.Tdiff, orV_rcv=self.orV_rcv, 
                mic_pattern=self.mic_pattern
        )
        RIRs2 = gpuRIR.simulateRIR(
                self.room_sz, self.beta, source_positions[1].reshape(1, 3), self.mic_positions, 
                self.nb_img, self.Tmax, self.fs, Tdiff=self.Tdiff, orV_rcv=self.orV_rcv, 
                mic_pattern=self.mic_pattern
        )
            # Perform convolution directly on GPU
        b1 = gpuRIR.gpuRIR_bind_simulator.gpu_conv(a_tgt[None, :], RIRs1).transpose(0, 2, 1)[0]
        b2 = gpuRIR.gpuRIR_bind_simulator.gpu_conv(a_int[None, :], RIRs2).transpose(0, 2, 1)[0]
        tpd= self.compute_tpd_time_domain(mic_pairs, None)

        return b1,b2,source_positions,tpd,main
            
rir_model = RIRmodel()

# Process dataset and ensure all files are used at least once
def process_dataset(set_type, data_dir):
    files = [os.path.join(root, f) for root, _, files in os.walk(data_dir) for f in files if f.endswith(".flac")]
    # Shuffle files
    random.shuffle(files)
    # Store selected pairs
    selected_pairs = []
    used_files = set()
    
    genders = [get_sex_by_id(f) for f in files]
    Fa=0
    Ma=0
    for  g in genders:
        if g == "M": Ma+=1
        else: Fa+=1
    while len(used_files) < len(files):
        remaining_files = [f for f in files if f not in used_files]
    
        if len(remaining_files) < 2 or Fa==0 or Ma==0:
            print("end")
            break  # Exit if not enough files remain
        file1, file2 = random.sample(remaining_files, 2)
       
        sex1 = get_sex_by_id(file1)
        sex2 = get_sex_by_id(file2)
        snr_value = np.random.uniform(-3, 3)
      
        if sex1 == sex2:continue
        
        # Mark files as used
        used_files.add(file1)
        used_files.add(file2)
        Fa-=1
        Ma-=1
        flag=0
        personid1, path1 = get_personid_and_path(file1)
        personid2, path2 = get_personid_and_path(file2)
        b1,b2,source_positions,tpd,main=rir_model(file1,file2,flag)

        output_filepath1 = os.path.join("SS-Libri", path1)
        output_filepath2 = os.path.join("SS-Libri", path2)
        sf.write(output_filepath1, b1, 16000, format='FLAC')
        sf.write(output_filepath2, b2, 16000, format='FLAC')
        # Get duration
        duration1 = librosa.get_duration(path=file1)
        duration2 = librosa.get_duration(path=file2)
        length = min(duration1, duration2)
        # Save data
        selected_pairs.append([
            set_type, tpd,main,
            personid1, path1, "0", sex1,source_positions[0],
            personid2, path2, snr_value, sex2, source_positions[1],length
        ])
        used_files.add(file1)
        used_files.add(file2)
        if len(used_files) % 30 == 0:
            print(len(used_files), len(files), set_type)

    return selected_pairs
# Write data to CSV
with open(csv_filename, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(header)

    for set_type, data_dir in data_dirs.items():
        print(f"Processing {set_type} set...")
        dataset_pairs = process_dataset(set_type, data_dir)
        writer.writerows(dataset_pairs)

print(f"CSV file saved to {csv_filename}")
