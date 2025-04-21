import numpy as np
import math, os, csv

import torch
import torch.nn as nn
import torch.utils.data as data
import librosa
import soundfile as sf
import cv2 as cv
import pyloudnorm as pyln
import scipy.io.wavfile as wav

import random
from .utils import DistributedSampler

def get_dataloader_text(args, partition):
    datasets = dataset_lip(args, partition)

    sampler = DistributedSampler(
        datasets,
        num_replicas=args.world_size,
        rank=args.local_rank) if args.distributed else None

    generator = data.DataLoader(datasets,
            batch_size = 1,
            shuffle = (sampler is None),
            num_workers = args.num_workers,
            persistent_workers = (args.num_workers !=0),
            drop_last=True, 
            sampler=sampler,
            collate_fn=custom_collate_fn)
    
    return sampler, generator

def custom_collate_fn(batch):
    a_mix, a_tgt, ref_tgt = batch[0]
    a_mix = torch.tensor(a_mix)
    a_tgt = torch.tensor(a_tgt) 
    ref_tgt = torch.tensor(ref_tgt) 
    return a_mix, a_tgt, ref_tgt

class dataset_lip(data.Dataset):
    def __init__(self, args, partition):
        self.minibatch =[]
        self.args = args
        self.partition = partition
        self.max_length = args.max_length
        self.audio_sr=args.audio_sr
        self.batch_size=args.batch_size
        self.speaker_no=args.speaker_no
        self.SNR =args.SNR
        self.mix_lst_path = args.mix_lst_path
        self.audio_direc = args.audio_direc
        self.ref_direc = args.reference_direc
        
        mix_lst=open(self.mix_lst_path).read().splitlines()
        mix_lst=list(filter(lambda x: x.split(',')[0]==partition, mix_lst))#[:200]
        mix_lst = sorted(mix_lst, key=lambda data: float(data.split(',')[-1]), reverse=True)
        
        start = 0
        while True:
            end = min(len(mix_lst), start + self.batch_size)
            self.minibatch.append(mix_lst[start:end])
            if end == len(mix_lst):
                break
            start = end

    def _audioread(self, path, min_length_audio, sampling_rate):
        data, fs = sf.read(path, dtype='float32')    
        if fs != sampling_rate:
            data = librosa.resample(data, orig_sr=fs, target_sr=sampling_rate)
        # if len(data.shape) >1:
        #     data = data[:, 0]    
        data = data[:min_length_audio]
        if data.shape[0] < min_length_audio:
            data = np.pad(data, (0, int(min_length_audio - data.shape[0])),mode = 'constant')
        return  data
    
    def __getitem__(self, index):
        mix_audios=[]
        tgt_audios=[]
        tgt_texts=[]
        
        batch_lst = self.minibatch[index]
        min_length_second = float(batch_lst[-1].split(',')[-1])      # truncate to the shortest utterance in the batch
        min_length_audio = math.floor(min_length_second*self.audio_sr)

        for line_cache in batch_lst:
            line=line_cache.split(',')

            # read tgt audio
            tgt_audio_path=self.audio_direc+line[4]
            a_tgt = self._audioread(tgt_audio_path, min_length_audio, self.audio_sr)

            target_power = np.linalg.norm(a_tgt, 2)**2 / a_tgt.size
            snr_0 = 10**(float(line[5])/20)
            
            
            int_audio_path=self.audio_direc+line[9]
            a_int = self._audioread(int_audio_path, min_length_audio, self.audio_sr)
            intef_power = np.linalg.norm(a_int, 2)**2 / a_int.size
            a_int *= np.sqrt(target_power/intef_power)
            
            snr_1 = 10**(float(line[10])/20)

            ref_audio_path = os.path.dirname(self.ref_direc+line[4])
            flac_files = [f for f in os.listdir(ref_audio_path) if f.endswith(".flac")]
            random_file = random.choice(flac_files)
            random_path = os.path.join(ref_audio_path, random_file)
            a_ref = self._audioread(random_path, min_length_audio, self.audio_sr)

            if self.args.speaker_no == 2:
                snr_factor_tgt = snr_0 / max(snr_0, snr_1)
                snr_factor_int = snr_1 / max(snr_0, snr_1)
                
                a_tgt *= snr_factor_tgt
                a_int *= snr_factor_int
                a_mix = a_tgt + a_int

            elif self.args.speaker_no == 3:
                c=2
                int_audio_path_2=self.audio_direc+line[c*4+1]+'/'+line[c*4+2]+'/'+line[c*4+3]+'.wav'
                a_int2 = self._audioread(int_audio_path_2, min_length_audio, self.audio_sr)
                intef_power_2 = np.linalg.norm(a_int2, 2)**2 / a_int2.size
                a_int2 *= np.sqrt(target_power/intef_power_2)
                snr_2 = 10**(float(line[c*4+4])/20)
                
                max_snr = max(snr_0, snr_1, snr_2)
                a_tgt /= max_snr
                a_int /= max_snr
                a_int2 /= max_snr

                a_tgt = a_tgt * snr_0
                a_int = a_int * snr_1
                a_int2 = a_int2 * snr_2

                a_mix = a_tgt + a_int + a_int2
            else:
                raise NameError('Wrong speaker_no selection')

            a_max_length = int(self.max_length*self.audio_sr)

            if min_length_audio > a_max_length:
                a_start=np.random.randint(0, (min_length_audio - a_max_length))
                a_mix = a_mix[a_start:a_start+a_max_length]
                a_tgt = a_tgt[a_start:a_start+a_max_length]
                a_int = a_int[a_start:a_start+a_max_length]
                a_ref = a_ref[a_start:a_start+a_max_length]

        
            a_tgt /= np.max(np.abs(a_tgt) + 1e-10)
            a_mix /= np.max(np.abs(a_mix) + 1e-10)
            a_int /= np.max(np.abs(a_int) + 1e-10)
            a_ref /= np.max(np.abs(a_ref) + 1e-10)

            t_spk = np.array([1 if line [6]=='F' else 2 ], dtype=np.float32)
            t_region = np.array([int(line[2])], dtype=np.float32)
            t_tpd = np.fromstring(line[1].strip('[]'), sep=' ', dtype=np.float32) 
            t_tgt = np.concatenate([t_spk, t_region,t_tpd])
            
            mix_audios.append(a_mix)
            tgt_audios.append(a_tgt)
            tgt_texts.append(t_tgt)

        return np.asarray(mix_audios, dtype=np.float32), np.asarray(tgt_audios, dtype=np.float32), np.asarray(tgt_texts, dtype=np.float32)


    def __len__(self):
        return len(self.minibatch)


