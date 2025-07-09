# SS-Libri Dataset Preparation

## 1. Introduction
Due to the lack of publicly available datasets containing both spatial and semantic information, we generate our own synthetic data. Specifically, we construct a spatialized multi-channel dataset, SS-Libri, by augmenting the single-channel recordings from the [LibriSpeech](https://www.openslr.org/12/) corpus with simulated spatial and directional cues.

## 2. Usage

### Step-by-Step Guide
1. **Download Dataset**
   > Download the train-clean-100, dev-clean and test-clean three datasets of LibriSpeech according to the [link](https://www.openslr.org/12/).
   ``` sh
   -LibriSpeech/
      -train-clean-100/
      -dev-clean/
      -test-clean/
      -SPEAKERS.TXT
   ```
2. **Speaker ID corresponds to gender**
   > Based on the downloaded SPEAKERS.TXT, obtain the correspondence between speaker IDs and genders, output to speaker.csv.
   ``` sh
    python id_sex.py
    ```
3. **Multi-channel Speech Generation**
   > Using the original single channel dataset LibriSpeech and gpuRIR tools to generate multi-channel audio with spatial information.
   ``` sh
    python gpurir.py
    ```
   > Organize the files into the following structure
   ``` sh
     -SS-Libri/
      -train-clean-100/
      -dev-clean/
      -test-clean/
      -speaker.csv
      -multi_speech.csv
    ```
3. **Generate 2mix**
   > Using the gender corresponding file speaker.csv and the audio location information file multi_stpeech.csv, generate sex_spa.csv, where each group of speeches has at least one different gender and location.
   ``` sh
    python data.py
    ```
