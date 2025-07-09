# SS-Libri Dataset Preparation

## 1. Introduction
Due to the lack of publicly available datasets containing both spatial and semantic information, we generate our own synthetic data. Specifically, we construct a spatialized multi-channel dataset, SS-Libri, by augmenting the single-channel recordings from the [LibriSpeech](https://www.openslr.org/12/) corpus with simulated spatial and directional cues.

## 2. Usage

### Step-by-Step Guide
1. **Download Dataset**
   > Download the train-clean-100, dev-clean and test-clean three datasets of LibriSpeech according to the [link](https://www.openslr.org/12/).
   > -LibriSppech
   > --train-clean-100
   > --dev-clean
   > --test-clean
2. **Speaker ID corresponds to gender**
   >Based on the downloaded SPEAKERS.TXT, obtain the correspondence between speaker IDs and genders,.
   ``` sh
    python id_sex.py
    ```
3. **Generate SS-Libri**
   >
   ``` sh
    python data.py
    ```
