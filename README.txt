# README – MSc Project Source Code

Project Title: Understanding the Robustness of Speech-Based Analyses for Remote Health Assessment  
Author: Allan Joe Achangudan   

---

##Project Summary

This project investigates the robustness and transferability of deep learning models for speech-based mental health monitoring. A dual-head classifier using Wav2Vec2-based audio features was trained on the IEMOCAP dataset and evaluated on DAIC-WOZ in a zero-shot transfer learning setting. The core research question is:  
**"How transferable are models trained on emotion-rich datasets like IEMOCAP to clinically relevant datasets such as DAIC-WOZ?"**

---

##Contents of Archive

/train_dualhead_iemocap.py         → Trains the dual-head emotion + depression classifier on IEMOCAP  
/precompute_iemocap_mean_features.py → Extracts and stores averaged Wav2Vec2 embeddings per utterance for IEMOCAP  
/extract_daicwoz_features.py       → Performs batch feature extraction on DAIC-WOZ audio files using Wav2Vec2  
/infer_daicwoz_binary_finetuned.py → Runs inference on DAIC-WOZ using IEMOCAP-trained model; outputs metrics  
/model.py                          → Defines the DualHeadClassifier with shared and task-specific heads  

---

##File Descriptions

### 1. `train_dualhead_iemocap.py`
- Trains a classifier with two output heads (emotion and binary depression)
- Input: Precomputed `.pt` features from IEMOCAP
- Output: Saved model checkpoint `classifier_final.pt`

### 2. `precompute_iemocap_mean_features.py`
- Loads IEMOCAP `.wav` files
- Extracts mean Wav2Vec2 embeddings
- Outputs feature `.pt` files and a master CSV

### 3. `extract_daicwoz_features.py`
- Runs offline feature extraction on DAIC-WOZ `.wav` files
- Saves speaker-wise `.pt` files

### 4. `infer_daicwoz_binary_finetuned.py`
- Loads DAIC-WOZ `.pt` features and a pretrained classifier
- Computes accuracy, precision, recall, F1
- Prints a classification report and confusion matrix

### 5. `model.py`
- Contains `DualHeadClassifier` class
- Shared feature encoder → separate emotion and binary heads

---

##Requirements

You can run the code using the following environment setup:

- Python 3.8
- torch==1.10.2
- torchaudio==0.10.2
- transformers==4.x
- pandas, numpy, sklearn
- tqdm, matplotlib, seaborn


##Notes

- This project simulates a zero-shot transfer learning scenario: training is performed on IEMOCAP only; DAIC-WOZ is used for evaluation only.
- Final model: `classifier_final.pt`
- DAIC-WOZ labels: binary (PHQ8_Binary = 0 for non-depressed, 1 for depressed)

---

End of README
