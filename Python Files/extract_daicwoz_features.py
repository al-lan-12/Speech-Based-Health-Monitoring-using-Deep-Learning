import os
import torch
import torchaudio
import pandas as pd
from transformers import Wav2Vec2Model, Wav2Vec2Processor

#paths
AUDIO_DIR = "/scratch/users/k24015724/wav2vec_project/data/DAICWOZ_audio"
META_CSV = "/scratch/users/k24015724/wav2vec_project/data/DAICWOZ_meta/dev_split_Depression_AVEC2017.csv"
OUTPUT_DIR = "/scratch/users/k24015724/wav2vec_project/features/DAICWOZ"
os.makedirs(OUTPUT_DIR, exist_ok=True)

#metadata
df = pd.read_csv(META_CSV)
print(f"Found {len(df)} entries in metadata")

#Load model
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
model.eval().cuda()

#Constants
MAX_SECONDS = 5 
MAX_LENGTH = 16000 * MAX_SECONDS

#Feature extraction loop
for _, row in df.iterrows():
    pid = str(row["Participant_ID"])
    audio_path = os.path.join(AUDIO_DIR, f"{pid}_AUDIO.wav")
    output_path = os.path.join(OUTPUT_DIR, f"{pid}.pt")

    if not os.path.exists(audio_path):
        print(f"Error processing {pid}: Audio file not found")
        continue

    try:
        waveform, sample_rate = torchaudio.load(audio_path)
        
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)

        if waveform.size(1) > MAX_LENGTH:
            waveform = waveform[:, :MAX_LENGTH]

        inputs = processor(waveform.squeeze(0), sampling_rate=16000, return_tensors="pt", padding=True)
        input_values = inputs.input_values.cuda()

        with torch.no_grad():
            features = model(input_values).last_hidden_state  # [1, T, 768]

        torch.save(features.cpu(), output_path)
        print(f"Extracted features for {pid}")

    except Exception as e:
        print(f"Error processing {pid}: {e}")
