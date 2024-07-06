import pandas as pd
import os
import time
from faster_whisper import WhisperModel
import warnings
from transcription_cleaner import *
from calculate_wer import *

# Suppress specific FutureWarning about DataFrame concatenation
warnings.filterwarnings('ignore', message='The behavior of DataFrame concatenation with empty or all-NA entries is deprecated.*')

# Set KMP_DUPLICATE_LIB_OK environment variable
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def transcribe_audio(audio_file, model_size):
    model = WhisperModel(model_size, device='cuda', compute_type='float16')
    segments, info = model.transcribe(audio_file, beam_size=5)
    segments = list(segments)  # The transcription will actually run here.
    return segments, info

def print_transcription(transcription, model_size, time_to_transcribe):
    print('---------------')
    print(f"\033[1;34mTranscription: {transcription}\033[0m")
    print(f"\033[1;32mModel: {model_size}\033[0m")
    print(f"\033[1;31mTime to transcribe: {time_to_transcribe} seconds\033[0m")

def benchmark_models_to_dataframe(audio_file, models, true_transcription_path):
    # Initialize an empty DataFrame
    benchmark_results = pd.DataFrame(columns=['Model', 'Time to Transcribe', 'Transcription', 'WER'])
    
    with open(true_transcription_path, 'r') as file:
        true_transcription = file.read().replace('\n', ' ').strip()
    cleaned_true_transcription = clean_transcription(true_transcription)
    
    for model_size in models:
        startTime = time.time()
        segments, info = transcribe_audio(audio_file, model_size)
        endTime = time.time()
        time_to_transcribe = endTime - startTime
        transcription = ' '.join(segment[4] for segment in segments)
        cleaned_model_transcription = clean_transcription(transcription)
        wer = calculate_wer(cleaned_true_transcription, cleaned_model_transcription)
        
        # Append the results to the DataFrame
        new_row = pd.DataFrame({
            'Model': [model_size],
            'Time to Transcribe': [time_to_transcribe],
            'Transcription': [transcription],
            'WER': [wer]
        })
        benchmark_results = pd.concat([benchmark_results, new_row], ignore_index=True)
        
        print_transcription(transcription, model_size, time_to_transcribe)
    
    return benchmark_results

models = ['tiny', 'small', 'medium', 'large-v2', 'distil-large-v3']
audio_file = 'dataset/never running out of things to say is easy actually.mp3'
true_transcription_path = 'dataset/youtubeTranscript.txt'

# Use the modified function to get results in a DataFrame
results_df = benchmark_models_to_dataframe(audio_file, models, true_transcription_path)

# append the true transcription to the DataFrame
with open(true_transcription_path, 'r') as file:
    true_transcription = file.read().replace('\n', ' ').strip()
cleaned_true_transcription = clean_transcription(true_transcription)

results_df.loc[len(results_df)] = ['true', None, cleaned_true_transcription, None]

# save the DataFrame to a CSV file
results_df.to_csv('generated/benchmark_results.csv', index=False)

print(results_df)
