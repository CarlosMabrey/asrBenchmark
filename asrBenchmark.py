import os
import time
import pandas as pd
import warnings
from faster_whisper import WhisperModel
from transcription_cleaner import clean_transcription
from calculate_wer import calculate_wer, calculate_matching_words_ratio

# Suppress specific FutureWarning about DataFrame concatenation
warnings.filterwarnings('ignore', message='The behavior of DataFrame concatenation with empty or all-NA entries is deprecated.*')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def transcribe_audio(audio_file, model_size):
    model = WhisperModel(model_size, device='cuda', compute_type='float16')
    segments, _ = model.transcribe(audio_file, beam_size=5)
    return ' '.join(segment[4] for segment in segments)

def print_transcription(model_size, time_to_transcribe):
    print(f"---------------\n\033[1;32mModel: {model_size}\033[0m\n\033[1;31mTime to transcribe: {time_to_transcribe} seconds\033[0m")

def read_and_clean_transcription(file_path):
    with open(file_path, 'r') as file:
        return clean_transcription(file.read().replace('\n', ' ').strip())

def benchmark_models(audio_file, models, true_transcription, results_df=None):
    if results_df is None:
        results_df = pd.DataFrame(columns=['Model', 'Time to Transcribe', 'Transcription', 'WER', 'Matching Words Ratio'])
    
    for model_size in models:
        if model_size in results_df['Model'].values:
            print(f"Model {model_size} has already been transcribed. Skipping...")
            continue
        
        start_time = time.time()
        transcription = transcribe_audio(audio_file, model_size)
        time_to_transcribe = time.time() - start_time
        cleaned_transcription = clean_transcription(transcription)
        wer = calculate_wer(true_transcription, cleaned_transcription)
        matching_words_ratio = calculate_matching_words_ratio(true_transcription, cleaned_transcription)[0]
        
        results_df = results_df.append({
            'Model': model_size,
            'Time to Transcribe': time_to_transcribe,
            'Transcription': cleaned_transcription,
            'WER': wer,
            'Matching Words Ratio': matching_words_ratio
        }, ignore_index=True)
        
        print_transcription(model_size, time_to_transcribe)
    
    return results_df

def main():
    models = ['tiny', 'small', 'medium', 'large-v3', 'distil-large-v3', 'distil-medium.en', 'distil-small.en']
    audio_file = 'dataset/never running out of things to say is easy actually.mp3'
    true_transcription_path = 'dataset/youtubeTranscript.txt'
    true_transcription = read_and_clean_transcription(true_transcription_path)
    
    results_file_path = 'generated/benchmark_results.csv'
    results_df = pd.read_csv(results_file_path) if os.path.exists(results_file_path) else None
    
    results_df = benchmark_models(audio_file, models, true_transcription, results_df)
    
    if 'true' not in results_df['Model'].values:
        results_df.loc[len(results_df)] = ['true', None, true_transcription, None, None]
    
    results_df.to_csv(results_file_path, index=False)
    print(results_df)

if __name__ == "__main__":
    main()