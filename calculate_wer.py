import difflib as dl

def calculate_wer(reference, hypothesis):
    """
    Calculates the Word Error Rate (WER) between two transcriptions.
    """
    wer = dl.SequenceMatcher(None, reference, hypothesis).ratio()
    return wer

def calculate_matching_metrics(cleaned_true_transcription, cleaned_model_transcription):
    """
    Calculates various matching metrics between the true transcription and the model transcription.
    """
    cleaned_true_transcription = str(cleaned_true_transcription)
    cleaned_model_transcription = str(cleaned_model_transcription)

    lcs = dl.SequenceMatcher(None, cleaned_true_transcription, cleaned_model_transcription).find_longest_match(0, len(cleaned_true_transcription), 0, len(cleaned_model_transcription)).size
    lcms = dl.SequenceMatcher(None, cleaned_true_transcription, cleaned_model_transcription).find_longest_match(0, len(cleaned_true_transcription), 0, len(cleaned_model_transcription)).size
    matching_ratio = dl.SequenceMatcher(None, cleaned_true_transcription, cleaned_model_transcription).ratio()
    matching_words_ratio = dl.SequenceMatcher(None, cleaned_true_transcription.split(), cleaned_model_transcription.split()).ratio()

    return lcs, lcms, matching_ratio, matching_words_ratio

def calculate_matching_words_ratio(true_transcription, model_transcription):
    true_words = true_transcription.split()
    model_words = model_transcription.split()
    
    matching_words_ratio = dl.SequenceMatcher(None, true_transcription.split(), model_transcription.split()).ratio()
    matcher = dl.SequenceMatcher(None, true_words, model_words)
    ratio = matcher.ratio()
    
    return ratio,
