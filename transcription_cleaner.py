def clean_transcription(txt_content):
    """
    Cleans the transcription text by removing special characters, making all characters lowercase,
    removing extra spaces, and removing uppercase characters.
    """
    cleaned = ' '.join(txt_content.split())
    cleaned = ''.join(e for e in cleaned if e.isalnum() or e.isspace())
    cleaned = cleaned.lower()
    cleaned = ' '.join(cleaned.split())  # Remove extra spaces
    cleaned = ''.join(e for e in cleaned if not e.isupper())  # Remove uppercase characters
    return cleaned