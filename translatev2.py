from google.cloud import translate_v2 as translate

def translate_text_v2(text_to_translate, target_language_code):
    """
    Translates text into the target language using the Basic (V2) client.

    The client automatically uses the credentials set in the
    GOOGLE_APPLICATION_CREDENTIALS environment variable.
    """
    # Create the v2 client
    translate_client = translate.Client()

    # The text to translate can be a single string or a list of strings
    if isinstance(text_to_translate, bytes):
        text_to_translate = text_to_translate.decode("utf-8")

    # Perform the translation
    result = translate_client.translate(
        text_to_translate, 
        target_language=target_language_code,
        source_language= "ig"
    )['translatedText']

    print(f"Original Text: {result}")
    #print(f"Translation: {result['translatedText']}")
    #print(f"Detected Source Language: {result['detectedSourceLanguage']}")

# --- Call the function ---
# Target language codes are ISO 639-1 identifiers (e.g., 'fr' for French, 'es' for Spanish)
text_to_translate = "Enkịta ngwangwa aja aja na-awụli elu n'elu nkịtà umengwụ."#"The quick brown fox jumps over the lazy dog."
target_lang = "en"  # Spanish

print("--- Translation Result ---")
translate_text_v2(text_to_translate, target_lang)