from google.cloud import translate

def translate_text_v3(text_to_translate: str, project_id: str, target_language_code: str):
    """
    Translates text using the Advanced (V3) client.
    
    V3 requires the Project ID to construct the parent resource path.
    """
    # 1. Initialize the V3 client (uses GOOGLE_APPLICATION_CREDENTIALS for auth)
    client = translate.TranslationServiceClient()

    # 2. Define the resource path: projects/YOUR_PROJECT_ID/locations/global
    # 'global' is the default and only region for general translation.
    parent = f"projects/medconnect-479308/locations/global"

    # 3. Build the request object
    response = client.translate_text(
        parent=parent,
        contents=[text_to_translate],
        source_language_code = 'ha',
        target_language_code=target_language_code,
        mime_type="text/plain", 
    )

    # 4. Process and print the result
    print(f"Original Text: {text_to_translate}")
    
    # V3 returns a list of translations, one for each item in contents
    for translation in response.translations:
        # Note: V3 result objects are slightly different from V2
        print(f"Translation: {translation.translated_text}")
        print(f"Detected Source Language: {translation.detected_language_code}")

# --- Call the function ---
# **REPLACE THIS WITH YOUR ACTUAL GOOGLE CLOUD PROJECT ID**
YOUR_PROJECT_ID = "medconnect-479308" 

text_to_translate = """PATIENT: ina kwana

DOCTOR: Sannu! Yaya kake?

PATIENT: lafia kalo

DOCTOR: Lafiya lau.

PATIENT: ka tashi lafia?

DOCTOR: Lafiya lau, nagode."""
target_lang = "en"  # Spanish

print("--- V3 Translation Result ---")
translate_text_v3(text_to_translate, YOUR_PROJECT_ID, target_lang)