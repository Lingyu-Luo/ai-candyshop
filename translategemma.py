import streamlit as st
import ollama

# Configuration: Map Language Names to ISO Codes
# The prompt requires both the language name and the code (e.g., English -> en)
# TranslateGemma supports 55 languages; here is a selection of common ones.
LANGUAGES = {
    "English": "en",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Italian": "it",
    "Portuguese": "pt",
    "Chinese (Simplified)": "zh",
    "Japanese": "ja",
    "Korean": "ko",
    "Russian": "ru",
    "Arabic": "ar",
    "Hindi": "hi",
    "Dutch": "nl",
    "Turkish": "tr",
    "Polish": "pl",
    "Vietnamese": "vi",
    "Thai": "th"
}


def construct_prompt(source_name, source_code, target_name, target_code, user_text):
    """
    Constructs the prompt exactly as specified in the TranslateGemma documentation.
    It requires specific wording and two blank lines before the text.
    """
    template = (
        f"You are a professional {source_name} ({source_code}) to {target_name} ({target_code}) translator. "
        f"Your goal is to accurately convey the meaning and nuances of the original {source_name} text "
        f"while adhering to {target_name} grammar, vocabulary, and cultural sensitivities. "
        f"Produce only the {target_name} translation, without any additional explanations or commentary. "
        f"Please translate the following {source_name} text into {target_name}:"
        "\n\n\n"  # Two blank lines (three newlines) as requested
        f"{user_text}"
    )
    return template


def translate_text(prompt):
    """Calls the local Ollama instance with the constructed prompt."""
    try:
        response = ollama.chat(
            model='translategemma',
            messages=[{'role': 'user', 'content': prompt}]
        )
        return response['message']['content']
    except Exception as e:
        return f"Error: {str(e)}. Make sure Ollama is running and 'translategemma' is pulled."


# Streamlit UI Layout

st.set_page_config(page_title="TranslateGemma UI", page_icon="🌐")

st.title("🌐 TranslateGemma Local Interface")
st.markdown("A wrapper for the open translation model running on Ollama.")

# Sidebar for Language Selection
with st.sidebar:
    st.header("Settings")

    # Source Language Selector
    source_lang_name = st.selectbox("Source Language", options=list(LANGUAGES.keys()), index=0)
    source_lang_code = LANGUAGES[source_lang_name]

    # Target Language Selector (default to Spanish for demo)
    target_lang_name = st.selectbox("Target Language", options=list(LANGUAGES.keys()), index=1)
    target_lang_code = LANGUAGES[target_lang_name]

    st.info(f"Translating **{source_lang_code}** ➝ **{target_lang_code}**")

# Main Content Area
col1, col2 = st.columns(2)

with col1:
    st.subheader(f"Input ({source_lang_name})")
    text_input = st.text_area("Enter text to translate:", height=300, placeholder="Type here...")

with col2:
    st.subheader(f"Output ({target_lang_name})")
    # Placeholder container to update later
    output_container = st.empty()
    output_container.text_area("Translation", height=300, disabled=True, placeholder="Translation will appear here...")

# Translation Trigger
if st.button("Translate", type="primary"):
    if not text_input.strip():
        st.warning("Please enter some text to translate.")
    else:
        with st.spinner("Translating..."):
            # 1. Build the prompt according to strict formatting rules
            full_prompt = construct_prompt(
                source_lang_name, source_lang_code,
                target_lang_name, target_lang_code,
                text_input
            )

            # 2. Call Ollama
            translation = translate_text(full_prompt)

            # 3. Update Output
            output_container.text_area("Translation", value=translation, height=300)
