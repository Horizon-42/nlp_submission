import requests
import gradio as gr

# ---- CONFIG ----

OLLAMA_URL = "http://localhost:11434/api/generate"

# Main poetry model (later this will be your fine-tuned model)
POETRY_MODEL = "llama3.2:latest"

# You can use the same model for translation, or switch to qwen3:4b if you want
TRANSLATION_MODEL = "llama3.2:latest"


# ---- HELPER FUNCTIONS ----

def call_ollama(model_name: str, prompt: str, num_predict: int | None = None,
                temperature: float = 0.9, top_p: float = 0.95) -> str:
    """Call a local Ollama model and return the response text or raise an error."""
    options = {
        "temperature": float(temperature),
        "top_p": float(top_p),
    }
    if num_predict is not None:
        options["num_predict"] = int(num_predict)

    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": options,
    }

    response = requests.post(OLLAMA_URL, json=payload, timeout=120)
    response.raise_for_status()
    data = response.json()
    return data.get("response", "").strip()


def translate_words_if_needed(words, language):
    """
    Optionally translate the input words into the target language
    so the poem can stay monolingual.
    For English, we keep words as-is.
    For Deutsch/Hindi/Russian/Chinese, we ask the translation model.
    """
    if not words:
        return words

    # Map UI label to language name for the prompt
    if language == "Deutsch (German)":
        target_lang = "German"
    elif language == "Hindi":
        target_lang = "Hindi (use Devanagari script only)"
    elif language == "Русский (Russian)":
        target_lang = "Russian (use Cyrillic script only)"
    elif language == "中文 (Chinese)":
        target_lang = "Chinese (use simplified Chinese characters only)"
    else:
        # English or anything else: no translation
        return words

    words_str = ", ".join(words)
    prompt = f"""
Translate the following words into {target_lang}.
Return only a comma-separated list of the translated words, in the same order,
with no explanations and no extra text.

Words: {words_str}
"""

    try:
        translated_text = call_ollama(
            model_name=TRANSLATION_MODEL,
            prompt=prompt,
            num_predict=80,
            temperature=0.3,  # more deterministic for translation
            top_p=0.8,
        )
        # Split back into a list
        translated_words = [w.strip() for w in translated_text.split(",") if w.strip()]
        # If splitting fails badly, fall back to original words
        if len(translated_words) != len(words):
            return words
        return translated_words
    except Exception:
        # If translation fails, just use original words
        return words


def clean_word(raw: str) -> str:
    """Clean a single 'word' input: strip, keep first token, limit length."""
    raw = (raw or "").strip()
    if not raw:
        return ""
    # Take only the first space-separated token to avoid whole sentences
    first = raw.split()[0]
    # Hard limit length so users don't paste long sentences
    return first[:30]


def enforce_form_lines(poem: str, form: str) -> str:
    """Truncate / lightly enforce line counts based on poetic form."""
    lines = [line for line in poem.splitlines() if line.strip()]

    if form == "Haiku-like (3 lines)":
        lines = lines[:3]
    elif form == "Quatrain (4 lines)":
        lines = lines[:4]
    elif form == "Couplets (2–4 rhymed lines)":
        if len(lines) > 4:
            lines = lines[:4]
    elif form == "Sonnet (14 lines)":
        lines = lines[:14]
    elif form == "Free form (up to 10 lines)":
        if len(lines) > 10:
            lines = lines[:10]

    return "\n".join(lines)


def missing_words(poem: str, words) -> list[str]:
    """Return list of words that do not appear in the poem (case-insensitive substring check)."""
    lower_poem = poem.lower()
    missing = []
    for w in words:
        if w and w.lower() not in lower_poem:
            missing.append(w)
    return missing


# ---- PROMPT BUILDING ----

def build_prompt(words, language, form, mood):
    if not words:
        return "ERROR:NO_WORDS"

    # Instructions per poetic form
    if form == "Haiku-like (3 lines)":
        form_instructions = (
            "The poem MUST have exactly 3 short lines, like a haiku. "
            "Focus on imagery and simplicity."
        )
    elif form == "Quatrain (4 lines)":
        form_instructions = (
            "The poem MUST have exactly 4 lines, like a quatrain. "
            "You may use gentle rhythm or rhyme, but structure is more important."
        )
    elif form == "Couplets (2–4 rhymed lines)":
        form_instructions = (
            "The poem MUST have 2 to 4 lines, written as rhyming couplets "
            "(pairs of lines that rhyme as much as possible)."
        )
    elif form == "Sonnet (14 lines)":
        form_instructions = (
            "The poem MUST have exactly 14 lines, like a sonnet. "
            "You may use rhyme and a gentle rhythm, but focus on clear imagery and flow."
        )
    elif form == "Free form (up to 10 lines)":
        form_instructions = (
            "The poem may have up to 10 lines, free form, without a strict rhyme scheme. "
            "Focus on natural flow and vivid imagery."
        )
    else:
        form_instructions = "Write a short poem."

    words_str = ", ".join(words)

    # Extra instruction for script & language purity
    if language == "Hindi":
        lang_instruction = (
            "Write the poem in Hindi using Devanagari script only "
            "(no Latin letters / Hinglish)."
        )
    elif language == "Deutsch (German)":
        lang_instruction = "Write only in German; do NOT mix other languages."
    elif language == "Русский (Russian)":
        lang_instruction = (
            "Write only in Russian using Cyrillic script; do NOT use Latin letters."
        )
    elif language == "中文 (Chinese)":
        lang_instruction = (
            "Write only in Chinese using Chinese characters (simplified is fine); "
            "do NOT use pinyin or Latin letters."
        )
    else:
        lang_instruction = "Write only in English; do NOT mix other languages."

    # Mood explanation for the model
    if mood == "Nature":
        mood_phrase = "nature-inspired, focusing on landscapes, seasons, and the natural world"
    else:
        mood_phrase = mood.lower()

    prompt = f"""
You are a skilled poet.

Language: {language}
Poetic form: {form}
Mood: {mood}
Words: {words_str}

Task:
Write a poem in the specified language that follows the given poetic form and mood.
- {form_instructions}
- The poem must naturally use ALL of the given words.
- The tone should clearly feel {mood_phrase}.
- {lang_instruction}
- Do NOT explain anything, only output the poem text.
"""
    return prompt.strip()


# ---- MAIN GENERATION FUNCTION ----

def generate_poem(word1, word2, word3, language, form, mood, temperature, top_p):
    # Clean and collect words
    cleaned = [clean_word(w) for w in [word1, word2, word3]]
    words = [w for w in cleaned if w]

    if not words:
        return "Please enter at least one non-empty word."

    # Translate words into target language if needed
    translated_words = translate_words_if_needed(words, language)

    # Build prompt
    prompt = build_prompt(translated_words, language, form, mood)
    if prompt == "ERROR:NO_WORDS":
        return "Please enter at least one valid word."

    # Choose max token budget based on form
    if form == "Haiku-like (3 lines)":
        max_tokens = 40
    elif form == "Quatrain (4 lines)":
        max_tokens = 80
    elif form == "Couplets (2–4 rhymed lines)":
        max_tokens = 100
    elif form == "Sonnet (14 lines)":
        max_tokens = 180
    elif form == "Free form (up to 10 lines)":
        max_tokens = 120
    else:
        max_tokens = 80

    try:
        poem = call_ollama(
            model_name=POETRY_MODEL,
            prompt=prompt,
            num_predict=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
    except requests.ConnectionError:
        return "Could not connect to Ollama. Please make sure the Ollama app is running."
    except Exception as e:
        return f"Error talking to the model: {e}"

    if not poem:
        return "Model returned an empty response."

    # Enforce line structure
    poem = enforce_form_lines(poem, form)

    # Check if words are present
    missing = missing_words(poem, translated_words)
    if missing:
        poem += "\n\n[Note: the model may have missed these word(s): " + ", ".join(missing) + "]"

    return poem


# ---- FRONTEND (GRADIO UI) ----

languages = [
    "English",
    "Deutsch (German)",
    "Hindi",
    "Русский (Russian)",
    "中文 (Chinese)",
]

poetic_forms = [
    "Haiku-like (3 lines)",
    "Quatrain (4 lines)",
    "Couplets (2–4 rhymed lines)",
    "Sonnet (14 lines)",
    "Free form (up to 10 lines)",
]

moods = [
    "Romantic",
    "Melancholic",
    "Nature",
]

with gr.Blocks() as demo:
    gr.Markdown(
        "Local Structured Poetry Generator\n"
        "Runs fully on your Mac using Ollama.\n\n"
        "- Choose 3 seed words\n"
        "- Pick language, poetic form, and mood\n"
        "- Adjust creativity and vocabulary richness\n"
    )

    with gr.Row():
        word1 = gr.Textbox(label="Word 1")
        word2 = gr.Textbox(label="Word 2")
        word3 = gr.Textbox(label="Word 3")

    with gr.Row():
        language = gr.Dropdown(choices=languages, value="English", label="Language")
        form = gr.Dropdown(
            choices=poetic_forms,
            value="Haiku-like (3 lines)",
            label="Poetic Form",
        )
        mood = gr.Dropdown(
            choices=moods,
            value="Romantic",
            label="Mood",
        )

    with gr.Row():
        temperature = gr.Slider(
            minimum=0.3,
            maximum=1.1,
            value=0.9,
            step=0.05,
            label="Creativity (temperature)",
            info="Lower = safer, higher = more creative",
        )
        top_p = gr.Slider(
            minimum=0.7,
            maximum=1.0,
            value=0.95,
            step=0.05,
            label="Vocabulary richness (top_p)",
            info="Lower = simpler words, higher = richer vocabulary",
        )

    generate_btn = gr.Button("Generate Poem")
    output = gr.Textbox(label="Poem", lines=16)

    generate_btn.click(
        fn=generate_poem,
        inputs=[word1, word2, word3, language, form, mood, temperature, top_p],
        outputs=output,
    )

demo.launch()
