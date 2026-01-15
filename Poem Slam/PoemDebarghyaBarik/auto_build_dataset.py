import json
import random
from pathlib import Path

import requests

OLLAMA_URL = "http://localhost:11434/api/generate"

# Use your base model for dataset generation
GEN_MODEL = "llama3.2:latest"

# Target file
DATASET_PATH = Path("dataset.jsonl")

# Languages as in your UI
LANGUAGES = [
    "English",
    "Deutsch (German)",
    "Hindi",
    "Русский (Russian)",
    "中文 (Chinese)",
]

POETIC_FORMS = [
    "Haiku-like (3 lines)",
    "Quatrain (4 lines)",
    "Couplets (2–4 rhymed lines)",
    "Sonnet (14 lines)",
    "Free form (up to 10 lines)",
]

MOODS = [
    "Romantic",
    "Melancholic",
    "Nature",
]

# Expanded word banks (100-150 words per language)
WORD_BANK = {
    "English": [
        # Nature
        "river", "leaf", "dawn", "shadow", "mountain", "forest", "ocean", "flame",
        "dust", "breath", "sky", "stone", "storm", "valley", "meadow", "tide",
        "wind", "snow", "rain", "cloud", "thunder", "lightning", "mist", "fog",
        "sunset", "sunrise", "twilight", "star", "moon", "sun", "earth", "grass",
        "flower", "tree", "branch", "root", "seed", "bloom", "petal", "thorn",
        "stream", "lake", "pond", "wave", "shore", "sand", "cliff", "hill",
        "garden", "field", "plain", "desert", "island", "glacier", "canyon",
        
        # Emotions & Abstract
        "love", "hope", "fear", "joy", "sorrow", "pain", "peace", "dream",
        "memory", "whisper", "silence", "echo", "song", "tale", "story", "word",
        "thought", "soul", "heart", "spirit", "longing", "desire", "passion",
        "grief", "tears", "laughter", "smile", "touch", "embrace", "kiss",
        "prayer", "wish", "promise", "secret", "truth", "lie", "faith", "doubt",
        
        # Time & Space
        "moment", "eternity", "hour", "season", "year", "century", "yesterday",
        "tomorrow", "forever", "never", "always", "sometimes", "distance",
        "journey", "path", "road", "bridge", "door", "window", "threshold",
        
        # Objects & Elements
        "mirror", "candle", "lantern", "feather", "shell", "pearl", "diamond",
        "iron", "silver", "gold", "bronze", "marble", "glass", "crystal",
        "book", "page", "ink", "quill", "letter", "sword", "shield", "crown",
        "ring", "chain", "thread", "cloth", "veil", "mask", "shadow", "light",
        "fire", "water", "air", "ice", "smoke", "ash", "embers", "spark"
    ],
    
    "Deutsch (German)": [
        # Natur
        "Regen", "Fenster", "Nacht", "Licht", "Herz", "Wald", "Fluss", "Stille",
        "Traum", "Abschied", "Himmel", "Stern", "Mond", "Sonne", "Wind", "Schnee",
        "Nebel", "Berg", "Tal", "Meer", "Welle", "Strand", "Baum", "Blatt",
        "Blume", "Rose", "Gras", "Wiese", "Feld", "Garten", "Erde", "Stein",
        "Fels", "Quelle", "Bach", "See", "Wolke", "Donner", "Blitz", "Sturm",
        
        # Emotionen & Abstraktes
        "Liebe", "Sehnsucht", "Schmerz", "Freude", "Trauer", "Hoffnung", "Angst",
        "Frieden", "Seele", "Geist", "Gedanke", "Erinnerung", "Vergessen",
        "Schweigen", "Stimme", "Lied", "Musik", "Wort", "Sprache", "Geschichte",
        "Märchen", "Wahrheit", "Lüge", "Glaube", "Zweifel", "Gebet", "Wunsch",
        "Versprechen", "Geheimnis", "Träne", "Lächeln", "Kuss", "Umarmung",
        
        # Zeit & Raum
        "Zeit", "Stunde", "Moment", "Augenblick", "Ewigkeit", "Gestern", "Morgen",
        "Heute", "Jahr", "Jahrhundert", "Jahreszeit", "Frühling", "Sommer",
        "Herbst", "Winter", "Dämmerung", "Morgenrot", "Mitternacht", "Weg",
        "Pfad", "Straße", "Brücke", "Tür", "Schwelle", "Reise", "Ferne", "Nähe",
        
        # Objekte & Elemente
        "Spiegel", "Kerze", "Laterne", "Feder", "Muschel", "Perle", "Diamant",
        "Silber", "Gold", "Eisen", "Glas", "Kristall", "Buch", "Seite", "Tinte",
        "Brief", "Schwert", "Schild", "Krone", "Ring", "Kette", "Faden", "Tuch",
        "Schleier", "Maske", "Schatten", "Feuer", "Wasser", "Luft", "Eis",
        "Rauch", "Asche", "Glut", "Funke", "Flamme"
    ],
    
    "Hindi": [
        # प्रकृति (Nature)
        "चाँद", "सपना", "स्पर्श", "नदी", "रात", "याद", "हवा", "धूप", "समुद्र",
        "पत्ते", "फूल", "पेड़", "जंगल", "पहाड़", "घाटी", "आकाश", "तारे", "सूरज",
        "बादल", "बारिश", "बर्फ", "कोहरा", "तूफान", "लहर", "किनारा", "रेत",
        "घास", "बगीचा", "खेत", "झील", "झरना", "पानी", "पत्थर", "चट्टान",
        
        # भावनाएँ (Emotions)
        "प्यार", "दर्द", "खुशी", "गम", "उम्मीद", "डर", "शांति", "सपने",
        "यादें", "आँसू", "मुस्कान", "हँसी", "चुंबन", "आलिंगन", "छुअन",
        "आत्मा", "दिल", "मन", "विचार", "भावना", "लालसा", "इच्छा", "प्रार्थना",
        "इच्छा", "वादा", "रहस्य", "सच", "झूठ", "विश्वास", "संदेह",
        
        # समय और स्थान (Time & Space)
        "समय", "पल", "क्षण", "अनंतता", "कल", "आज", "सदा", "कभी", "मौसम",
        "वसंत", "गर्मी", "सर्दी", "पतझड़", "सुबह", "शाम", "संध्या", "मध्यरात्रि",
        "रास्ता", "मार्ग", "सड़क", "पुल", "दरवाजा", "खिड़की", "दूरी", "यात्रा",
        
        # वस्तुएँ (Objects)
        "दर्पण", "मोमबत्ती", "लालटेन", "पंख", "सीप", "मोती", "हीरा", "चांदी",
        "सोना", "लोहा", "शीशा", "क्रिस्टल", "किताब", "पन्ना", "स्याही", "पत्र",
        "तलवार", "ढाल", "मुकुट", "अंगूठी", "जंजीर", "धागा", "कपड़ा", "घूंघट",
        "मुखौटा", "छाया", "रोशनी", "आग", "बर्फ", "धुआं", "राख", "चिंगारी",
        "ज्वाला", "प्रकाश", "अंधकार", "गीत", "संगीत", "शब्द", "कहानी"
    ],
    
    "Русский (Russian)": [
        # Природа
        "туман", "путь", "сердце", "река", "ветер", "память", "ночь", "звёзды",
        "лист", "окно", "небо", "луна", "солнце", "дождь", "снег", "облако",
        "гром", "молния", "буря", "лес", "гора", "долина", "море", "волна",
        "берег", "песок", "дерево", "цветок", "роза", "трава", "поле", "сад",
        "земля", "камень", "скала", "ручей", "озеро", "вода", "источник",
        
        # Эмоции и абстракция
        "любовь", "мечта", "боль", "радость", "печаль", "надежда", "страх",
        "покой", "душа", "дух", "мысль", "воспоминание", "забвение", "молчание",
        "голос", "песня", "музыка", "слово", "речь", "история", "сказка",
        "правда", "ложь", "вера", "сомнение", "молитва", "желание", "обещание",
        "тайна", "слеза", "улыбка", "смех", "поцелуй", "объятие", "прикосновение",
        
        # Время и пространство
        "время", "час", "миг", "мгновение", "вечность", "вчера", "завтра",
        "сегодня", "год", "век", "сезон", "весна", "лето", "осень", "зима",
        "рассвет", "закат", "сумерки", "полночь", "дорога", "тропа", "улица",
        "мост", "дверь", "порог", "путешествие", "даль", "близость", "расстояние",
        
        # Предметы и элементы
        "зеркало", "свеча", "фонарь", "перо", "ракушка", "жемчуг", "алмаз",
        "серебро", "золото", "железо", "стекло", "кристалл", "книга", "страница",
        "чернила", "письмо", "меч", "щит", "корона", "кольцо", "цепь", "нить",
        "ткань", "вуаль", "маска", "тень", "свет", "огонь", "лёд", "дым", "пепел",
        "уголь", "искра", "пламя", "заря", "мрак", "тишина", "эхо", "отражение"
    ],
    
    "中文 (Chinese)": [
        # 自然 (Nature)
        "山谷", "雨声", "绿叶", "星光", "河流", "夜色", "清风", "花瓣", "黎明",
        "云朵", "月亮", "太阳", "天空", "星星", "雨水", "雪花", "雾气", "雷声",
        "闪电", "风暴", "森林", "山峰", "海洋", "波浪", "沙滩", "草地", "田野",
        "花园", "大地", "石头", "岩石", "溪流", "湖泊", "泉水", "树木", "枝条",
        "根须", "种子", "花朵", "玫瑰", "荆棘", "秋叶", "春芽",
        
        # 情感与抽象 (Emotions & Abstract)
        "爱情", "梦想", "痛苦", "欢乐", "悲伤", "希望", "恐惧", "宁静", "灵魂",
        "心灵", "思想", "记忆", "遗忘", "沉默", "声音", "歌声", "音乐", "语言",
        "故事", "童话", "真相", "谎言", "信仰", "怀疑", "祈祷", "愿望", "诺言",
        "秘密", "眼泪", "微笑", "笑声", "亲吻", "拥抱", "触摸", "渴望", "激情",
        
        # 时间与空间 (Time & Space)
        "时光", "时刻", "瞬间", "永恒", "昨日", "明天", "今天", "岁月", "世纪",
        "季节", "春天", "夏日", "秋季", "冬天", "日出", "日落", "黄昏", "午夜",
        "道路", "小径", "街道", "桥梁", "门户", "窗户", "门槛", "旅程", "远方",
        "距离", "空间",
        
        # 物品与元素 (Objects & Elements)
        "镜子", "蜡烛", "灯笼", "羽毛", "贝壳", "珍珠", "钻石", "白银", "黄金",
        "铁器", "玻璃", "水晶", "书籍", "纸页", "墨水", "信件", "刀剑", "盾牌",
        "王冠", "戒指", "锁链", "丝线", "布匹", "面纱", "面具", "影子", "光芒",
        "火焰", "冰霜", "烟雾", "灰烬", "火花", "余烬", "曙光", "暗夜", "回声",
        "倒影", "露珠", "霜降", "晨曦", "暮色"
    ],
}


def call_ollama(model_name: str, prompt: str, num_predict: int = 180,
                temperature: float = 0.9, top_p: float = 0.95) -> str:
    """Call Ollama API to generate text."""
    options = {
        "temperature": float(temperature),
        "top_p": float(top_p),
        "num_predict": int(num_predict),
    }
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": options,
    }
    resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    return data.get("response", "").strip()


def form_instructions(form: str) -> str:
    """Generate form-specific instructions."""
    if form == "Haiku-like (3 lines)":
        return ("The poem MUST have exactly 3 short lines, like a haiku. "
                "Focus on imagery and simplicity.")
    elif form == "Quatrain (4 lines)":
        return ("The poem MUST have exactly 4 lines, like a quatrain. "
                "You may use gentle rhythm or rhyme, but structure is more important.")
    elif form == "Couplets (2–4 rhymed lines)":
        return ("The poem MUST have 2 to 4 lines, written as rhyming couplets "
                "(pairs of lines that rhyme as much as possible).")
    elif form == "Sonnet (14 lines)":
        return ("The poem MUST have exactly 14 lines, like a sonnet. "
                "You may use rhyme and a gentle rhythm, but focus on clear imagery and flow.")
    elif form == "Free form (up to 10 lines)":
        return ("The poem may have up to 10 lines, free form, without a strict rhyme scheme. "
                "Focus on natural flow and vivid imagery.")
    return "Write a short poem."


def lang_instruction(language: str) -> str:
    """Generate language-specific instructions."""
    if language == "Hindi":
        return "Write the poem in Hindi using Devanagari script only (no Latin letters / Hinglish)."
    elif language == "Deutsch (German)":
        return "Write only in German; do NOT mix other languages."
    elif language == "Русский (Russian)":
        return "Write only in Russian using Cyrillic script; do NOT use Latin letters."
    elif language == "中文 (Chinese)":
        return ("Write only in Chinese using Chinese characters (simplified is fine); "
                "do NOT use pinyin or Latin letters.")
    else:
        return "Write only in English; do NOT mix other languages."


def mood_phrase(mood: str) -> str:
    """Convert mood to descriptive phrase."""
    if mood == "Nature":
        return "nature-inspired, focusing on landscapes, seasons, and the natural world"
    return mood.lower()


def build_instruction(language: str, form: str, mood: str, words) -> str:
    """Build complete instruction prompt for the model."""
    words_str = ", ".join(words)
    fi = form_instructions(form)
    li = lang_instruction(language)
    mp = mood_phrase(mood)

    instruction = f"""You are a skilled poet.

Language: {language}
Poetic form: {form}
Mood: {mood}
Words: {words_str}

Task:
Write a poem in the specified language that follows the given poetic form and mood.
- {fi}
- The poem must naturally use ALL of the given words.
- The tone should clearly feel {mp}.
- {li}
- Do NOT explain anything, only output the poem text."""
    return instruction


def enforce_lines(poem: str, form: str) -> str:
    """Enforce line count based on poetic form."""
    lines = [l for l in poem.splitlines() if l.strip()]
    if form == "Haiku-like (3 lines)":
        lines = lines[:3]
    elif form == "Quatrain (4 lines)":
        lines = lines[:4]
    elif form == "Couplets (2–4 rhymed lines)":
        lines = lines[:4]
    elif form == "Sonnet (14 lines)":
        lines = lines[:14]
    elif form == "Free form (up to 10 lines)":
        lines = lines[:10]
    return "\n".join(lines)


def missing_words(poem: str, words) -> list[str]:
    """Check which required words are missing from the poem."""
    lp = poem.lower()
    missing = []
    for w in words:
        if w and w.lower() not in lp:
            missing.append(w)
    return missing


def main():
    """Generate synthetic poetry dataset."""
    # 10 samples per combination: 5 languages × 5 forms × 3 moods = 75 combos × 10 = 750
    samples_per_combo = 10

    print("Starting automatic dataset generation...")
    print(f"Target: 750 samples (10 per Language×Form×Mood combination)")
    print(f"Vocabulary: {len(WORD_BANK['English'])} English, "
          f"{len(WORD_BANK['Deutsch (German)'])} German, "
          f"{len(WORD_BANK['Hindi'])} Hindi, "
          f"{len(WORD_BANK['Русский (Russian)'])} Russian, "
          f"{len(WORD_BANK['中文 (Chinese)'])} Chinese words\n")

    with DATASET_PATH.open("a", encoding="utf-8") as f:
        for language in LANGUAGES:
            bank = WORD_BANK[language]
            print(f"\n{'='*60}")
            print(f"Processing: {language} ({len(bank)} vocabulary words)")
            print(f"{'='*60}")
            
            for form in POETIC_FORMS:
                for mood in MOODS:
                    for i in range(samples_per_combo):
                        # Choose 3 distinct words from expanded vocabulary
                        words = random.sample(bank, 3)
                        instruction = build_instruction(language, form, mood, words)

                        # Choose token budget based on form
                        if form == "Haiku-like (3 lines)":
                            num_predict = 40
                        elif form == "Quatrain (4 lines)":
                            num_predict = 80
                        elif form == "Couplets (2–4 rhymed lines)":
                            num_predict = 100
                        elif form == "Sonnet (14 lines)":
                            num_predict = 180
                        else:
                            num_predict = 120  # Free form

                        try:
                            poem = call_ollama(
                                model_name=GEN_MODEL,
                                prompt=instruction,
                                num_predict=num_predict,
                                temperature=0.8,
                                top_p=0.9,
                            )
                        except Exception as e:
                            print(f"❌ Error generating for {language}, {form}, {mood}: {e}")
                            continue

                        poem = poem.strip()
                        if not poem:
                            print(f"⚠️  Empty poem for {language}, {form}, {mood}, skipping.")
                            continue

                        # Enforce form lines
                        poem = enforce_lines(poem, form)

                        # Optional: skip if too many words missing
                        miss = missing_words(poem, words)
                        if len(miss) > 1:
                            print(f"⚠️  Too many missing words ({miss}) for {language}, {form}, {mood}, retrying...")
                            continue

                        example = {
                            "instruction": instruction,
                            "output": poem,
                        }
                        f.write(json.dumps(example, ensure_ascii=False) + "\n")

                        print(f"✓ Saved: {language} | {form} | {mood} | Sample {i+1}/{samples_per_combo}")

    print("\n" + "="*60)
    print("✅ Finished generating dataset.jsonl")
    print("="*60)


if __name__ == "__main__":
    main()