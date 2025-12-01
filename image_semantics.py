import io
import re
from collections import Counter
from functools import lru_cache

import numpy as np
import requests
import torch
from PIL import Image
from transformers import (
    BlipForConditionalGeneration,
    BlipProcessor,
    CLIPProcessor,
    CLIPModel,
)
from sentence_transformers import SentenceTransformer

from config import HEADERS

# -------------------------------------------------------------------
# text helpers & NLP utilities
# -------------------------------------------------------------------

#Stopwords to ignore when building keywords
STOPWORDS = {
    "a","an","and","the","or","but","if","then","else","when","while","for","to","of","in","on","at","by","with",
    "from","about","as","is","am","are","was","were","be","been","being","it","its","this","that","these","those",
    "do","did","does","doing","done","can","could","should","would","will","just","so","very","really","much","many",
    "more","most","less","least","here","there","where","why","how","what","who","whom","which","because","than",
    "into","out","off","up","down","over","under","again","once","ever","never","always","also","too","not","no","yes",
    "ok","okay","gt","lt","amp","etc","are","were","your","you","me","my","our","we","they","them","their","i"
}

#Regex for tokenization
TOKEN_REGEX = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-']{1,}")


def tokenize(text: str):
    return [m.group(0) for m in TOKEN_REGEX.finditer(text or "")]


#Normalize tokens (lowercase, remove some characters, collapse repeated letters)
def normalize_token(tok: str) -> str:
    t = tok.lower().strip("-' ")
    return re.sub(r"(.)\1{2,}", r"\1\1", t)


#Remove duplicates 
def remove_duplicates(seq):
    seen, out = set(), []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


# -------------------------------------------------------------------
# BLIP + CLIP + SentenceTransformer
# -------------------------------------------------------------------
#Use gpu for processing if available, otherwise CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BLIP = BLIP_PROC = CLIP = CLIP_PROC = None

SENT_EMB_MODEL = None
CAT_EMB = None  # for subreddit categories
CANONICAL_EMB = None  # for generalized meme tags


#Load blip/clip models for image processing
def load_vision_models():
    global BLIP, BLIP_PROC, CLIP, CLIP_PROC
    if BLIP is None:
        BLIP_PROC = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        BLIP = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        ).to(DEVICE)
        BLIP.eval()
    if CLIP is None:
        CLIP_PROC = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        CLIP = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
        CLIP.eval()

#Category descriptions for subreddit classifications
CATEGORY_DESCRIPTORS = {
    "political":  "elections, public policy, governments, politicians, geopolitics",
    "humor":      "funny content, jokes, memes, comedy, satire",
    "technology": "software, programming, gadgets, AI, computing, engineering",
    "gaming":     "video games, franchises, esports, consoles, pc gaming, game tips",
    "art":        "drawing, painting, design, illustration, digital art, animation",
    "finance":    "investing, stocks, crypto, personal finance, markets",
    "sports":     "teams, leagues, matches, players, football, basketball, hockey",
    "education":  "university, college, studying, research, science, academia",
    "lifestyle":  "fitness, health, food, travel, fashion, cooking, wellness",
    "music":      "songs, albums, artists, bands, genres like rock pop hip hop r&b edm k-pop; concerts, lyrics, production",
    "other":      "general topics that do not fit the above",
}

# generalized meme tags
MEME_TAGS = [
    # Self-Deprecating Memes 
    "self-deprecating",
    "personal failure",
    "social awkwardness",
    "exhausted",

    # Doomer / Reactionary Memes 
    "anti-societal",
    "doom-predictions",
    "anti-doomer",
    "genre-reactionary",

    # Hobby / Niche Memes 
    "hobby or niche meme",
    "fandom in-joke",
    "work or school niche meme",
    "internet niche meme",

    # Relatable Memes 
    "personal stories and situations",
    "mood relatability",
    "everyday inconveniences",
    "relationships and social life relatability",

    # Misc / Meta Memes 
    "referential meme",
    "abstract or bizarre humour",
    "chaotic low-effort meme",
    "meme culture",

    # Geographic Memes 
    "country or region humour",
    "city-specific meme",
    "global north vs global south",
    "regional stereotypes",

    # Stereotype Memes 
    "national or ethnic stereotypes",
    "gender stereotypes",
    "job or class stereotypes",
    "age group stereotypes",

    # Political Memes 
    "domestic politics meme",
    "geopolitics and international relations",
    "ideological or partisan conflict",
    "policy debate humour",

    # Animal Memes 
    "wholesome animal meme",
    "funny pet behaviour",
    "animals acting like humans",
    "animal reaction meme",

    # Cultural / Historical Memes 
    "historical reference meme",
    "historical figure in modern setting",
    "traditions and heritage humour",
    "modern vs historical comparison",

    # Pop Culture Memes 
    "tv and movie reference",
    "music and celebrity culture",
    "anime and fandom reference",
    "gaming pop culture joke",

    # Cultural Memes 
    "culture war discourse",
    "internet culture commentary",
    "cross-cultural misunderstanding",
    "chronically online behaviour",
]


MEME_DESCRIPTIONS = [
    # Self-Deprecating 
    "memes where the creator mocks their own flaws or failures; self-deprecating humour (e.g., joking about ‘me trying to be productive and immediately taking a nap’)",
    "memes about messing up, not being good enough, or personal incompetence (like comparing yourself to a computer that keeps crashing)",
    "memes about being socially awkward, embarrassing oneself, or cringe behaviour (such as rehearsing a greeting and still saying the wrong thing)",
    "memes about burnout, feeling drained, or being exhausted with life (like using a low-battery icon to describe your energy)",

    # Doomer / Reactionary 
    "memes critical of society, systems, or culture; anti-societal sentiment (e.g., someone staring out a window saying ‘everything is broken’)",
    "memes predicting catastrophe, societal collapse, or pessimistic futures (like joking that ‘it’s 2030 and I still haven’t fixed my sleep schedule’)",
    "memes mocking extreme or overly negative doomer attitudes (such as calling a tiny inconvenience ‘the downfall of civilization’)",
    "memes reacting to online trends, formats, genres, or commentary culture (e.g., eye-rolling at yet another TikTok clone trend)",

    # Hobby / Niche 
    "memes tied to a specific hobby such as coding, cars, music, etc. (like ‘fix one bug, create three more’ for programmers)",
    "memes requiring knowledge of a specific fandom, show, or game (such as inside jokes only anime or Marvel fans would get)",
    "memes only understood by people in certain jobs, majors, or workplaces (e.g., scientists joking about peer review pain)",
    "memes relying on technical or internet-lore knowledge (like referencing the classic ‘GPU go brrr’ meme)",

    # Relatable 
    "memes framed as relatable personal stories or everyday experiences (e.g., checking the fridge 3 times expecting new food)",
    "memes expressing moods, vibes, or emotional relatability (like someone wrapped in blankets labelled ‘today’s energy’)",
    "memes about daily frustrations, minor inconveniences, or small struggles (such as dropping your phone on your face in bed)",
    "memes about friendships, relationships, dating, or social dynamics (e.g., the friend who just walks into your house and eats your snacks)",

    # Misc / Meta 
    "memes referencing specific media, creators, or internet trends (like joking about the narrator voice from documentary TikToks)",
    "memes built on surreal, bizarre, or abstract humour (e.g., a floating banana with no explanation)",
    "memes intentionally low-effort, chaotic, or nonsense for comedic effect (such as blurry screenshots with random text)",
    "memes about meme formats, meme history, or meta-meme commentary (like complaining that the ‘Distracted Boyfriend’ format won’t die)",

    # Geographic 
    "memes about national or regional identity, cultural differences, or geography (e.g., roasting your country’s unpredictable weather)",
    "memes tied to specific cities, neighbourhoods, or local in-jokes (like ‘Toronto traffic ages me 5 years’)",
    "memes contrasting global north vs south, rich vs poor countries (such as exaggerated split-screen comparisons)",
    "memes playing on stereotypes of different regions or locales (e.g., Americans reacting to the word ‘kilometres’)",

    # Stereotype 
    "memes using national or ethnic stereotypes as the basis of the joke (like someone bragging that spicy food is ‘nothing’)",
    "memes about gender expectations or stereotypical male/female behaviour (e.g., guys refusing to ask for directions)",
    "memes about class, jobs, professions, or workplace stereotypes (such as ‘this meeting could’ve been an email’)",
    "memes based on stereotypes of age groups such as boomers or zoomers (e.g., boomers calling every console ‘the Nintendo’)",

    # Political 
    "memes about domestic elections, politicians, or political debates (like checking polls as if they’re sports scores)",
    "memes referencing geopolitics, foreign policy, or global conflicts (such as simplified maps with dramatic labels)",
    "memes about ideological clashes such as left vs right politics (e.g., stick figures screaming over basic facts)",
    "memes joking about public policy issues or government decisions (like ‘budget cuts removed the floor’)",

    # Animal 
    "memes using cute or wholesome animals for feel-good humour (e.g., a puppy labelled ‘emotional support chaos’)",
    "memes about chaotic, silly, or mischievous behaviour of pets (like a cat knocking something over for no reason)",
    "memes where animals are given human roles or emotions (such as a dog at a desk titled ‘first day at work’)",
    "memes using animals as reaction images or emotional expressions (e.g., an owl representing pure confusion)",

    # Cultural / Historical 
    "memes referencing historical events, wars, or major past eras (like Romans dramatically reacting to small problems)",
    "memes placing historical figures in modern-day situations (such as Napoleon using an iPad)",
    "memes about traditions, customs, or heritage-related humour (e.g., comparing holiday rituals across families)",
    "memes comparing historical norms to modern behaviour or lifestyles (like medieval peasants discovering Wi-Fi)",

    # Pop Culture 
    "memes referencing movies, shows, cinematic universes, or scenes (e.g., using a Thanos quote to describe chores)",
    "memes involving celebrities, musicians, stan culture, or drama (such as fans overreacting to a tiny announcement)",
    "memes tied to anime, manga, or large fandom communities (like referencing over-the-top power-up tropes)",
    "memes referencing games, gaming culture, or gaming characters (e.g., respawning after an embarrassing death)",

    # Cultural Memes 
    "memes about cultural conflicts, identity debates, or social issues (such as two characters arguing over slang)",
    "memes commenting on internet culture, platforms, and trends (e.g., ‘Twitter at 3am’ starting pointless fights)",
    "memes about cultural misunderstandings or translation failures (like misreading signs abroad in funny ways)",
    "memes about people who are extremely online or terminally online (such as obsessing over follower counts)",
]





TAG_CATEGORY = {
    # Self-Deprecating
    "self-deprecating": "self-deprecating",
    "personal failure": "self-deprecating",
    "social awkwardness": "self-deprecating",
    "exhausted": "self-deprecating",

    # Doomer / Reactionary
    "anti-societal": "doomer / reactionary",
    "doom-predictions": "doomer / reactionary",
    "anti-doomer": "doomer / reactionary",
    "genre-reactionary": "doomer / reactionary",

    # Hobby / Niche
    "hobby or niche meme": "hobby / niche",
    "fandom in-joke": "hobby / niche",
    "work or school niche meme": "hobby / niche",
    "internet niche meme": "hobby / niche",

    # Relatable
    "personal stories and situations": "relatable",
    "mood relatability": "relatable",
    "everyday inconveniences": "relatable",
    "relationships and social life relatability": "relatable",

    # Misc / Meta
    "referential meme": "misc / meta",
    "abstract or bizarre humour": "misc / meta",
    "chaotic low-effort meme": "misc / meta",
    "meme culture": "misc / meta",

    # Geographic
    "country or region humour": "geographic",
    "city-specific meme": "geographic",
    "global north vs global south": "geographic",
    "regional stereotypes": "geographic",

    # Stereotype
    "national or ethnic stereotypes": "stereotypes",
    "gender stereotypes": "stereotypes",
    "job or class stereotypes": "stereotypes",
    "age group stereotypes": "stereotypes",

    # Political
    "domestic politics meme": "political",
    "geopolitics and international relations": "political",
    "ideological or partisan conflict": "political",
    "policy debate humour": "political",

    # Animal
    "wholesome animal meme": "animals",
    "funny pet behaviour": "animals",
    "animals acting like humans": "animals",
    "animal reaction meme": "animals",

    # Cultural / Historical
    "historical reference meme": "cultural / historical",
    "historical figure in modern setting": "cultural / historical",
    "traditions and heritage humour": "cultural / historical",
    "modern vs historical comparison": "cultural / historical",

    # Pop Culture
    "tv and movie reference": "pop culture",
    "music and celebrity culture": "pop culture",
    "anime and fandom reference": "pop culture",
    "gaming pop culture joke": "pop culture",

    # Cultural Memes 
    "culture war discourse": "cultural memes",
    "internet culture commentary": "cultural memes",
    "cross-cultural misunderstanding": "cultural memes",
    "chronically online behaviour": "cultural memes",
}


#Load sentencetransformer for categories and meme tags
def _load_text_model():
    global SENT_EMB_MODEL, CAT_EMB, CANONICAL_EMB
    if SENT_EMB_MODEL is None:
        SENT_EMB_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

        #Encode embeddings for each category description
        cat_texts = [f"{k}: {v}" for k, v in CATEGORY_DESCRIPTORS.items()]
        CAT_EMB = SENT_EMB_MODEL.encode(cat_texts, normalize_embeddings=True)

        #Compute embeddings for each meme description
        CANONICAL_EMB = SENT_EMB_MODEL.encode(
            MEME_DESCRIPTIONS,
            normalize_embeddings=True
        )

#Download images and adjust if needed
def fetch_image(url: str):
    try:
        resp = requests.get(url, headers=HEADERS, timeout=20)
        resp.raise_for_status()
        img = Image.open(io.BytesIO(resp.content)).convert("RGB")
        if max(img.size) < 384:
            factor = 384 / max(img.size)
            img = img.resize((int(img.width * factor), int(img.height * factor)), Image.BICUBIC)
        return img
    except Exception as e:
        print(f"[warn] Failed to fetch image: {e}")
        return None

#Generate blip captions for an image
@torch.no_grad()
def blip_captions(img: Image.Image, n_return: int = 5) -> list[str]:
    load_vision_models()
    inputs = BLIP_PROC(images=img, text=None, return_tensors="pt").to(DEVICE)
    #BLIP settings 
    outputs = BLIP.generate(
        **inputs,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7,
        num_return_sequences=n_return,
        max_new_tokens=32,
        repetition_penalty=1.1,
    )
    caps = BLIP_PROC.batch_decode(outputs, skip_special_tokens=True)
    cleaned = []
    for c in caps:
        c = re.sub(r"\s+", " ", c).strip(" .,!?:;\"'()[]{}")
        if c:
            cleaned.append(c)
    #Remove dups 
    return remove_duplicates(cleaned)

#Extract unigrams and bigram candidates from BLIP
def uni_and_bigram_from_captions(captions: list[str]) -> list[str]:
    uni = []
    bi = []
    #Tokenize captions
    for c in captions:
        toks = [normalize_token(t) for t in tokenize(c)]
        toks = [t for t in toks if len(t) >= 3 and t not in STOPWORDS and not t.isdigit()]
        uni.extend(toks)
        #Build bigram 
        for i in range(len(toks) - 1):
            if toks[i] != toks[i + 1]:
                bi.append(f"{toks[i]} {toks[i+1]}")

    cand_scores = Counter()
    for t in uni:
        cand_scores[t] += 1.0 + 0.15 * len(t)
    for t in bi:
        cand_scores[t] += 1.25 + 0.1 * sum(len(w) for w in t.split())
    return [w for w, _ in cand_scores.most_common(50)]


# CLIP ranking text by similarity to image
@torch.no_grad()
def clip_rank(image: Image.Image, candidates: list[str], top_n: int = 10) -> list[str]:
    load_vision_models()
    if not candidates:
        return []
    inputs = CLIP_PROC(text=candidates, images=image, return_tensors="pt", padding=True).to(DEVICE)
    outputs = CLIP(**inputs)
    img_emb = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
    txt_emb = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)
    #Compute similarity between the image and each candidate text
    sim = (img_emb @ txt_emb.t()).squeeze(0)

    #Select indices for top_n most similar candidates
    idxs = torch.topk(sim, k=min(top_n, len(candidates))).indices.tolist()
    ranked = [candidates[i] for i in idxs]
    
    flat = []
    for kw in ranked:
        if " " in kw:
            parts = [p for p in kw.split() if p not in STOPWORDS and len(p) >= 3]
            flat.extend(parts)
        else:
            flat.append(kw)
    flat = [normalize_token(t) for t in flat]
    flat = [t for t in flat if t and t not in STOPWORDS]
    return remove_duplicates(flat)[:top_n]


#Map keywords to meme tags
def generalize_keywords(
    raw_keywords: list[str],
    texts_for_context: list[str],
    want_n: int = 6,
    min_sim: float = 0.20,
    return_with_scores: bool = False,
):

    _load_text_model()

    if not raw_keywords and not texts_for_context:
        return []
    #Combine captions and keywords into one document string
    doc_parts = []
    if texts_for_context:
        doc_parts.extend(texts_for_context)
    if raw_keywords:
        doc_parts.append(" ".join(raw_keywords))

    doc_text = " ".join(doc_parts)
    if not doc_text.strip():
        return []
    #Compare to meme descriptions
    doc_emb = SENT_EMB_MODEL.encode([doc_text], normalize_embeddings=True)
    sims = (doc_emb @ CANONICAL_EMB.T).ravel()
    idxs = np.argsort(-sims)

    picked: list[tuple[str, float]] = []
    used = set()

    #Checks min_sim threshold 
    for idx in idxs:
        sim = float(sims[idx])
        if sim < min_sim:
            break
        label = MEME_TAGS[idx].strip().lower()
        if label in used:
            continue
        used.add(label)
        picked.append((label, sim))
        if len(picked) >= want_n:
            break

    #Fallback if not enough keywords for 6 
    if len(picked) < want_n:
        for idx in idxs:
            label = MEME_TAGS[idx].strip().lower()
            if label in used:
                continue
            used.add(label)
            sim = float(sims[idx])
            picked.append((label, sim))
            if len(picked) >= want_n:
                break

    if not picked and len(MEME_TAGS) > 0:
        best_idx = int(np.argmax(sims))
        picked = [(MEME_TAGS[best_idx].strip().lower(), float(sims[best_idx]))]

    if return_with_scores:
        return [(label, round(sim * 100.0, 2)) for label, sim in picked]

    return [label for label, _ in picked]


#Keyword extraction
def extract_image_keywords_with_scores(
    image_url: str,
    want_n: int = 6,
    context_text: str = ""
):

    img = fetch_image(image_url)
    if img is None:
        return [], []

    try:
        caps = blip_captions(img, n_return=6)

        if context_text:
            caps_with_context = caps + [context_text]
        else:
            caps_with_context = caps

        cands = uni_and_bigram_from_captions(caps_with_context)

        #CLIP ranking of keywords from captions
        raw_keywords = clip_rank(img, cands, top_n=want_n * 3)

        tag_scores = generalize_keywords(
            raw_keywords,
            caps_with_context,
            want_n=want_n,
            min_sim=0.20,
            return_with_scores=True,
        )

        tags = [t for t, _ in tag_scores]
        return tags[:want_n], tag_scores[:want_n]

    except Exception as e:
        print(f"[warn] Keyword extraction failed: {e}")
        return [], []


def extract_image_keywords(
    image_url: str,
    want_n: int = 6,
    context_text: str = ""
) -> list[str]:
    tags, _ = extract_image_keywords_with_scores(image_url, want_n=want_n, context_text=context_text)
    return tags

#Subreddit classification
def normalize_subreddit_name(name: str) -> str:
    n = re.sub(r"[_\-]+", " ", name or "")
    n = re.sub(r"(?<=\D)(\d+)", r" \1 ", n)
    n = re.sub(r"\s+", " ", n).strip()
    return n


@lru_cache(maxsize=4096)
def classify_subreddit_theme(sub_name: str, title: str = "", desc: str = "") -> str:

    _load_text_model()
    pieces = [
        normalize_subreddit_name(sub_name or ""),
        title or "",
        desc or "",
    ]
    text = " | ".join([p for p in pieces if p]).strip()
    if not text:
        return "other"
    
    #Similarity to CATEGORY_DESCRIPTORS
    sub_emb = SENT_EMB_MODEL.encode([text], normalize_embeddings=True)
    sims = (sub_emb @ CAT_EMB.T).ravel()
    labels = list(CATEGORY_DESCRIPTORS.keys())
    best_idx = int(np.argmax(sims))
    best_label = labels[best_idx]
    best_sim = float(sims[best_idx])

    if best_sim >= 0.35 and best_label != "other":
        return best_label

    text_lower = f"{(sub_name or '').lower()} {(title or '').lower()} {(desc or '').lower()}"

    #Fallback 
    if any(k in text_lower for k in ["politic", "trump", "biden", "democrat", "republican",
                                     "election", "government", "congress", "policy"]):
        return "political"
    if any(k in text_lower for k in ["meme", "funny", "humor", "comedy", "joke", "laugh"]):
        return "humor"
    if any(k in text_lower for k in ["tech", "programming", "coding", "python", "ai",
                                     "machinelearning", "data", "developer", "software"]):
        return "technology"
    if any(k in text_lower for k in ["game", "gaming", "xbox", "playstation", "nintendo",
                                     "minecraft", "fortnite", "valorant", "steam"]):
        return "gaming"
    if any(k in text_lower for k in ["art", "artist", "drawing", "painting", "design",
                                     "illustration", "sketch"]):
        return "art"
    if any(k in text_lower for k in ["crypto", "bitcoin", "ethereum", "finance",
                                     "stocks", "invest", "trading", "wallstreet"]):
        return "finance"
    if any(k in text_lower for k in ["sport", "nba", "soccer", "nfl", "mlb", "hockey", "fifa", "ufc"]):
        return "sports"
    if any(k in text_lower for k in ["university", "college", "school", "study", "education",
                                     "research", "science", "physics", "chemistry", "biology"]):
        return "education"
    if any(k in text_lower for k in ["fitness", "health", "food", "recipe", "cooking",
                                     "travel", "fashion", "lifestyle", "diet"]):
        return "lifestyle"

    music_terms = [
        "music","song","songs","album","albums","single","singles","artist","band","bands","lyrics","discography",
        "concert","tour","setlist","playlist","producer","dj","guitar","drums","bass","vocal","vocals","singer",
        "hiphop","hip-hop","rap","rnb","r&b","rock","metal","punk","indie","pop","kpop","k-pop","edm","electronic",
        "house","techno","trance","dubstep","jazz","classical","opera","folk","country","reggae","afrobeats","latin"
    ]
    #Remove non-alphanumeric characters
    name_hint = re.sub(r"[^a-z0-9]+", "", (sub_name or "").lower())
    if any(t in text_lower for t in music_terms) or any(t in name_hint for t in [
        "music","theweeknd","taylorswift","drake","kanyewest","arcticmonkeys","radiohead","metallica",
        "beatles","postmalone","badbunny","bts","blackpink","brunomars"
    ]):
        return "music"

    return "niche"
