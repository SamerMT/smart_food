import os
import json
import random
import logging
import traceback
import unicodedata
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, Header
from pydantic import BaseModel
import google.generativeai as genai
from groq import Groq

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("smart-food")

app = FastAPI(title="Smart Food API")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
APP_SECRET = os.getenv("APP_SECRET")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
if GROQ_API_KEY:
    groq_client = Groq(api_key=GROQ_API_KEY)

TEXT_MODEL = "llama-3.3-70b-versatile"

# Çalışan model adını EN BAŞA koy. /debug/models ile listeyi görebilirsin.
VISION_CANDIDATES = [
    "models/gemini-3.6-flash",
    "models/gemini-3.5-flash",
    "models/gemini-2.5-flash",
    "models/gemini-flash-latest",
]
_working_vision_model: Optional[str] = None

_RECIPES_CACHE: Optional[list] = None


# ─────────────────────────── Modeller ───────────────────────────

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    user_message: str
    fridge_items: list[str] = []
    history: list[ChatMessage] = []
    context_notes: list[str] = []
    exclude_recipe_names: list[str] = []
    diet_prefs: list[str] = []


class ProductChatRequest(BaseModel):
    """Etiket taraması sonrası, yalnızca o ürüne özel sohbet."""
    user_message: str
    product: dict
    diet_prefs: list[str] = []
    allergens: list[str] = []
    history: list[ChatMessage] = []


# ─────────────────────────── Yardımcılar ───────────────────────────

def check_auth(x_app_key: Optional[str]):
    if APP_SECRET and x_app_key != APP_SECRET:
        raise HTTPException(401, "Unauthorized")


def normalize(text: str) -> str:
    text = str(text).lower()
    for a, b in [("ı", "i"), ("ş", "s"), ("ğ", "g"), ("ü", "u"), ("ö", "o"), ("ç", "c")]:
        text = text.replace(a, b)
    return unicodedata.normalize("NFKD", text).strip()


def groq_call(messages: list, temperature: float, json_mode: bool = False):
    """Groq çağrısı. Kota hatasını 429 olarak yüzeye çıkarır."""
    try:
        kwargs = dict(messages=messages, model=TEXT_MODEL, temperature=temperature)
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        return groq_client.chat.completions.create(**kwargs)
    except Exception as e:
        text = str(e)
        if "429" in text or "rate limit" in text.lower():
            log.warning(f"[groq] RATE LIMIT: {text[:300]}")
            raise HTTPException(
                status_code=429,
                detail="Günlük yapay zeka kotası doldu. Lütfen daha sonra tekrar deneyin.",
            )
        raise


# ───────────────────── Tarif veri seti yardımcıları ─────────────────────
# Veri setinin alan adları TÜRKÇE:
#   tarif_adi, kategori, malzemeler[{isim, miktar, birim}],
#   yapilis_adimlari[], pisirme_suresi_dk, zorluk

def load_recipes() -> list:
    global _RECIPES_CACHE
    if _RECIPES_CACHE is None:
        try:
            with open("recipes_groq_cleaned.json", "r", encoding="utf-8") as f:
                _RECIPES_CACHE = json.load(f)
            log.info(f"[recipes] loaded {len(_RECIPES_CACHE)}")
        except Exception as e:
            log.warning(f"[recipes] load failed: {e}")
            _RECIPES_CACHE = []
    return _RECIPES_CACHE


def get_recipe_name(r: dict) -> str:
    return r.get("tarif_adi") or r.get("name") or r.get("title") or ""


def get_ingredient_names(r: dict) -> list[str]:
    """malzemeler: [{isim, miktar, birim}] → ['kıyma', 'soğan', ...]"""
    raw = r.get("malzemeler") or r.get("ingredients") or []
    out = []
    for i in raw:
        n = i.get("isim") or i.get("name") or "" if isinstance(i, dict) else str(i)
        n = n.strip()
        if n:
            out.append(n)
    return out


# Dolapta zaten bulunan / kullanıcının envantere eklemediği temel malzemeler.
# Eşleştirme oranını bunlar bozmasın diye hesaptan çıkarılır.
PANTRY_STAPLES = {
    "tuz", "karabiber", "kimyon", "kirmizi toz biber", "kirmizi pul biber",
    "pul biber", "toz biber", "nane", "kekik", "seker", "toz seker", "un",
    "su", "sivi yag", "zeytinyagi", "yemeklik yag", "aycicek yagi",
    "kabartma tozu", "vanilya", "vanilin", "sirke", "limon suyu",
    "salca", "domates salcasi", "biber salcasi", "nisasta", "maydanoz",
    "dereotu", "sumak", "tarcin", "karanfil", "defne yapragi", "susam",
    "galeta unu", "irmik", "bulyon", "seker surubu", "bal",
}


def core_ingredients(names: list[str]) -> list[str]:
    """Baharat ve temel malzemeleri eşleştirme dışında bırakır."""
    core = [n for n in names if normalize(n) not in PANTRY_STAPLES]
    return core or names


# Diyet filtreleri — malzeme adları üzerinde tam/kelime eşleşmesi
MEAT_WORDS = {"et", "kiyma", "tavuk", "hindi", "kuzu", "dana", "balik",
              "sucuk", "pastirma", "salam", "sosis", "ciger", "kusbasi",
              "but", "gogus", "kanat", "midye", "karides", "ton baligi"}
DAIRY_WORDS = {"sut", "peynir", "yogurt", "kaymak", "krema", "tereyagi",
               "labne", "ayran", "kasar", "lor", "cokelek"}
ANIMAL_WORDS = MEAT_WORDS | DAIRY_WORDS | {"yumurta", "jelatin", "bal"}
HARAM_WORDS = {"domuz", "jambon", "bacon", "alkol", "sarap", "likor",
               "rom", "bira", "votka", "konyak"}


def has_word(ings_norm: list[str], words: set) -> bool:
    for ing in ings_norm:
        tokens = set(ing.split())
        if ing in words or tokens & words:
            return True
    return False


def passes_diet(ings: list[str], prefs: set) -> bool:
    n = [normalize(i) for i in ings]
    if "helal" in prefs and has_word(n, HARAM_WORDS):
        return False
    if "vejetaryen" in prefs and has_word(n, MEAT_WORDS):
        return False
    if "vegan" in prefs and has_word(n, ANIMAL_WORDS):
        return False
    if "laktozsuz" in prefs and has_word(n, DAIRY_WORDS):
        return False
    return True


def slim_recipe(r: dict) -> dict:
    """TOKEN TASARRUFU: modele sadece ad + malzeme adları + süre gider."""
    return {
        "name": get_recipe_name(r),
        "ingredients": get_ingredient_names(r)[:12],
        "minutes": r.get("pisirme_suresi_dk"),
        "difficulty": r.get("zorluk"),
    }


def find_matching_recipes(
    fridge_items: list[str],
    exclude_names: list[str],
    diet_prefs: list[str],
) -> str:
    """Dolaba göre gerçek eşleştirme + çeşitlilik. fridge_items risk sıralı gelir."""
    all_recipes = load_recipes()
    if not all_recipes:
        return "[]"

    excluded = {normalize(x) for x in exclude_names}
    prefs = {normalize(p) for p in diet_prefs}
    fridge_norm = [normalize(i) for i in fridge_items]

    scored = []
    for r in all_recipes:
        if normalize(get_recipe_name(r)) in excluded:
            continue

        all_ings = get_ingredient_names(r)
        if not all_ings:
            continue

        if not passes_diet(all_ings, prefs):
            continue

        ings = core_ingredients(all_ings)

        matches, urgent = 0, 0
        for ing in ings:
            ing_n = normalize(ing)
            words = [w for w in ing_n.split() if len(w) > 3]
            for idx, item in enumerate(fridge_norm):
                if ing_n in item or any(w in item for w in words):
                    matches += 1
                    if idx < 3 or "acil" in item:
                        urgent += 1
                    break

        ratio = matches / len(ings)
        if ratio >= 0.4:
            scored.append((urgent, round(ratio, 1), r))

    if not scored:
        pool = [r for r in all_recipes
                if normalize(get_recipe_name(r)) not in excluded
                and passes_diet(get_ingredient_names(r), prefs)]
        picks = random.sample(pool, min(5, len(pool))) if pool else []
        log.info(f"[recipes] no match, random {len(picks)}")
        return json.dumps([slim_recipe(r) for r in picks], ensure_ascii=False)

    # ÇEŞİTLİLİK: eşit puanlıları karıştır, sonra sırala
    random.shuffle(scored)
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)

    top = [slim_recipe(r) for _, _, r in scored[:6]]
    log.info(f"[recipes] matched {len(scored)}, sending {len(top)}")
    return json.dumps(top, ensure_ascii=False)


def run_vision(parts: list) -> str:
    global _working_vision_model
    candidates = ([_working_vision_model] if _working_vision_model else []) + [
        m for m in VISION_CANDIDATES if m != _working_vision_model
    ]
    last = None
    for name in candidates:
        try:
            log.info(f"[vision] trying {name}")
            res = genai.GenerativeModel(name).generate_content(parts)
            _working_vision_model = name
            log.info(f"[vision] SUCCESS {name}")
            return res.text
        except Exception as e:
            last = e
            log.warning(f"[vision] FAILED {name}: {type(e).__name__}: {e}")
    raise RuntimeError(f"No working Gemini model. Last: {last}")


# ─────────────────────────── Endpointler ───────────────────────────

@app.get("/")
def read_root():
    return {
        "status": "success",
        "message": "Smart Food Backend is running!",
        "vision_model": _working_vision_model or "not tested yet",
        "recipes_loaded": len(load_recipes()),
    }


@app.get("/debug/models")
def debug_models():
    if not GEMINI_API_KEY:
        raise HTTPException(500, "GEMINI_API_KEY not set")
    try:
        names = [m.name for m in genai.list_models()
                 if "generateContent" in getattr(m, "supported_generation_methods", [])]
        return {"status": "success", "available": names}
    except Exception as e:
        log.error(traceback.format_exc())
        raise HTTPException(500, f"{type(e).__name__}: {e}")


@app.get("/debug/recipe")
def debug_recipe():
    r = load_recipes()
    if not r:
        return {"count": 0, "sample": None}
    s = r[0]
    return {
        "count": len(r),
        "keys": list(s.keys()),
        "parsed_name": get_recipe_name(s),
        "parsed_ingredients": get_ingredient_names(s),
        "core_ingredients": core_ingredients(get_ingredient_names(s)),
        "slim": slim_recipe(s),
    }


@app.get("/debug/match")
def debug_match(items: str = "domates,sogan,yumurta"):
    """Eşleştirmeyi tarayıcıdan test et: /debug/match?items=domates,kiyma"""
    fridge = [i.strip() for i in items.split(",") if i.strip()]
    result = find_matching_recipes(fridge, [], [])
    return {"fridge": fridge, "matches": json.loads(result)}


@app.post("/analyze-fridge")
async def analyze_fridge(
    images: list[UploadFile] = File(...),
    x_app_key: Optional[str] = Header(None),
):
    check_auth(x_app_key)
    log.info(f"[analyze-fridge] {len(images)} image(s)")

    if not GEMINI_API_KEY or not GROQ_API_KEY:
        raise HTTPException(500, "API keys are not configured.")

    try:
        parts = [
            "List every food item, drink and package visible across these images. "
            "Read visible labels. List each item once. Include approximate quantity."
        ]
        for img in images[:4]:
            data = await img.read()
            log.info(f"[analyze-fridge] {img.filename} {len(data)} bytes")
            parts.append({"mime_type": img.content_type or "image/jpeg", "data": data})

        raw = run_vision(parts)

        prompt = f"""Clean this fridge item list.
1. Remove duplicates.
2. Turkish names WITH proper characters (Süt not Sut).
3. category: exactly one of Sebze, Meyve, Süt Ürünleri, Protein, Tahıl, İçecek, Baharat, Donuk, Diğer
4. freshness_points 0-100 = how fresh the item LOOKS only. Cannot judge → 85.
5. quantity (number) + unit (adet, paket, kap, gram, ml, kg, litre).

Return ONLY JSON, no markdown:
{{"items":[{{"name":"Domates","category":"Sebze","freshness_points":80,"quantity":5,"unit":"adet"}}]}}

Text: {raw[:3000]}"""

        completion = groq_call([{"role": "user", "content": prompt}], 0.1, json_mode=True)
        data = json.loads(completion.choices[0].message.content)
        log.info(f"[analyze-fridge] {len(data.get('items', []))} items")
        return {"status": "success", "data": data}

    except HTTPException:
        raise
    except Exception as e:
        log.error("[analyze-fridge] FAILED\n" + traceback.format_exc())
        raise HTTPException(500, f"{type(e).__name__}: {e}")


@app.post("/analyze-label")
async def analyze_label(
    images: list[UploadFile] = File(...),
    x_app_key: Optional[str] = Header(None),
):
    check_auth(x_app_key)
    log.info(f"[analyze-label] {len(images)} image(s)")

    if not GEMINI_API_KEY or not GROQ_API_KEY:
        raise HTTPException(500, "API keys are not configured.")

    try:
        parts = [
            "You are an OCR engine. Transcribe ALL text on this product packaging "
            "across the images, exactly as written: product name, brand, ingredients, "
            "allergen warnings, net weight, storage instructions, expiry date. "
            "Do not summarize or translate. Raw transcription only."
        ]
        for img in images[:5]:
            data = await img.read()
            log.info(f"[analyze-label] {img.filename} {len(data)} bytes")
            parts.append({"mime_type": img.content_type or "image/jpeg", "data": data})

        ocr = run_vision(parts)

        prompt = f"""Parse this food label OCR text.

1. product_name: Turkish, proper characters. Unreadable → null.
2. ingredients: ARRAY of strings (Turkish). None found → [].
3. allergens: ARRAY of strings — ONLY the allergens actually present.
   Allowed keys: sut, yumurta, gluten, findik, fistik, soya, susam,
   balik, kabuklu_deniz, hardal, kereviz
   With milk and hazelnut: "allergens": ["sut", "findik"]
   With none: "allergens": []
   NEVER return an object with true/false values. ALWAYS an array of strings.
4. contains_lactose, contains_pork, contains_alcohol: booleans.
   pork = domuz, jambon, bacon, lard, unknown-origin gelatin.
5. category: exactly one of Sebze, Meyve, Süt Ürünleri, Protein, Tahıl, İçecek, Baharat, Donuk, Diğer
6. expiry_date: ISO "YYYY-MM-DD" or null.
7. storage: Buzdolabı, Dondurucu, Oda Sıcaklığı or Kiler. Default Buzdolabı.
8. confidence: 0-100, how readable the label was.
9. summary: ONE short Turkish sentence describing the product.
10. raw_text: original OCR text, max 1500 chars.

Return ONLY JSON, no markdown.

OCR: {ocr[:4000]}"""

        completion = groq_call([{"role": "user", "content": prompt}], 0.1, json_mode=True)
        data = json.loads(completion.choices[0].message.content)

        # Güvenlik ağı: model yine de obje döndürürse diziye çevir
        al = data.get("allergens")
        if isinstance(al, dict):
            data["allergens"] = [k for k, v in al.items() if v is True]
            log.warning("[analyze-label] allergens was object, converted to array")
        elif not isinstance(al, list):
            data["allergens"] = []

        return {"status": "success", "data": data}

    except HTTPException:
        raise
    except Exception as e:
        log.error("[analyze-label] FAILED\n" + traceback.format_exc())
        raise HTTPException(500, f"{type(e).__name__}: {e}")


@app.post("/product-chat")
async def product_chat(
    request: ProductChatRequest,
    x_app_key: Optional[str] = Header(None),
):
    """Etiket taraması sonrası, YALNIZCA o ürün hakkında sohbet."""
    check_auth(x_app_key)

    if not GROQ_API_KEY:
        raise HTTPException(500, "Groq API key not configured.")

    try:
        p = request.product
        product_block = json.dumps({
            "name": p.get("product_name"),
            "ingredients": (p.get("ingredients") or [])[:25],
            "allergens": p.get("allergens") or [],
            "contains_lactose": p.get("contains_lactose"),
            "contains_pork": p.get("contains_pork"),
            "contains_alcohol": p.get("contains_alcohol"),
            "category": p.get("category"),
            "storage": p.get("storage"),
            "expiry_date": p.get("expiry_date"),
        }, ensure_ascii=False)

        system = f"""Your name is "Gıda Asistanı".

You are discussing ONE specific product the user just scanned. Nothing else.

PRODUCT:
{product_block}

USER PROFILE:
- Diet preferences: {', '.join(request.diet_prefs) or 'yok'}
- Allergens: {', '.join(request.allergens) or 'yok'}

Rules:
1. Answer ONLY about this product: ingredients, whether it matches the user's
   diet and allergens, how to store it, what it contains.
2. Do NOT suggest recipes. Do NOT talk about the fridge. Do NOT offer to add it anywhere.
3. If asked whether it suits them, compare against the profile and answer clearly:
   uygun / uygun değil / emin değilim — and say exactly WHY, naming the ingredient.
4. State facts only. Never medical or nutrition advice.
   Say "Bu üründe süt var", never "sağlığınız için kaçının".
5. If the label was unreadable (null name, empty ingredients), say so honestly and
   suggest rescanning with better lighting.
6. Out of scope → "Ben taradığınız ürün hakkında yardımcı olabilirim."
7. Short answers: 2-4 sentences. This is a phone screen.

Always answer in Turkish."""

        messages = [{"role": "system", "content": system}]
        for m in request.history[-6:]:
            if m.role in ("user", "assistant"):
                messages.append({"role": m.role, "content": m.content[:600]})
        messages.append({"role": "user", "content": request.user_message})

        completion = groq_call(messages, 0.3)
        return {"status": "success", "reply": completion.choices[0].message.content}

    except HTTPException:
        raise
    except Exception as e:
        log.error("[product-chat] FAILED\n" + traceback.format_exc())
        raise HTTPException(500, f"{type(e).__name__}: {e}")


@app.post("/chat")
async def chat_bot(
    request: ChatRequest,
    x_app_key: Optional[str] = Header(None),
):
    """Dolap sohbeti — tarif önerisi ve ikame."""
    check_auth(x_app_key)

    if not GROQ_API_KEY:
        raise HTTPException(500, "Groq API key not configured.")

    try:
        items = "\n".join(f"- {i}" for i in request.fridge_items[:25]) or "(dolap boş)"
        recipes = find_matching_recipes(
            request.fridge_items,
            request.exclude_recipe_names,
            request.diet_prefs,
        )

        ctx = ""
        if request.context_notes:
            ctx = "\nSON TARANAN ETİKET:\n" + "\n".join(request.context_notes[-1:])[:800]

        system = f"""Your name is "Gıda Asistanı". If asked who you are:
"Ben Gıda Asistanınızım."

FRIDGE (most urgent first):
{items}

DIET: {', '.join(request.diet_prefs) or 'yok'}
{ctx}
AVAILABLE RECIPES (name, ingredients, minutes, difficulty):
{recipes}

Rules:
1. Suggest ONLY from AVAILABLE RECIPES. Never invent a recipe.
2. Never mention internal labels like "AVAILABLE RECIPES".
3. Prioritise items marked ACİL and say WHY you picked that dish.
4. Suggest ONE dish at a time unless asked for more. Do not list everything.
5. If the user asks for something different, vegetarian, oil-free, quicker etc.,
   pick a DIFFERENT dish from the list that fits. Never repeat the previous one.
6. Suggest substitutions for missing ingredients.
7. Salt, pepper, oil, flour and common spices are assumed to be at home —
   do not treat them as missing.
8. Scope: fridge, recipes, cooking, substitutions. Otherwise reply:
   "Ben Gıda Asistanınızım. Dolabınızdaki malzemeler ve tarifler konusunda
   yardımcı olabilirim."
9. Never medical or nutrition advice. Facts only.
10. If diet includes HELAL, never suggest pork or alcohol.
11. Short: 3-5 sentences.

Always answer in Turkish."""

        messages = [{"role": "system", "content": system}]
        for m in request.history[-6:]:
            if m.role in ("user", "assistant"):
                messages.append({"role": m.role, "content": m.content[:600]})
        messages.append({"role": "user", "content": request.user_message})

        completion = groq_call(messages, 0.6)
        return {"status": "success", "reply": completion.choices[0].message.content}

    except HTTPException:
        raise
    except Exception as e:
        log.error("[chat] FAILED\n" + traceback.format_exc())
        raise HTTPException(500, f"{type(e).__name__}: {e}")
