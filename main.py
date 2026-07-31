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

# Kendi çalışan model adını EN BAŞA koy
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
    exclude_recipe_names: list[str] = []   # YENİ: çeşitlilik için
    diet_prefs: list[str] = []             # YENİ: laktozsuz, helal, vejetaryen...


class ProductChatRequest(BaseModel):
    """YENİ: etiket taraması sonrası ürüne özel sohbet."""
    user_message: str
    product: dict                  # /analyze-label çıktısı
    diet_prefs: list[str] = []
    allergens: list[str] = []
    history: list[ChatMessage] = []


# ─────────────────────────── Yardımcılar ───────────────────────────

def check_auth(x_app_key: Optional[str]):
    if APP_SECRET and x_app_key != APP_SECRET:
        raise HTTPException(401, "Unauthorized")


def normalize(text: str) -> str:
    text = text.lower()
    for a, b in [("ı", "i"), ("ş", "s"), ("ğ", "g"), ("ü", "u"), ("ö", "o"), ("ç", "c")]:
        text = text.replace(a, b)
    return unicodedata.normalize("NFKD", text)


def groq_call(messages: list, temperature: float, json_mode: bool = False):
    """Groq çağrısı + rate limit'i 429 olarak yüzeye çıkarma."""
    try:
        kwargs = dict(messages=messages, model=TEXT_MODEL, temperature=temperature)
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        return groq_client.chat.completions.create(**kwargs)
    except Exception as e:
        text = str(e)
        if "429" in text or "rate limit" in text.lower() or "Rate limit" in text:
            log.warning(f"[groq] RATE LIMIT: {text[:300]}")
            raise HTTPException(
                status_code=429,
                detail="Günlük yapay zeka kotası doldu. Lütfen daha sonra tekrar deneyin.",
            )
        raise


def load_recipes() -> list:
    global _RECIPES_CACHE
    if _RECIPES_CACHE is None:
        try:
            with open("recipes_groq_cleaned.json", "r", encoding="utf-8") as f:
                _RECIPES_CACHE = json.load(f)
            log.info(f"[recipes] loaded {len(_RECIPES_CACHE)} recipes")
        except Exception as e:
            log.warning(f"[recipes] load failed: {e}")
            _RECIPES_CACHE = []
    return _RECIPES_CACHE


def slim_recipe(r: dict) -> dict:
    """
    TOKEN TASARRUFU: modele sadece ad + malzeme gider.
    Adımlar, açıklamalar, görseller GÖNDERİLMEZ.
    """
    ings = r.get("ingredients") or []
    return {
        "name": r.get("name") or r.get("title") or "",
        "ingredients": [str(i)[:40] for i in ings[:12]],
    }


def find_matching_recipes(
    fridge_items: list[str],
    exclude_names: list[str],
    diet_prefs: list[str],
) -> str:
    """
    Dolaba göre gerçek eşleştirme + çeşitlilik.
    fridge_items risk sırasına göre gelir (en riskli ilk).
    """
    all_recipes = load_recipes()
    if not all_recipes:
        return "[]"

    excluded = {normalize(n) for n in exclude_names}
    prefs = {normalize(p) for p in diet_prefs}
    fridge_norm = [normalize(i) for i in fridge_items]

    scored = []
    for r in all_recipes:
        name_n = normalize(r.get("name") or r.get("title") or "")
        if name_n in excluded:
            continue

        ings = r.get("ingredients") or []
        if not ings:
            continue

        blob = normalize(json.dumps(r, ensure_ascii=False))

        # Diyet filtreleri
        if "helal" in prefs and any(w in blob for w in
                                    ["domuz", "jambon", "bacon", "alkol", "sarap", "likor", "rom"]):
            continue
        if "vejetaryen" in prefs and any(w in blob for w in
                                         ["et", "tavuk", "balik", "kiyma", "sucuk", "pastirma"]):
            continue
        if "vegan" in prefs and any(w in blob for w in
                                    ["et", "tavuk", "balik", "kiyma", "sut", "peynir", "yumurta", "yogurt", "tereyagi"]):
            continue
        if "laktozsuz" in prefs and any(w in blob for w in
                                        ["sut", "peynir", "yogurt", "kaymak", "krema", "tereyagi"]):
            continue

        matches, urgent = 0, 0
        for ing in ings:
            ing_n = normalize(str(ing))
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
                if normalize(r.get("name") or r.get("title") or "") not in excluded]
        picks = random.sample(pool, min(5, len(pool))) if pool else []
        return json.dumps([slim_recipe(r) for r in picks], ensure_ascii=False)

    # ÇEŞİTLİLİK: aynı puandakileri karıştır, sonra sırala
    random.shuffle(scored)
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)

    top = [slim_recipe(r) for _, _, r in scored[:6]]
    log.info(f"[recipes] matched {len(scored)}, sending {len(top)} slim")
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
3. Category: exactly one of Sebze, Meyve, Süt Ürünleri, Protein, Tahıl, İçecek, Baharat, Donuk, Diğer
4. freshness_points 0-100 = how fresh it LOOKS only. Cannot judge → 85.
5. quantity + unit (adet, paket, kap, gram, ml, kg, litre).

Return ONLY JSON:
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
2. ingredients: array (Turkish).
3. allergens: ONLY these keys — sut, yumurta, gluten, findik, fistik, soya, susam,
   balik, kabuklu_deniz, hardal, kereviz
4. contains_lactose, contains_pork, contains_alcohol: booleans.
   pork = domuz, jambon, bacon, lard, unknown-origin gelatin.
5. category: Sebze, Meyve, Süt Ürünleri, Protein, Tahıl, İçecek, Baharat, Donuk, Diğer
6. expiry_date: ISO "YYYY-MM-DD" or null.
7. storage: Buzdolabı, Dondurucu, Oda Sıcaklığı, Kiler. Default Buzdolabı.
8. confidence 0-100.
9. summary: 1 short Turkish sentence describing the product.
10. raw_text: original OCR, max 1500 chars.

Return ONLY JSON.

OCR: {ocr[:4000]}"""

        completion = groq_call([{"role": "user", "content": prompt}], 0.1, json_mode=True)
        return {"status": "success", "data": json.loads(completion.choices[0].message.content)}

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
    """
    YENİ — Etiket taraması sonrası, YALNIZCA o ürün hakkında sohbet.
    Dolap sohbetinden tamamen ayrıdır; tarif önermez, dolaba ekleme önermez.
    """
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
1. Answer ONLY about this product: its ingredients, whether it matches the user's
   diet and allergens, how to store it, what it contains.
2. Do NOT suggest recipes. Do NOT talk about the fridge. Do NOT offer to add it anywhere.
3. If asked whether it suits them, compare against the profile and answer clearly:
   uygun / uygun değil / emin değilim — and say exactly WHY, naming the ingredient.
4. State facts only. Never medical or nutrition advice.
   Say "Bu üründe süt var", never "sağlığınız için kaçının".
5. If the label was unreadable (null name, empty ingredients), say so honestly and
   suggest rescanning with better lighting.
6. Out of scope → "Ben taradığınız ürün hakkında yardımcı olabilirim."
7. Short answers: 2-4 sentences. Phone screen.

Always answer in Turkish."""

        messages = [{"role": "system", "content": system}]
        for m in request.history[-6:]:
            if m.role in ("user", "assistant"):
                messages.append({"role": m.role, "content": m.content})
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
AVAILABLE RECIPES (name + ingredients only):
{recipes}

Rules:
1. Suggest ONLY from AVAILABLE RECIPES. Never invent a recipe.
2. Never mention internal labels like "AVAILABLE RECIPES" or "MATCHED".
3. Prioritise items marked ACİL and say WHY you picked that dish.
4. Suggest ONE dish at a time unless asked for more. Do not list everything.
5. If the user asks for something different, vegetarian, oil-free, quicker, etc.
   — pick a DIFFERENT dish from the list that fits. Never repeat the previous one.
6. Suggest substitutions for missing ingredients.
7. Scope: fridge, recipes, cooking, substitutions. Otherwise:
   "Ben Gıda Asistanınızım. Dolabınızdaki malzemeler ve tarifler konusunda
   yardımcı olabilirim."
8. Never medical or nutrition advice. Facts only.
9. If diet includes HELAL, never suggest pork or alcohol.
10. Short: 3-5 sentences.

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

# test
@app.get("/debug/recipe")
def debug_recipe():
    r = load_recipes()
    return {"count": len(r), "sample": r[0] if r else None,
            "keys": list(r[0].keys()) if r else []}
