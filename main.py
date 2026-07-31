"""
Smart Food API

İKİ TAMAMEN AYRI AKIŞ — hiçbir noktada karışmazlar:

  A) ÜRÜN ETİKETİ AKIŞI (içindekiler okuma)
     /analyze-label   → Qwen 3.6 27B, saf OCR (layout korunur)
     /product-chat    → Qwen 3.6 27B, kendi OCR çıktısını kullanarak
                        SADECE o ürün hakkında sohbet (kalori, içindekiler,
                        "bana uygun mu")

  B) BUZDOLABI AKIŞI (envanter)
     /analyze-fridge  → Qwen 3.6 27B, görseldeki ürünleri sayar
     /chat            → niyet çıkarımı → yerel filtre → tarif önerisi

Tümü Groq üzerinden. Gemini kullanılmıyor.
NOT: llama-3.3-70b-versatile Groq'ta kullanımdan kaldırılıyor; yedek listede.
"""

import os
import re
import json
import base64
import random
import logging
import traceback
import unicodedata
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, Header
from pydantic import BaseModel
from groq import Groq

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("smart-food")

app = FastAPI(title="Smart Food API")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
APP_SECRET = os.getenv("APP_SECRET")

groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

# ── Modeller ────────────────────────────────────────────────────────────
VISION_MODELS = [
    "qwen/qwen3.6-27b",
    "qwen/qwen3.5-27b",
]
PRODUCT_CHAT_MODELS = [       # OCR'ı yapan modelin aynısı
    "qwen/qwen3.6-27b",
    "openai/gpt-oss-120b",
]
RECIPE_CHAT_MODELS = [
    "openai/gpt-oss-120b",
    "qwen/qwen3.6-27b",
    "llama-3.3-70b-versatile",
]
INTENT_MODELS = [             # küçük ve ucuz
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
]

_working: dict = {}
_RECIPES_CACHE = None

MAX_LABEL_IMAGES = 5
MAX_FRIDGE_IMAGES = 4


# ── Şemalar ─────────────────────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    user_message: str
    fridge_items: list[str] = []
    history: list[ChatMessage] = []
    exclude_recipe_names: list[str] = []
    diet_prefs: list[str] = []


class ProductChatRequest(BaseModel):
    user_message: str
    product: dict = {}
    raw_text: str = ""          # OCR ham metni — kalori/besin tablosu burada
    diet_prefs: list[str] = []
    allergens: list[str] = []
    history: list[ChatMessage] = []


# ── Ortak yardımcılar ───────────────────────────────────────────────────

def check_auth(x_app_key: Optional[str]):
    if APP_SECRET and x_app_key != APP_SECRET:
        raise HTTPException(401, "Unauthorized")


def normalize(text) -> str:
    t = str(text).lower()
    for a, b in [("ı", "i"), ("ş", "s"), ("ğ", "g"), ("ü", "u"), ("ö", "o"), ("ç", "c")]:
        t = t.replace(a, b)
    return unicodedata.normalize("NFKD", t).strip()


def strip_think(text: str) -> str:
    """Qwen düşünce çıktısını temizle — hem etiket hem düz metin biçimi."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = text.replace("<think>", "").replace("</think>", "")

    for marker in ("thinking process:", "thought process:", "thinking:",
                   "thought:", "reasoning:", "analysis:"):
        low = text.lower()
        i = low.find(marker)
        if i != -1:
            rest = text[i + len(marker):]
            parts = rest.split("\n\n", 1)
            text = text[:i] + (parts[1] if len(parts) > 1 else "")

    return text.strip()


def parse_json_loose(text: str) -> dict:
    """Markdown çiti veya ön/son metinle gelen JSON'u kurtarır."""
    t = strip_think(text).strip()
    t = re.sub(r"^```(?:json)?", "", t).strip()
    t = re.sub(r"```$", "", t).strip()
    try:
        return json.loads(t)
    except Exception:
        m = re.search(r"\{.*\}", t, re.DOTALL)
        if m:
            return json.loads(m.group(0))
        raise


def groq_chat(role: str, models: list, messages: list, temperature: float,
              json_mode: bool = False, max_tokens: Optional[int] = None) -> str:
    """Model listesini sırayla dener, çalışanı hatırlar. 429'u yüzeye çıkarır."""
    if not groq_client:
        raise HTTPException(500, "GROQ_API_KEY is not configured.")

    order = ([_working[role]] if role in _working else []) + \
            [m for m in models if m != _working.get(role)]

    last = None
    for name in order:
        try:
            kwargs = dict(model=name, messages=messages, temperature=temperature)
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            if max_tokens:
                kwargs["max_tokens"] = max_tokens
            if name.startswith("qwen"):                    # ← جديد
                kwargs["reasoning_format"] = "hidden"      # ← جديد


            res = groq_client.chat.completions.create(**kwargs)
            if _working.get(role) != name:
                log.info(f"[{role}] using model: {name}")
                _working[role] = name
            return res.choices[0].message.content or ""

        except Exception as e:
            msg = str(e)
            if "429" in msg or "rate limit" in msg.lower():
                log.warning(f"[{role}] RATE LIMIT on {name}")
                raise HTTPException(
                    429,
                    "Günlük yapay zeka kotası doldu. Lütfen daha sonra tekrar deneyin.",
                )
            last = e
            log.warning(f"[{role}] FAILED {name}: {type(e).__name__}: {msg[:200]}")

    raise RuntimeError(f"No working model for '{role}'. Last: {last}")


async def to_data_uri(img: UploadFile) -> str:
    data = await img.read()
    log.info(f"  image {img.filename} {img.content_type} {len(data)} bytes")
    if len(data) == 0:
        raise ValueError(f"Empty image: {img.filename}")
    mime = img.content_type or "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(data).decode()}"


def vision_message(prompt: str, data_uris: list) -> list:
    content = [{"type": "text", "text": prompt}]
    for uri in data_uris:
        content.append({"type": "image_url", "image_url": {"url": uri}})
    return [{"role": "user", "content": content}]


# ════════════════════════════════════════════════════════════════════════
#  A) ÜRÜN ETİKETİ AKIŞI
# ════════════════════════════════════════════════════════════════════════

# Colab'da doğrulanmış OCR promptu — DEĞİŞTİRME
OCR_PROMPT = """
ACT AS A HIGH-PRECISION OCR SCANNER.
Extract ALL visible text from the provided packaging images.

CRITICAL RULES:
1. Extract EVERYTHING: Ingredients, Allergen warnings, Nutritional tables (Energy, Fat, Carbs, etc.), Company details, Dates, and Weights.
2. PRESERVE THE LAYOUT: Maintain original line breaks, spacing, and structural formatting exactly as printed on the box. Do NOT merge everything into a single line.
3. If there is a table (like nutritional facts), format it clearly with line breaks.
4. NO <think> tags, NO commentary, NO intro/outro text. Output ONLY the raw extracted text.
"""

LABEL_STRUCTURE_PROMPT = """You are a food label parser. Below is OCR text from a product package.

Return ONLY a JSON object, no markdown, no commentary:

{
  "product_name": "Turkish name with proper characters, or null if unreadable",
  "brand": "brand or null",
  "ingredients": ["array of Turkish ingredient strings, [] if none found"],
  "allergens": ["ARRAY of keys actually present"],
  "contains_lactose": false,
  "contains_pork": false,
  "contains_alcohol": false,
  "category": "one of: Sebze, Meyve, Süt Ürünleri, Protein, Tahıl, İçecek, Baharat, Donuk, Diğer",
  "net_weight": "e.g. 500 g, or null",
  "expiry_date": "YYYY-MM-DD or null",
  "storage": "one of: Buzdolabı, Dondurucu, Oda Sıcaklığı, Kiler",
  "nutrition": {
    "serving": "per 100g / per portion, or null",
    "energy_kcal": null, "fat_g": null, "saturated_fat_g": null,
    "carbs_g": null, "sugar_g": null, "protein_g": null,
    "salt_g": null, "fiber_g": null
  },
  "confidence": 0,
  "summary": "ONE short Turkish sentence describing the product"
}

Rules:
- allergens: allowed keys ONLY — sut, yumurta, gluten, findik, fistik, soya,
  susam, balik, kabuklu_deniz, hardal, kereviz
  With milk and hazelnut: ["sut","findik"]. With none: [].
  NEVER an object with true/false. ALWAYS an array of strings.
- contains_pork: domuz, jambon, bacon, lard, unknown-origin gelatin.
- nutrition: read numbers from the nutritional table. Missing → null.
  Numbers only, no units.
- confidence: 0-100, how readable the label was.

OCR TEXT:
"""


@app.post("/analyze-label")
async def analyze_label(
    images: list[UploadFile] = File(...),
    x_app_key: Optional[str] = Header(None),
):
    """
    ÜRÜN ETİKETİ — içindekiler okuma.
    1) Qwen saf OCR (layout korunur)
    2) Aynı metni yapılandırılmış JSON'a çevir
    raw_text daima döner — /product-chat kalori sorularında bunu kullanır.
    """
    check_auth(x_app_key)
    log.info(f"[LABEL] {len(images)} image(s)")

    try:
        uris = [await to_data_uri(i) for i in images[:MAX_LABEL_IMAGES]]

        raw_text = strip_think(
            groq_chat("vision", VISION_MODELS,
                      vision_message(OCR_PROMPT, uris),
                      temperature=0.0, max_tokens=2500)
        )
        log.info(f"[LABEL] OCR {len(raw_text)} chars")

        if not raw_text.strip():
            return {"status": "success", "data": {
                "product_name": None, "ingredients": [], "allergens": [],
                "contains_lactose": False, "contains_pork": False,
                "contains_alcohol": False, "category": "Diğer",
                "storage": "Buzdolabı", "expiry_date": None, "nutrition": {},
                "confidence": 0, "summary": "Etiket okunamadı.", "raw_text": "",
            }}

        structured = parse_json_loose(
            groq_chat("label_struct", PRODUCT_CHAT_MODELS,
                      [{"role": "user",
                        "content": LABEL_STRUCTURE_PROMPT + raw_text[:6000]}],
                      temperature=0.1, json_mode=True)
        )

        al = structured.get("allergens")
        if isinstance(al, dict):
            structured["allergens"] = [k for k, v in al.items() if v is True]
            log.warning("[LABEL] allergens was object → converted to array")
        elif not isinstance(al, list):
            structured["allergens"] = []

        structured["raw_text"] = raw_text[:6000]
        log.info(f"[LABEL] name={structured.get('product_name')} "
                 f"conf={structured.get('confidence')}")
        return {"status": "success", "data": structured}

    except HTTPException:
        raise
    except Exception as e:
        log.error("[LABEL] FAILED\n" + traceback.format_exc())
        raise HTTPException(500, f"{type(e).__name__}: {e}")


@app.post("/product-chat")
async def product_chat(
    request: ProductChatRequest,
    x_app_key: Optional[str] = Header(None),
):
    """
    ÜRÜN SOHBETİ — aynı Qwen modeli, kendi OCR çıktısını yorumlar.
    SADECE taranan ürün. Tarif önermez, dolaptan bahsetmez.
    """
    check_auth(x_app_key)

    try:
        p = request.product or {}
        summary = json.dumps({
            "name": p.get("product_name"),
            "brand": p.get("brand"),
            "ingredients": (p.get("ingredients") or [])[:30],
            "allergens": p.get("allergens") or [],
            "contains_lactose": p.get("contains_lactose"),
            "contains_pork": p.get("contains_pork"),
            "contains_alcohol": p.get("contains_alcohol"),
            "category": p.get("category"),
            "net_weight": p.get("net_weight"),
            "storage": p.get("storage"),
            "expiry_date": p.get("expiry_date"),
            "nutrition": p.get("nutrition") or {},
        }, ensure_ascii=False)

        raw = (request.raw_text or p.get("raw_text") or "")[:5000]

        system = f"""Your name is "Gıda Asistanı".

You read this product's packaging yourself. Discuss ONLY this product.

STRUCTURED DATA:
{summary}

FULL LABEL TEXT (as printed — use this for calories, nutrition table,
weights, dates and anything not in the structured data above):
---
{raw}
---

USER PROFILE:
- Diet preferences: {', '.join(request.diet_prefs) or 'yok'}
- Allergens: {', '.join(request.allergens) or 'yok'}

Rules:
1. Answer ONLY about this product: ingredients, nutrition, calories, allergens,
   storage, weight, dates, and whether it fits the user's profile.
2. For calories or nutrition, read the FULL LABEL TEXT above. If a value is not
   printed there, say it is not on the label — never invent a number.
3. If asked whether it suits them, compare against the profile and answer clearly:
   uygun / uygun değil / emin değilim — and name the exact ingredient that decides it.
4. NEVER suggest recipes. NEVER mention the fridge or inventory. NEVER offer to
   add this product anywhere.
5. State facts only. No medical or nutrition advice.
   Say "Bu üründe süt var", never "sağlığınız için kaçının".
6. If the label was unreadable, say so honestly and suggest rescanning closer
   with better lighting.
7. Out of scope → "Ben taradığınız ürün hakkında yardımcı olabilirim."
8. Short: 2-4 sentences. This is a phone screen.
9. No <think> tags, no internal reasoning in the output.

Always answer in Turkish."""

        messages = [{"role": "system", "content": system}]
        for m in request.history[-6:]:
            if m.role in ("user", "assistant"):
                messages.append({"role": m.role, "content": m.content[:600]})
        messages.append({"role": "user", "content": request.user_message})

        reply = strip_think(
            groq_chat("product_chat", PRODUCT_CHAT_MODELS, messages, temperature=0.3)
        )
        return {"status": "success", "reply": reply}

    except HTTPException:
        raise
    except Exception as e:
        log.error("[PRODUCT-CHAT] FAILED\n" + traceback.format_exc())
        raise HTTPException(500, f"{type(e).__name__}: {e}")


# ════════════════════════════════════════════════════════════════════════
#  B) BUZDOLABI AKIŞI
# ════════════════════════════════════════════════════════════════════════

FRIDGE_PROMPT = """You are looking INSIDE a refrigerator or at grocery items.

Identify every distinct food item, drink and package visible across the images.
Read visible labels to name products accurately. If the same item appears in more
than one image, count it once.

Return ONLY a JSON object, no markdown:

{"items":[{"name":"Domates","category":"Sebze","freshness_points":80,"quantity":5,"unit":"adet"}]}

Rules:
- name: Turkish, proper characters (Süt not Sut). Be specific when the label is
  readable, e.g. "Laktozsuz Süt" rather than just "Süt".
- category: exactly one of Sebze, Meyve, Süt Ürünleri, Protein, Tahıl, İçecek,
  Baharat, Donuk, Diğer
- freshness_points: 0-100, how fresh the item LOOKS (colour, wilting, packaging
  condition, visible mould). Cannot judge visually → 85. This is NOT shelf life.
- quantity: a number. unit: adet, paket, kap, gram, ml, kg or litre.
- Do not list shelves, containers or the fridge itself. Food and drink only.
- No <think> tags, no commentary.
"""


@app.post("/analyze-fridge")
async def analyze_fridge(
    images: list[UploadFile] = File(...),
    x_app_key: Optional[str] = Header(None),
):
    """BUZDOLABI — envantere eklenecek ürünleri sayar. Etiket akışıyla ilgisi yok."""
    check_auth(x_app_key)
    log.info(f"[FRIDGE] {len(images)} image(s)")

    try:
        uris = [await to_data_uri(i) for i in images[:MAX_FRIDGE_IMAGES]]

        data = parse_json_loose(
            groq_chat("vision", VISION_MODELS,
                      vision_message(FRIDGE_PROMPT, uris),
                      temperature=0.1, json_mode=True)
        )

        items = data.get("items") or []
        log.info(f"[FRIDGE] detected {len(items)} items")
        return {"status": "success", "data": {"items": items}}

    except HTTPException:
        raise
    except Exception as e:
        log.error("[FRIDGE] FAILED\n" + traceback.format_exc())
        raise HTTPException(500, f"{type(e).__name__}: {e}")


# ── Tarif veri seti (Türkçe alan adları) ────────────────────────────────

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


def r_name(r: dict) -> str:
    return r.get("tarif_adi") or r.get("name") or ""


def r_category(r: dict) -> str:
    return r.get("kategori") or "diğer"


def r_ingredients(r: dict) -> list:
    raw = r.get("malzemeler") or r.get("ingredients") or []
    out = []
    for i in raw:
        n = (i.get("isim") or i.get("name") or "") if isinstance(i, dict) else str(i)
        if n.strip():
            out.append(n.strip())
    return out


PANTRY_STAPLES = {
    "tuz", "karabiber", "kimyon", "kirmizi toz biber", "kirmizi pul biber",
    "pul biber", "toz biber", "nane", "kekik", "seker", "toz seker", "un",
    "su", "sivi yag", "zeytinyagi", "yemeklik yag", "aycicek yagi",
    "kabartma tozu", "vanilya", "vanilin", "sirke", "limon suyu", "salca",
    "domates salcasi", "biber salcasi", "nisasta", "sumak", "tarcin",
    "karanfil", "defne yapragi", "galeta unu", "irmik", "bulyon",
}


def core_ingredients(names: list) -> list:
    core = [n for n in names if normalize(n) not in PANTRY_STAPLES]
    return core or names


MEAT = {"et", "kiyma", "tavuk", "hindi", "kuzu", "dana", "balik", "sucuk",
        "pastirma", "salam", "sosis", "ciger", "kusbasi", "but", "kanat",
        "midye", "karides"}
DAIRY = {"sut", "peynir", "yogurt", "kaymak", "krema", "tereyagi", "labne",
         "ayran", "kasar", "lor", "cokelek"}
ANIMAL = MEAT | DAIRY | {"yumurta", "jelatin"}
HARAM = {"domuz", "jambon", "bacon", "alkol", "sarap", "likor", "rom",
         "bira", "votka", "konyak"}


def has_any(ings_norm: list, words: set) -> bool:
    for ing in ings_norm:
        if ing in words or set(ing.split()) & words:
            return True
    return False


def passes_diet(ings: list, prefs: set) -> bool:
    n = [normalize(i) for i in ings]
    if "helal" in prefs and has_any(n, HARAM):
        return False
    if "vejetaryen" in prefs and has_any(n, MEAT):
        return False
    if "vegan" in prefs and has_any(n, ANIMAL):
        return False
    if "laktozsuz" in prefs and has_any(n, DAIRY):
        return False
    return True


def slim(r: dict) -> dict:
    return {
        "name": r_name(r),
        "category": r_category(r),
        "ingredients": r_ingredients(r)[:12],
        "minutes": r.get("pisirme_suresi_dk"),
        "difficulty": r.get("zorluk"),
    }


# ── Niyet çıkarımı — modelin kendisi anlar ──────────────────────────────

INTENT_PROMPT = """Extract the user's cooking intent from their message.
Use the recent conversation for context (e.g. "another one" refers to the
previous suggestion).

Return ONLY JSON:
{
  "want_categories": [],
  "exclude_categories": [],
  "must_ingredients": [],
  "exclude_ingredients": [],
  "max_minutes": null,
  "wants_different": false
}

Recipe categories in the database are Turkish and lowercase, e.g.:
  "ana yemek", "çorba", "salata", "tatlı", "kahvaltı", "börek", "makarna",
  "pilav", "içecek", "meze", "kurabiye", "kek", "hamur işi"

Guidance:
- "tatlı istemiyorum", "şekerli olmasın" → exclude_categories: ["tatlı","kek","kurabiye"]
- "yemek istiyorum", "akşam yemeği" → want_categories: ["ana yemek"],
  exclude_categories: ["tatlı","kek","kurabiye","içecek"]
- "domatesli bir şey" → must_ingredients: ["domates"]
- "etsiz olsun" → exclude_ingredients: ["et","kıyma","tavuk"]
- "hızlı olsun" → max_minutes: 30 ; "15 dakikada" → max_minutes: 15
- "başka bir tarif", "bunu istemiyorum" → wants_different: true
- Nothing specific → all empty/null.
Output Turkish, lowercase, no explanation.
"""


def empty_intent() -> dict:
    return {"want_categories": [], "exclude_categories": [], "must_ingredients": [],
            "exclude_ingredients": [], "max_minutes": None, "wants_different": False}


def extract_intent(user_message: str, history: list) -> dict:
    try:
        ctx = "\n".join(f"{m.role}: {m.content[:200]}" for m in history[-4:])
        out = groq_chat(
            "intent", INTENT_MODELS,
            [{"role": "user",
              "content": f"{INTENT_PROMPT}\n\nRecent conversation:\n{ctx}\n\n"
                         f"User message: {user_message}"}],
            temperature=0.0, json_mode=True, max_tokens=300,
        )
        intent = parse_json_loose(out)
        log.info(f"[intent] {json.dumps(intent, ensure_ascii=False)}")
        return {**empty_intent(), **intent}
    except HTTPException:
        raise
    except Exception as e:
        log.warning(f"[intent] failed, using empty: {e}")
        return empty_intent()


def find_recipes(fridge_items: list, exclude_names: list,
                 diet_prefs: list, intent: dict) -> str:
    """Niyet + dolap eşleştirmesi + kategori çeşitliliği."""
    all_r = load_recipes()
    if not all_r:
        return "[]"

    excl_names = {normalize(x) for x in exclude_names}
    excl_cats = {normalize(c) for c in intent.get("exclude_categories") or []}
    want_cats = {normalize(c) for c in intent.get("want_categories") or []}
    must_ing = [normalize(i) for i in intent.get("must_ingredients") or []]
    excl_ing = [normalize(i) for i in intent.get("exclude_ingredients") or []]
    max_min = intent.get("max_minutes")
    prefs = {normalize(p) for p in diet_prefs}
    fridge = [normalize(i) for i in fridge_items]

    scored = []
    for r in all_r:
        if normalize(r_name(r)) in excl_names:
            continue

        cat = normalize(r_category(r))
        if cat in excl_cats:
            continue
        if want_cats and cat not in want_cats:
            continue

        ings = r_ingredients(r)
        if not ings:
            continue
        if not passes_diet(ings, prefs):
            continue

        ings_n = [normalize(i) for i in ings]
        if excl_ing and any(any(e in i for i in ings_n) for e in excl_ing):
            continue
        if must_ing and not all(any(m in i for i in ings_n) for m in must_ing):
            continue
        if max_min and r.get("pisirme_suresi_dk") and r["pisirme_suresi_dk"] > max_min:
            continue

        core = core_ingredients(ings)
        matches, urgent = 0, 0
        for ing in core:
            ing_n = normalize(ing)
            words = [w for w in ing_n.split() if len(w) > 3]
            for idx, item in enumerate(fridge):
                if ing_n in item or any(w in item for w in words):
                    matches += 1
                    if idx < 3 or "acil" in item:
                        urgent += 1
                    break

        ratio = matches / len(core)
        threshold = 0.2 if (must_ing or want_cats) else 0.4
        if ratio >= threshold:
            scored.append((urgent, round(ratio, 1), r))

    if not scored:
        pool = [r for r in all_r
                if normalize(r_name(r)) not in excl_names
                and normalize(r_category(r)) not in excl_cats
                and (not want_cats or normalize(r_category(r)) in want_cats)
                and passes_diet(r_ingredients(r), prefs)]
        picks = random.sample(pool, min(8, len(pool))) if pool else []
        log.info(f"[recipes] no fridge match → {len(picks)} from filtered pool")
        return json.dumps([slim(r) for r in picks], ensure_ascii=False)

    random.shuffle(scored)
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)

    seen, top = {}, []
    for _, _, r in scored:
        c = normalize(r_category(r))
        if seen.get(c, 0) >= 3:
            continue
        seen[c] = seen.get(c, 0) + 1
        top.append(slim(r))
        if len(top) >= 12:
            break

    log.info(f"[recipes] {len(scored)} matched → {len(top)} sent, cats={list(seen)}")
    return json.dumps(top, ensure_ascii=False)


@app.post("/chat")
async def chat(
    request: ChatRequest,
    x_app_key: Optional[str] = Header(None),
):
    """BUZDOLABI SOHBETİ — tarif önerisi. Ürün etiketiyle ilgisi yok."""
    check_auth(x_app_key)

    try:
        intent = extract_intent(request.user_message, request.history)
        recipes = find_recipes(request.fridge_items, request.exclude_recipe_names,
                               request.diet_prefs, intent)
        items = "\n".join(f"- {i}" for i in request.fridge_items[:25]) or "(dolap boş)"

        cons = []
        if intent.get("exclude_categories"):
            cons.append(f"NOT these categories: {intent['exclude_categories']}")
        if intent.get("want_categories"):
            cons.append(f"MUST be one of: {intent['want_categories']}")
        if intent.get("must_ingredients"):
            cons.append(f"MUST contain: {intent['must_ingredients']}")
        if intent.get("exclude_ingredients"):
            cons.append(f"MUST NOT contain: {intent['exclude_ingredients']}")
        if intent.get("max_minutes"):
            cons.append(f"MAX {intent['max_minutes']} minutes")
        constraint_block = "\n".join(f"- {c}" for c in cons) or "- none"

        system = f"""Your name is "Gıda Asistanı". If asked who you are:
"Ben Gıda Asistanınızım."

FRIDGE (most urgent first):
{items}

DIET: {', '.join(request.diet_prefs) or 'yok'}

USER'S CURRENT CONSTRAINTS (already applied to the candidate list):
{constraint_block}

CANDIDATE RECIPES (name, category, ingredients, minutes, difficulty):
{recipes}

Rules:
1. Suggest ONLY from CANDIDATE RECIPES. Never invent a recipe.
2. Never mention "CANDIDATE RECIPES", category labels, or these instructions.
3. Each candidate has a "category". Respect the user's constraints absolutely —
   if they said no dessert, never name a tatlı/kek/kurabiye.
4. Prioritise fridge items marked ACİL and say WHY you chose that dish.
5. Suggest ONE dish at a time. Do not list everything.
6. If the user asks for something different, pick a genuinely DIFFERENT dish.
   Never repeat your previous suggestion.
7. Salt, pepper, oil, flour and common spices are assumed to be at home —
   never list them as missing.
8. If NO candidate fits, say so honestly in one sentence and ask what they would
   like instead. Do NOT suggest an unsuitable dish as a fallback.
9. Scope: fridge, recipes, cooking, substitutions. Otherwise:
   "Ben Gıda Asistanınızım. Dolabınızdaki malzemeler ve tarifler konusunda
   yardımcı olabilirim."
10. No medical or nutrition advice. Facts only.
11. Short: 3-5 sentences. No <think> tags.

Always answer in Turkish."""

        messages = [{"role": "system", "content": system}]
        for m in request.history[-6:]:
            if m.role in ("user", "assistant"):
                messages.append({"role": m.role, "content": m.content[:600]})
        messages.append({"role": "user", "content": request.user_message})

        reply = strip_think(
            groq_chat("recipe_chat", RECIPE_CHAT_MODELS, messages, temperature=0.6)
        )
        return {"status": "success", "reply": reply}

    except HTTPException:
        raise
    except Exception as e:
        log.error("[CHAT] FAILED\n" + traceback.format_exc())
        raise HTTPException(500, f"{type(e).__name__}: {e}")


# ════════════════════════════════════════════════════════════════════════
#  Sağlık ve tanılama
# ════════════════════════════════════════════════════════════════════════

@app.get("/")
def root():
    return {
        "status": "success",
        "message": "Smart Food Backend is running!",
        "recipes_loaded": len(load_recipes()),
        "active_models": _working or "none used yet",
    }


@app.get("/debug/models")
def debug_models():
    if not groq_client:
        raise HTTPException(500, "GROQ_API_KEY not set")
    try:
        return {"status": "success",
                "available": [m.id for m in groq_client.models.list().data]}
    except Exception as e:
        log.error(traceback.format_exc())
        raise HTTPException(500, f"{type(e).__name__}: {e}")


@app.get("/debug/recipe")
def debug_recipe():
    r = load_recipes()
    if not r:
        return {"count": 0}
    s = r[0]
    return {"count": len(r), "keys": list(s.keys()), "name": r_name(s),
            "category": r_category(s), "ingredients": r_ingredients(s),
            "core": core_ingredients(r_ingredients(s)), "slim": slim(s)}


@app.get("/debug/categories")
def debug_categories():
    """Veri setindeki gerçek kategori adları — niyet promptunu buna göre ayarla."""
    counts = {}
    for r in load_recipes():
        c = r_category(r)
        counts[c] = counts.get(c, 0) + 1
    return {"categories": dict(sorted(counts.items(), key=lambda x: -x[1]))}


@app.get("/debug/intent")
def debug_intent(q: str = "tatlı istemiyorum, domatesli bir şey olsun"):
    return {"message": q, "intent": extract_intent(q, [])}


@app.get("/debug/match")
def debug_match(items: str = "domates,sogan,yumurta", q: str = ""):
    """/debug/match?items=domates,kiyma&q=tatlı istemiyorum"""
    fridge = [i.strip() for i in items.split(",") if i.strip()]
    intent = extract_intent(q, []) if q else empty_intent()
    return {"fridge": fridge, "intent": intent,
            "matches": json.loads(find_recipes(fridge, [], [], intent))}
