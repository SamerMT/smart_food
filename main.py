"""
Smart Food API

Model dağılımı — her modelin Groq'ta AYRI TPM kotası var, iş bölümü
günlük kotayı üçe böler:

  Qwen 3.6 27B (Groq)    → SADECE etiket OCR (düz metin, JSON yok)
  Llama 3.3 70B (Groq)   → tüm sohbetler, etiket yapılandırma, niyet çıkarımı
  Gemini 3.6 Flash       → SADECE buzdolabı görüntü tanıma

İki ayrı akış, hiçbir noktada karışmazlar:
  A) ETİKET:  /analyze-label (Qwen OCR → Llama JSON) → /product-chat (Llama)
  B) DOLAP:   /analyze-fridge (Gemini)               → /chat (Llama)
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
import google.generativeai as genai

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("smart-food")

app = FastAPI(title="Smart Food API")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
APP_SECRET = os.getenv("APP_SECRET")

# max_retries=0 → Groq'un kendi 40sn beklemesini kapat, 429'u biz yönetelim
groq_client = Groq(api_key=GROQ_API_KEY, max_retries=0) if GROQ_API_KEY else None

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

QWEN_OCR = "qwen/qwen3.6-27b"                # sadece OCR
LLAMA_CHAT = "llama-3.3-70b-versatile"       # tüm sohbet + JSON işleri
GEMINI_VISION = "models/gemini-3.6-flash"    # sadece dolap

# Groq TPM limiti 8000 → Qwen'e aynı anda en fazla 2 görsel gönderilebilir
MAX_LABEL_IMAGES = 2
MAX_FRIDGE_IMAGES = 4

_RECIPES_CACHE = None


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
    raw_text: str = ""
    diet_prefs: list[str] = []
    allergens: list[str] = []
    history: list[ChatMessage] = []


# ── Yardımcılar ─────────────────────────────────────────────────────────

def check_auth(x_app_key: Optional[str]):
    if APP_SECRET and x_app_key != APP_SECRET:
        raise HTTPException(401, "Unauthorized")


def normalize(text) -> str:
    t = str(text).lower()
    for a, b in [("ı", "i"), ("ş", "s"), ("ğ", "g"), ("ü", "u"), ("ö", "o"), ("ç", "c")]:
        t = t.replace(a, b)
    return unicodedata.normalize("NFKD", t).strip()


def strip_think(text: str) -> str:
    """Qwen düşünce çıktısını temizle."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = text.replace("<think>", "").replace("</think>", "")
    for marker in ("thinking process:", "thought process:", "thinking:",
                   "thought:", "reasoning:", "analysis:"):
        i = text.lower().find(marker)
        if i != -1:
            rest = text[i + len(marker):]
            parts = rest.split("\n\n", 1)
            text = text[:i] + (parts[1] if len(parts) > 1 else "")
    return text.strip()


def parse_json_loose(text: str) -> dict:
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


def _groq(model: str, messages: list, temperature: float,
          json_mode: bool = False, max_tokens: Optional[int] = None,
          hide_reasoning: bool = False) -> str:
    if not groq_client:
        raise HTTPException(500, "GROQ_API_KEY is not configured.")
    try:
        kwargs = dict(model=model, messages=messages, temperature=temperature)
        if hide_reasoning:
            kwargs["reasoning_format"] = "hidden"
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if max_tokens:
            kwargs["max_tokens"] = max_tokens
        res = groq_client.chat.completions.create(**kwargs)
        return res.choices[0].message.content or ""
    except Exception as e:
        msg = str(e)
        # 413 = tek istek TPM limitini aştı, 429 = kota doldu → ikisi de kullanıcıya aynı
        if "429" in msg or "413" in msg or "rate limit" in msg.lower():
            log.warning(f"[{model}] RATE LIMIT: {msg[:200]}")
            raise HTTPException(
                429,
                "Yapay zeka kotası doldu. Lütfen bir dakika sonra tekrar deneyin.")
        log.error(f"[{model}] {type(e).__name__}: {msg[:400]}")
        raise


def qwen_ocr(messages: list, max_tokens: int = 2000) -> str:
    """Qwen — sadece OCR. Düz metin çıktı, json_mode YOK."""
    return strip_think(_groq(QWEN_OCR, messages, 0.0,
                             max_tokens=max_tokens, hide_reasoning=True))


def llama(messages: list, temperature: float = 0.3,
          json_mode: bool = False, max_tokens: Optional[int] = None) -> str:
    """Llama — tüm sohbetler, JSON yapılandırma, niyet çıkarımı."""
    return _groq(LLAMA_CHAT, messages, temperature,
                 json_mode=json_mode, max_tokens=max_tokens)


async def read_image(img: UploadFile) -> tuple:
    data = await img.read()
    log.info(f"  image {img.filename} {img.content_type} {len(data)} bytes")
    if not data:
        raise ValueError(f"Empty image: {img.filename}")
    return data, (img.content_type or "image/jpeg")


def qwen_vision_message(prompt: str, images: list) -> list:
    content = [{"type": "text", "text": prompt}]
    for data, mime in images:
        b64 = base64.b64encode(data).decode()
        content.append({"type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{b64}"}})
    return [{"role": "user", "content": content}]


def gemini_vision(prompt: str, images: list) -> str:
    """Sadece buzdolabı görüntü tanıma."""
    if not GEMINI_API_KEY:
        raise HTTPException(500, "GEMINI_API_KEY is not configured.")
    parts = [prompt]
    for data, mime in images:
        parts.append({"mime_type": mime, "data": data})
    res = genai.GenerativeModel(GEMINI_VISION).generate_content(parts)
    return res.text or ""


# ════════════════════════════════════════════════════════════════════════
#  A) ETİKET AKIŞI — Qwen OCR + Llama yapılandırma/sohbet
# ════════════════════════════════════════════════════════════════════════

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

Return ONLY a JSON object, no markdown:

{
  "product_name": "Turkish name, or null if unreadable",
  "brand": "brand or null",
  "ingredients": ["Turkish ingredient strings, [] if none"],
  "allergens": ["ARRAY of keys actually present"],
  "contains_lactose": false,
  "contains_pork": false,
  "contains_alcohol": false,
  "category": "one of: Sebze, Meyve, Süt Ürünleri, Protein, Tahıl, İçecek, Baharat, Donuk, Diğer",
  "net_weight": "e.g. 500 g, or null",
  "expiry_date": "YYYY-MM-DD or null",
  "storage": "one of: Buzdolabı, Dondurucu, Oda Sıcaklığı, Kiler",
  "nutrition": {
    "serving": null, "energy_kcal": null, "fat_g": null,
    "saturated_fat_g": null, "carbs_g": null, "sugar_g": null,
    "protein_g": null, "salt_g": null, "fiber_g": null
  },
  "confidence": 0,
  "summary": "ONE short Turkish sentence"
}

Rules:
- allergens: allowed keys ONLY — sut, yumurta, gluten, findik, fistik, soya,
  susam, balik, kabuklu_deniz, hardal, kereviz
  With milk and hazelnut: ["sut","findik"]. With none: [].
  NEVER an object with true/false. ALWAYS an array of strings.
- contains_pork: domuz, jambon, bacon, lard, unknown-origin gelatin.
- nutrition: read numbers from the table. Missing → null. Numbers only, no units.
- confidence: 0-100, how readable the label was.

OCR TEXT:
"""


@app.post("/analyze-label")
async def analyze_label(
    images: list[UploadFile] = File(...),
    x_app_key: Optional[str] = Header(None),
):
    """ETİKET — Qwen OCR (düz metin) → Llama yapılandırma (JSON)."""
    check_auth(x_app_key)
    log.info(f"[LABEL] {len(images)} image(s)")

    try:
        imgs = [await read_image(i) for i in images[:MAX_LABEL_IMAGES]]

        try:
            raw_text = qwen_ocr(qwen_vision_message(OCR_PROMPT, imgs), max_tokens=2500)
            log.info(f"[LABEL] Qwen OCR {len(raw_text)} chars:\n{raw_text[:2000]}")
        except HTTPException as e:
            if e.status_code == 429 and GEMINI_API_KEY:
                log.warning("[LABEL] Qwen limit asildi → Gemini yedegi")
                raw_text = strip_think(gemini_vision(OCR_PROMPT, imgs))
                log.info(f"[LABEL] Gemini OCR {len(raw_text)} chars")
            else:
                raise

        if not raw_text.strip():
            return {"status": "success", "data": {
                "product_name": None, "ingredients": [], "allergens": [],
                "contains_lactose": False, "contains_pork": False,
                "contains_alcohol": False, "category": "Diğer",
                "storage": "Buzdolabı", "expiry_date": None, "nutrition": {},
                "confidence": 0, "summary": "Etiket okunamadı.", "raw_text": "",
            }}

        data = parse_json_loose(
            llama([{"role": "user",
                    "content": LABEL_STRUCTURE_PROMPT + raw_text[:5000]}],
                  temperature=0.1, json_mode=True)
        )

        al = data.get("allergens")
        if isinstance(al, dict):
            data["allergens"] = [k for k, v in al.items() if v is True]
        elif not isinstance(al, list):
            data["allergens"] = []

        data["raw_text"] = raw_text[:6000]
        log.info(f"[LABEL] name={data.get('product_name')} conf={data.get('confidence')}")
        return {"status": "success", "data": data}

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
    """ÜRÜN SOHBETİ — Llama. Sadece taranan ürün hakkında."""
    check_auth(x_app_key)

    try:
        p = request.product or {}
        summary = json.dumps({
            "name": p.get("product_name"), "brand": p.get("brand"),
            "ingredients": (p.get("ingredients") or [])[:30],
            "allergens": p.get("allergens") or [],
            "contains_lactose": p.get("contains_lactose"),
            "contains_pork": p.get("contains_pork"),
            "contains_alcohol": p.get("contains_alcohol"),
            "category": p.get("category"), "net_weight": p.get("net_weight"),
            "storage": p.get("storage"), "expiry_date": p.get("expiry_date"),
            "nutrition": p.get("nutrition") or {},
        }, ensure_ascii=False)

        raw = (request.raw_text or p.get("raw_text") or "")[:4000]

        system = f"""Your name is "Gıda Asistanı".

The user scanned this product's packaging. Discuss ONLY this product.

STRUCTURED DATA:
{summary}

FULL LABEL TEXT (as printed — use this for calories, nutrition table,
weights and dates):
---
{raw}
---

USER PROFILE:
- Diet preferences: {', '.join(request.diet_prefs) or 'yok'}
- Allergens: {', '.join(request.allergens) or 'yok'}

Rules:
1. Answer ONLY about this product: ingredients, nutrition, calories, allergens,
   storage, weight, dates, and whether it fits the user's profile.
2. For calories or nutrition read the FULL LABEL TEXT. If a value is not printed
   there, say it is not on the label — never invent a number.
3. If asked whether it suits them, answer clearly uygun / uygun değil /
   emin değilim and name the exact ingredient that decides it.
4. NEVER suggest recipes. NEVER mention the fridge. NEVER offer to add it anywhere.
5. Facts only. No medical or nutrition advice.
6. If the label was unreadable, say so and suggest rescanning closer.
7. Out of scope → "Ben taradığınız ürün hakkında yardımcı olabilirim."
8. Short: 2-4 sentences.

Always answer in Turkish."""

        messages = [{"role": "system", "content": system}]
        for m in request.history[-6:]:
            if m.role in ("user", "assistant"):
                messages.append({"role": m.role, "content": m.content[:600]})
        messages.append({"role": "user", "content": request.user_message})

        return {"status": "success", "reply": llama(messages, 0.3)}

    except HTTPException:
        raise
    except Exception as e:
        log.error("[PRODUCT-CHAT] FAILED\n" + traceback.format_exc())
        raise HTTPException(500, f"{type(e).__name__}: {e}")


# ════════════════════════════════════════════════════════════════════════
#  B) DOLAP AKIŞI — Gemini görüntü + Llama sohbet
# ════════════════════════════════════════════════════════════════════════

FRIDGE_PROMPT = """You are looking INSIDE a refrigerator or at grocery items.

Identify every distinct food item, drink and package visible across the images.
Read visible labels to name products accurately. If the same item appears in more
than one image, count it once.

Return ONLY a JSON object, no markdown, no commentary:

{"items":[{"name":"Domates","category":"Sebze","freshness_points":80,"quantity":5,"unit":"adet"}]}

Rules:
- name: Turkish with proper characters (Süt not Sut). Be specific when the label
  is readable, e.g. "Laktozsuz Süt" rather than just "Süt".
- category: exactly one of Sebze, Meyve, Süt Ürünleri, Protein, Tahıl, İçecek,
  Baharat, Donuk, Diğer
- freshness_points: 0-100, how fresh it LOOKS (colour, wilting, packaging).
  Cannot judge → 85. This is NOT shelf life.
- quantity: a number. unit: adet, paket, kap, gram, ml, kg or litre.
- Food and drink only. Do not list shelves, containers or the fridge itself.
"""


@app.post("/analyze-fridge")
async def analyze_fridge(
    images: list[UploadFile] = File(...),
    x_app_key: Optional[str] = Header(None),
):
    """DOLAP — Gemini 3.6 Flash ile ürün tanıma."""
    check_auth(x_app_key)
    log.info(f"[FRIDGE] {len(images)} image(s)")

    try:
        imgs = [await read_image(i) for i in images[:MAX_FRIDGE_IMAGES]]
        raw = gemini_vision(FRIDGE_PROMPT, imgs)
        log.info(f"[FRIDGE] gemini raw: {raw[:400]}")

        data = parse_json_loose(raw)
        items = data.get("items") or []
        log.info(f"[FRIDGE] detected {len(items)} items")
        return {"status": "success", "data": {"items": items}}

    except HTTPException:
        raise
    except Exception as e:
        log.error("[FRIDGE] FAILED\n" + traceback.format_exc())
        raise HTTPException(500, f"{type(e).__name__}: {e}")


# ── Tarif veri seti (Türkçe alanlar) ────────────────────────────────────

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
        "pastirma", "salam", "sosis", "ciger", "kusbasi", "but", "kanat"}
DAIRY = {"sut", "peynir", "yogurt", "kaymak", "krema", "tereyagi", "labne",
         "ayran", "kasar", "lor", "cokelek"}
ANIMAL = MEAT | DAIRY | {"yumurta", "jelatin"}
HARAM = {"domuz", "jambon", "bacon", "alkol", "sarap", "likor", "rom", "bira"}


def has_any(ings_norm: list, words: set) -> bool:
    return any(i in words or set(i.split()) & words for i in ings_norm)


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
    return {"name": r_name(r), "category": r_category(r),
            "ingredients": r_ingredients(r)[:10],
            "minutes": r.get("pisirme_suresi_dk"), "difficulty": r.get("zorluk")}


# ── Niyet çıkarımı — Llama ──────────────────────────────────────────────

INTENT_PROMPT = """Extract the user's cooking intent from their message.
Use the recent conversation for context.

Return ONLY JSON:
{"want_categories":[],"exclude_categories":[],"must_ingredients":[],
 "exclude_ingredients":[],"max_minutes":null,"wants_different":false}

The database has EXACTLY these 8 categories. Use ONLY these strings,
never invent others:
  "ana yemek", "tatlı", "salata", "çorba", "kahvaltı",
  "atıştırmalık", "içecek", "turşu"

Guidance:
- "tatlı istemiyorum", "şekerli olmasın" → exclude_categories: ["tatlı"]
- "yemek istiyorum", "akşam yemeği" → want_categories: ["ana yemek"]
- "çorba istiyorum" → want_categories: ["çorba"]
- "hafif bir şey" → want_categories: ["salata"]
- "kahvaltılık" → want_categories: ["kahvaltı"]
- "domatesli bir şey" → must_ingredients: ["domates"]
- "etsiz olsun" → exclude_ingredients: ["et","kıyma","tavuk"]
- "hızlı olsun" → max_minutes: 30 ; "15 dakikada" → max_minutes: 15
- "başka bir tarif" → wants_different: true
- Nothing specific → all empty/null.
Turkish, lowercase, no explanation.
"""


def empty_intent() -> dict:
    return {"want_categories": [], "exclude_categories": [], "must_ingredients": [],
            "exclude_ingredients": [], "max_minutes": None, "wants_different": False}


def extract_intent(user_message: str, history: list) -> dict:
    try:
        ctx = "\n".join(f"{m.role}: {m.content[:150]}" for m in history[-4:])
        out = llama([{"role": "user",
                      "content": f"{INTENT_PROMPT}\n\nConversation:\n{ctx}\n\n"
                                 f"User message: {user_message}"}],
                    temperature=0.0, json_mode=True, max_tokens=300)
        intent = parse_json_loose(out)
        log.info(f"[intent] {json.dumps(intent, ensure_ascii=False)}")
        return {**empty_intent(), **intent}
    except HTTPException:
        raise
    except Exception as e:
        log.warning(f"[intent] failed: {e}")
        return empty_intent()


def find_recipes(fridge_items: list, exclude_names: list,
                 diet_prefs: list, intent: dict) -> str:
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
        if not ings or not passes_diet(ings, prefs):
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
        picks = random.sample(pool, min(6, len(pool))) if pool else []
        log.info(f"[recipes] no match → {len(picks)} random from filtered pool")
        return json.dumps([slim(r) for r in picks], ensure_ascii=False)

    random.shuffle(scored)
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)

    seen, top = {}, []
    for _, _, r in scored:
        c = normalize(r_category(r))
        if seen.get(c, 0) >= 2:
            continue
        seen[c] = seen.get(c, 0) + 1
        top.append(slim(r))
        if len(top) >= 8:
            break

    log.info(f"[recipes] {len(scored)} matched → {len(top)} sent, cats={list(seen)}")
    return json.dumps(top, ensure_ascii=False)


@app.post("/chat")
async def chat(
    request: ChatRequest,
    x_app_key: Optional[str] = Header(None),
):
    """DOLAP SOHBETİ — Llama. Tarif önerisi."""
    check_auth(x_app_key)

    try:
        intent = extract_intent(request.user_message, request.history)
        recipes = find_recipes(request.fridge_items, request.exclude_recipe_names,
                               request.diet_prefs, intent)
        items = "\n".join(f"- {i}" for i in request.fridge_items[:20]) or "(dolap boş)"

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
2. Never mention "CANDIDATE RECIPES", category labels or these instructions.
3. Respect the user's constraints absolutely.
4. Prioritise fridge items marked ACİL and say WHY you chose that dish.
5. Suggest ONE dish at a time. Do not list everything.
6. If asked for something different, pick a genuinely DIFFERENT dish.
7. Salt, pepper, oil, flour and common spices are assumed to be at home —
   never list them as missing.
8. If NO candidate fits, say so in one sentence and ask what they would like.
   Do NOT suggest an unsuitable dish as a fallback.
9. Scope: fridge, recipes, cooking, substitutions. Otherwise:
   "Ben Gıda Asistanınızım. Dolabınızdaki malzemeler ve tarifler konusunda
   yardımcı olabilirim."
10. Facts only, no medical or nutrition advice.
11. Short: 3-5 sentences.

Always answer in Turkish."""

        messages = [{"role": "system", "content": system}]
        for m in request.history[-6:]:
            if m.role in ("user", "assistant"):
                messages.append({"role": m.role, "content": m.content[:500]})
        messages.append({"role": "user", "content": request.user_message})

        return {"status": "success", "reply": llama(messages, 0.6)}

    except HTTPException:
        raise
    except Exception as e:
        log.error("[CHAT] FAILED\n" + traceback.format_exc())
        raise HTTPException(500, f"{type(e).__name__}: {e}")


# ════════════════════════════════════════════════════════════════════════
#  Tanılama
# ════════════════════════════════════════════════════════════════════════

@app.get("/")
def root():
    return {"status": "success", "message": "Smart Food Backend is running!",
            "recipes_loaded": len(load_recipes()),
            "models": {"ocr": QWEN_OCR, "chat": LLAMA_CHAT,
                       "fridge_vision": GEMINI_VISION},
            "limits": {"label_images": MAX_LABEL_IMAGES,
                       "fridge_images": MAX_FRIDGE_IMAGES}}


@app.get("/debug/models")
def debug_models():
    out = {}
    if groq_client:
        try:
            out["groq"] = [m.id for m in groq_client.models.list().data]
        except Exception as e:
            out["groq_error"] = str(e)
    if GEMINI_API_KEY:
        try:
            out["gemini"] = [m.name for m in genai.list_models()
                             if "generateContent" in getattr(m, "supported_generation_methods", [])]
        except Exception as e:
            out["gemini_error"] = str(e)
    return out


@app.get("/debug/recipe")
def debug_recipe():
    r = load_recipes()
    if not r:
        return {"count": 0}
    s = r[0]
    return {"count": len(r), "name": r_name(s), "category": r_category(s),
            "ingredients": r_ingredients(s),
            "core": core_ingredients(r_ingredients(s)), "slim": slim(s)}


@app.get("/debug/categories")
def debug_categories():
    counts = {}
    for r in load_recipes():
        c = r_category(r)
        counts[c] = counts.get(c, 0) + 1
    return {"categories": dict(sorted(counts.items(), key=lambda x: -x[1]))}


@app.get("/debug/intent")
def debug_intent(q: str = "tatlı istemiyorum"):
    return {"message": q, "intent": extract_intent(q, [])}


@app.get("/debug/match")
def debug_match(items: str = "domates,sogan", q: str = ""):
    fridge = [i.strip() for i in items.split(",") if i.strip()]
    intent = extract_intent(q, []) if q else empty_intent()
    return {"fridge": fridge, "intent": intent,
            "matches": json.loads(find_recipes(fridge, [], [], intent))}

#test
@app.post("/debug/ocr")
async def debug_ocr(images: list[UploadFile] = File(...)):
    """Sadece OCR — yapılandırma yok. Qwen'in gerçekte ne okuduğunu gösterir."""
    imgs = [await read_image(i) for i in images[:2]]
    sizes = [len(d) for d, _ in imgs]
    try:
        text = qwen_ocr(qwen_vision_message(OCR_PROMPT, imgs), max_tokens=2500)
        return {"model": "qwen", "image_bytes": sizes,
                "chars": len(text), "text": text}
    except HTTPException as e:
        if GEMINI_API_KEY:
            text = gemini_vision(OCR_PROMPT, imgs)
            return {"model": "gemini_fallback", "qwen_error": e.detail,
                    "image_bytes": sizes, "chars": len(text), "text": text}
        raise
