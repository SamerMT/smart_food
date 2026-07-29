import os
import json
import unicodedata
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, Header
from pydantic import BaseModel
import google.generativeai as genai
from groq import Groq

app = FastAPI(title="Smart Food API")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
APP_SECRET = os.getenv("APP_SECRET")  # opsiyonel basit koruma

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

if GROQ_API_KEY:
    groq_client = Groq(api_key=GROQ_API_KEY)

VISION_MODEL = "gemini-3.6-flash"
TEXT_MODEL = "llama-3.3-70b-versatile"


# ─────────────────────────── Modeller ───────────────────────────

class ChatMessage(BaseModel):
    role: str          # "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    user_message: str
    fridge_items: list[str] = []
    history: list[ChatMessage] = []       # YENİ: konuşma hafızası
    context_notes: list[str] = []         # YENİ: taranan etiket metinleri vb.


# ─────────────────────────── Yardımcılar ───────────────────────────

def check_auth(x_app_key: Optional[str]):
    """Basit koruma. APP_SECRET tanımlı değilse atlanır."""
    if APP_SECRET and x_app_key != APP_SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized")


def normalize(text: str) -> str:
    text = text.lower()
    for a, b in [("ı", "i"), ("ş", "s"), ("ğ", "g"), ("ü", "u"), ("ö", "o"), ("ç", "c")]:
        text = text.replace(a, b)
    return unicodedata.normalize("NFKD", text)


def find_matching_recipes(fridge_items: list[str]) -> str:
    """
    Dolap içeriğine göre GERÇEK eşleştirme.
    fridge_items risk sırasına göre gelir (en riskli ilk sırada).
    """
    try:
        with open("recipes_groq_cleaned.json", "r", encoding="utf-8") as f:
            all_recipes = json.load(f)
    except Exception:
        return "No local dataset found. Proceed using general knowledge."

    if not fridge_items:
        return json.dumps(all_recipes[:5], ensure_ascii=False)

    fridge_norm = [normalize(i) for i in fridge_items]
    scored = []

    for recipe in all_recipes:
        ingredients = recipe.get("ingredients") or []
        if not ingredients:
            continue

        matches = 0
        urgent = 0

        for ing in ingredients:
            ing_norm = normalize(str(ing))
            words = [w for w in ing_norm.split() if len(w) > 3]

            for idx, item in enumerate(fridge_norm):
                if ing_norm in item or any(w in item for w in words):
                    matches += 1
                    # ilk 3 sıra = en riskli, ya da "acil" etiketli
                    if idx < 3 or "acil" in item:
                        urgent += 1
                    break

        ratio = matches / len(ingredients)
        if ratio >= 0.5:
            scored.append((urgent, ratio, recipe))

    # Önce acil malzeme sayısı, sonra eşleşme oranı
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    top = [r for _, _, r in scored[:6]]

    return json.dumps(top or all_recipes[:5], ensure_ascii=False)


# ─────────────────────────── Endpointler ───────────────────────────

@app.get("/")
def read_root():
    return {"status": "success", "message": "Smart Food Backend is running!"}


@app.post("/analyze-fridge")
async def analyze_fridge(
    images: list[UploadFile] = File(...),
    x_app_key: Optional[str] = Header(None),
):
    """Buzdolabı fotoğrafı/fotoğrafları → tespit edilen ürünler."""
    check_auth(x_app_key)

    if not GEMINI_API_KEY or not GROQ_API_KEY:
        raise HTTPException(500, "API Keys are not configured on the server.")

    try:
        parts = [
            "List every food item, drink and package you can see across these "
            "images. Read any visible labels carefully. If the same item appears "
            "in multiple images, list it only once. Include approximate quantity."
        ]

        for img in images[:4]:  # en fazla 4 görsel
            data = await img.read()
            parts.append({"mime_type": img.content_type, "data": data})

        vision = genai.GenerativeModel(VISION_MODEL)
        raw_text = vision.generate_content(parts).text

        groq_prompt = f"""
You are a data cleaner. Below is raw text describing fridge items from a vision model.

Tasks:
1. Clean the list, remove duplicates.
2. Translate item names to Turkish.
3. Categorize: Sebze, Meyve, Süt Ürünleri, Protein, Tahıl, İçecek, Baharat, Donuk, Diğer.
4. Assign a VISUAL freshness score (0-100) based ONLY on how fresh the item LOOKS
   (color, wilting, packaging condition, visible mold).
   Do NOT guess shelf life — the client app has a USDA database for that.
   If you cannot judge visually, return 85.
5. Estimate quantity and unit (adet, paket, kap, gram, ml, kg, litre).

CRITICAL: Return ONLY valid JSON. No markdown, no explanation.
Format:
{{"items":[{{"name":"Domates","category":"Sebze","freshness_points":80,"quantity":5,"unit":"adet"}}]}}

Raw text: {raw_text}
"""

        completion = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": groq_prompt}],
            model=TEXT_MODEL,
            temperature=0.1,
            response_format={"type": "json_object"},
        )

        return {
            "status": "success",
            "data": json.loads(completion.choices[0].message.content),
        }

    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/analyze-label")
async def analyze_label(
    images: list[UploadFile] = File(...),
    x_app_key: Optional[str] = Header(None),
):
    """
    YENİ: Ürün etiketi fotoğrafı/fotoğrafları (ön yüz + içindekiler).
    Gemini OCR olarak çalışır, Groq yapılandırır.
    """
    check_auth(x_app_key)

    if not GEMINI_API_KEY or not GROQ_API_KEY:
        raise HTTPException(500, "API Keys are not configured on the server.")

    try:
        parts = [
            "You are an OCR engine. Transcribe ALL text visible on this product "
            "packaging across the images, exactly as written. Include: product name, "
            "brand, ingredients list, allergen warnings, net weight, storage "
            "instructions, and any expiry / best-before date. Do not summarize, "
            "do not translate — output raw transcribed text only."
        ]

        for img in images[:3]:
            data = await img.read()
            parts.append({"mime_type": img.content_type, "data": data})

        vision = genai.GenerativeModel(VISION_MODEL)
        ocr_text = vision.generate_content(parts).text

        groq_prompt = f"""
You are a food label parser. Below is OCR text from a product package.

Tasks:
1. Extract the product name in Turkish.
2. Extract the full ingredients list as an array (Turkish).
3. Detect allergens. Use ONLY these keys:
   sut, yumurta, gluten, findik, fistik, soya, susam, balik, kabuklu_deniz, hardal, kereviz
4. Set contains_lactose true if milk/dairy is present.
5. Categorize: Sebze, Meyve, Süt Ürünleri, Protein, Tahıl, İçecek, Baharat, Donuk, Diğer.
6. Extract expiry date if visible, as ISO "YYYY-MM-DD". If not visible, null.
7. Extract storage instruction: Buzdolabı, Dondurucu, Oda Sıcaklığı, Kiler. Default Buzdolabı.
8. Give confidence 0-100 for how readable the label was.

CRITICAL: Return ONLY valid JSON. No markdown.
Format:
{{"product_name":"Laktozsuz Süt","ingredients":["Süt","Laktaz enzimi"],
"allergens":["sut"],"contains_lactose":false,"category":"Süt Ürünleri",
"expiry_date":"2026-08-15","storage":"Buzdolabı","confidence":92,
"raw_text":"..."}}

Set raw_text to the original OCR text so the client can keep it as context.

OCR text: {ocr_text}
"""

        completion = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": groq_prompt}],
            model=TEXT_MODEL,
            temperature=0.1,
            response_format={"type": "json_object"},
        )

        return {
            "status": "success",
            "data": json.loads(completion.choices[0].message.content),
        }

    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/chat")
async def chat_bot(
    request: ChatRequest,
    x_app_key: Optional[str] = Header(None),
):
    """Türk mutfağı asistanı. Konuşma hafızası ve tarama bağlamı destekler."""
    check_auth(x_app_key)

    if not GROQ_API_KEY:
        raise HTTPException(500, "Groq API Key is not configured.")

    try:
        items_str = "\n".join(f"- {i}" for i in request.fridge_items) or "(dolap boş)"
        recipes = find_matching_recipes(request.fridge_items)

        context_block = ""
        if request.context_notes:
            notes = "\n---\n".join(request.context_notes[-3:])
            context_block = f"""
RECENTLY SCANNED PRODUCT LABELS (the user scanned these; refer to them if asked):
{notes}
"""

        system_prompt = f"""
You are an expert Turkish home-cooking assistant inside a food-waste app.

USER'S FRIDGE (sorted by spoilage risk, most urgent first):
{items_str}
{context_block}
MATCHED RECIPES FROM DATABASE:
{recipes}

Rules:
1. Only suggest dishes from MATCHED RECIPES above. Never invent recipes.
2. Always prioritise items marked ACİL or with few days left. Say WHY you chose them.
3. Suggest substitutions when an ingredient is missing.
4. Stay strictly within scope: fridge contents, recipes, cooking, ingredient
   substitution, and scanned product labels. If asked anything else, reply:
   "Ben mutfak asistanınızım. Dolabınızdaki malzemeler ve tarifler konusunda
   yardımcı olabilirim."
5. Never give medical, allergy or nutrition advice. If a label shows an allergen,
   you may state the fact only ("Bu üründe süt var"), never advise.
6. Keep answers short — 3-5 sentences. This is a phone screen.

CRITICAL: Always answer in Turkish. Never use English.
"""

        messages = [{"role": "system", "content": system_prompt}]

        # Konuşma hafızası — son 10 mesaj
        for m in request.history[-10:]:
            if m.role in ("user", "assistant"):
                messages.append({"role": m.role, "content": m.content})

        messages.append({"role": "user", "content": request.user_message})

        completion = groq_client.chat.completions.create(
            messages=messages,
            model=TEXT_MODEL,
            temperature=0.4,
        )

        return {"status": "success", "reply": completion.choices[0].message.content}

    except Exception as e:
        raise HTTPException(500, str(e))
