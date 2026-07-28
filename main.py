import os
import json
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import google.generativeai as genai
from groq import Groq

# Initialize FastAPI application
app = FastAPI(title="Smart Food API")

# Fetch API keys from environment variables (configured in Render settings)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Configure model connections if keys are available
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

if GROQ_API_KEY:
    groq_client = Groq(api_key=GROQ_API_KEY)

# Data model for receiving chat requests from the mobile application
class ChatRequest(BaseModel):
    user_message: str
    fridge_items: list[str]  # List of current items inside the fridge

def find_matching_recipes(fridge_items: list[str]) -> str:
    """
    Helper function to read the local recipe dataset and filter it.
    Currently, it loads the data and returns a chunk of it to be injected 
    into the LLM context. You can later upgrade this to implement a strict 
    matching algorithm based on the freshness points and fridge items.
    """
    try:
        # Loading the dataset provided by the user
        with open("recipes_groq_cleaned.json", "r", encoding="utf-8") as file:
            all_recipes = json.load(file)
            
        # Example logic: Return the first 5 recipes to avoid token limit exceptions.
        # Once your algorithm is ready, replace this with your filtered list.
        suggested_recipes = all_recipes[:5] 
        return json.dumps(suggested_recipes, ensure_ascii=False)
    except FileNotFoundError:
        return "No local dataset found. Proceed using general knowledge."
    except Exception as e:
        return f"Error reading dataset: {str(e)}"

@app.get("/")
def read_root():
    return {"status": "success", "message": "Smart Food Backend is running!"}

@app.post("/analyze-fridge")
async def analyze_fridge(image: UploadFile = File(...)):
    """
    This endpoint receives an image from the mobile app, sends it to Gemini for vision analysis,
    and then forwards the resulting text to Groq (Llama 3.3) for cleaning and categorization.
    """
    if not GEMINI_API_KEY or not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="API Keys are not configured on the server.")
        
    try:
        # 1. Read image data
        image_data = await image.read()
        
        # 2. Send image to Gemini 1.5 Flash (fastest and most efficient for vision tasks)
        gemini_model = genai.GenerativeModel('gemini-1.5-flash')
        vision_prompt = "List all food items, drinks, and packages you see in this image. Read labels carefully."
        
        vision_response = gemini_model.generate_content([
            vision_prompt,
            {"mime_type": image.content_type, "data": image_data}
        ])
        raw_items_text = vision_response.text
        
        # 3. Forward the raw text to Llama 3.3 for cleaning and categorization
        groq_prompt = f"""
        You are a data cleaner. I will give you raw text describing fridge items from an AI vision model.
        Task:
        1. Clean the list and remove duplicates.
        2. Translate item names to Turkish.
        3. Categorize them into main categories (e.g., Sebze, Meyve, Süt Ürünleri, İçecekler, Atıştırmalıklar).
        4. Assign a hypothetical freshness score (0-100) based on typical shelf life to help our recipe AI later.
        
        CRITICAL: Return ONLY a valid JSON object. No markdown, no explanations.
        JSON Format must be exactly like this:
        {{"items": [{{"name": "Domates", "category": "Sebze", "freshness_points": 80}}]}}
        
        Raw text: {raw_items_text}
        """
        
        chat_completion = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": groq_prompt}],
            model="llama-3.3-70b-versatile",
            temperature=0.1, # Very low temperature to ensure strict JSON formatting
            response_format={"type": "json_object"} 
        )
        
        clean_json = json.loads(chat_completion.choices[0].message.content)
        
        return {"status": "success", "data": clean_json}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat_bot(request: ChatRequest):
    """
    This endpoint receives the user's message alongside the current fridge inventory,
    reads the dataset, and suggests Turkish recipes based on the available ingredients.
    """
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="Groq API Key is not configured.")
        
    try:
        items_str = ", ".join(request.fridge_items)
        
        # 1. Fetch recipes from the JSON dataset
        dataset_recipes = find_matching_recipes(request.fridge_items)
        
        # 2. Construct the RAG (Retrieval-Augmented Generation) prompt
        system_prompt = f"""
        You are an expert Turkish Chef AI.
        The user currently has these items in their fridge: {items_str}.
        
        DATABASE RECIPES TO USE:
        {dataset_recipes}
        
        Your tasks:
        1. If the user asks for what to cook, ONLY suggest dishes from the 'DATABASE RECIPES TO USE' provided above.
        2. Answer their cooking-related questions.
        3. Prioritize items that typically spoil fast (if mentioned in the fridge list).
        4. Suggest alternatives if they are missing a specific ingredient required by the database recipe.
        
        CRITICAL: Always answer in clear, polite Turkish. Do not use English in your response.
        """
        
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": request.user_message}
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.5, # Lowered temperature to enforce adherence to the provided dataset
        )
        
        return {"status": "success", "reply": chat_completion.choices[0].message.content}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
