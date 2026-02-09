import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import time
from utils import translator, build_transformer

app = FastAPI(title="English to Hindi Translator")

app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")
templates = Jinja2Templates(directory="frontend")

transformer = None
MODEL_WEIGHTS_PATH = "en_hi_weights.h5"

class TranslationRequest(BaseModel):
    text: str

class TranslationResponse(BaseModel):
    translation: str
    processing_time: float
    note: str
    model_info: dict

def get_model():
    global transformer
    if transformer is None:
        if os.path.exists(MODEL_WEIGHTS_PATH):
            transformer = build_transformer()
            transformer.load_weights(MODEL_WEIGHTS_PATH)
        else:
            raise HTTPException(status_code=503, detail="Model weights not found")
    return transformer

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/translate", response_model=TranslationResponse)
async def translate(req: TranslationRequest):
    model = get_model()
    start = time.time()
    translation, note = translator(req.text, model)
    return TranslationResponse(
        translation=translation,
        processing_time=round(time.time() - start, 3),
        note=note,
        model_info={"model": "Transformer"}
    )
