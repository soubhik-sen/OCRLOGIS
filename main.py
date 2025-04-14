# main.py
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from typing import List
import tempfile
import fitz  # PyMuPDF
from ocr_utils import extract_text_from_pdf, fuse_ocr_outputs, search_texts
import shutil
import json

app = FastAPI()

@app.post("/search-document")
async def search_document(file: UploadFile = File(...), search_list: str = Form(...)):
    # Parse search list
    try:
        search_list = json.loads(search_list)
    except json.JSONDecodeError:
        return JSONResponse(status_code=400, content={"error": "search_list must be a valid JSON list"})

    # Save uploaded PDF to a temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    # Extract text using PaddleOCR and EasyOCR
    easy_text, paddle_text = extract_text_from_pdf(tmp_path)
    fused_text = fuse_ocr_outputs(easy_text, paddle_text)
    search_result = search_texts(fused_text, search_list)

    return {"fused_text": fused_text, "search_result": search_result}