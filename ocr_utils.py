# ocr_utils.py
import fitz  # PyMuPDF
from pdf2image import convert_from_path
from PIL import Image
import numpy as np
import easyocr
from paddleocr import PaddleOCR
from typing import List
from collections import Counter
import re

def get_easy_reader():
    return easyocr.Reader(['en'])

def get_paddle_reader():
    return PaddleOCR(use_angle_cls=True, lang='en')

def extract_text_from_pdf(pdf_path: str):
    images = convert_from_path(pdf_path, dpi=300)
    _easy_reader = get_easy_reader()
    _paddle_reader = get_paddle_reader()
    easy_text = ""
    paddle_text = ""

    for img in images:
        img_np = np.array(img.convert("RGB"))

        # EasyOCR
        easy_result = _easy_reader.readtext(img_np, detail=0)
        easy_text += "\n".join(easy_result) + "\n"

        # PaddleOCR
        paddle_result = _paddle_reader.ocr(img_np, cls=True)
        paddle_page = " ".join([line[1][0] for line in paddle_result[0]])
        paddle_text += paddle_page + "\n"

    return easy_text, paddle_text

def fuse_ocr_outputs(easy_text: str, paddle_text: str) -> str:
    easy_lines = set([line.strip() for line in easy_text.splitlines() if line.strip()])
    paddle_lines = set([line.strip() for line in paddle_text.splitlines() if line.strip()])
    combined_lines = list(easy_lines.union(paddle_lines))
    return "\n".join(sorted(combined_lines))

def search_texts(fused_text: str, search_list: List[str]):
    result = []
    text_lower = fused_text.lower()
    for keyword in search_list:
        found = keyword.lower() in text_lower
        result.append({"text": keyword, "found": found})
    return result
