# -*- coding: utf-8 -*-
"""Jednorazowy test: pobiera modele i sprawdza GPU oraz tłumaczenie."""
import app  # noqa: F401  (dodaje biblioteki CUDA do PATH)

print("1) Test Whisper na GPU...")
from faster_whisper import WhisperModel
import numpy as np

whisper = WhisperModel("medium", device="cuda", compute_type="int8_float16")
segments, info = whisper.transcribe(np.zeros(16000, dtype=np.float32), language="ru")
list(segments)
print("   Whisper OK (GPU dziala)")

print("2) Test tlumaczenia ru->pl...")
from transformers import MarianMTModel, MarianTokenizer

name = "Helsinki-NLP/opus-mt-tc-big-zle-zlw"
tok = MarianTokenizer.from_pretrained(name)
model = MarianMTModel.from_pretrained(name)
batch = tok([">>pol<< Привет, как дела? Сегодня хорошая погода."], return_tensors="pt")
out = model.generate(**batch, max_new_tokens=64)
print("   Tlumaczenie:", tok.decode(out[0], skip_special_tokens=True))
print("WSZYSTKO OK")
