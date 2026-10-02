# -*- coding: utf-8 -*-
"""
Tłumacz na żywo: rosyjski (audio z przeglądarki) -> polskie napisy.

Domyślnie przechwytuje dźwięk TYLKO z przeglądarki (Chrome/Edge/Firefox/Opera/
Brave - wykrywana automatycznie), rozpoznaje mowę rosyjską modelem
faster-whisper (GPU), tłumaczy na polski lokalnym modelem Opus-MT
i wyświetla napisy w okienku zawsze-na-wierzchu.

Uruchomienie:  python app.py                     (dźwięk tylko z przeglądarki)
               python app.py --system            (cały dźwięk systemowy)
               python app.py --app vlc.exe       (dźwięk z innego programu)
               python app.py --model small       (szybszy, mniej dokładny)
Sterowanie:    przeciągnij okno myszką, Esc lub podwójny klik = zamknij.
"""

import argparse
import os
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np


def _add_cuda_dll_dirs():
    """Dodaje biblioteki cuBLAS/cuDNN (z pakietów pip nvidia-*) do PATH,
    żeby ctranslate2/faster-whisper mogły użyć karty NVIDIA."""
    try:
        import nvidia
    except ImportError:
        return
    for base in nvidia.__path__:
        for sub in ("cublas", "cudnn"):
            bin_dir = Path(base) / sub / "bin"
            if bin_dir.is_dir():
                os.add_dll_directory(str(bin_dir))
                os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")


_add_cuda_dll_dirs()

TARGET_SR = 16000          # Whisper i Silero VAD wymagają 16 kHz
VAD_THRESHOLD = 0.5        # próg Silero VAD (0-1): powyżej = mowa
SILENCE_CUT_SEC = 0.35     # tyle przerwy w mowie kończy segment wypowiedzi
MIN_SEGMENT_SEC = 0.6      # krótsze segmenty ignorujemy
MAX_SEGMENT_SEC = 5.0      # przymusowe cięcie długiej wypowiedzi
PARTIAL_EVERY_SEC = 0.45   # co tyle wysyłamy wstępny (rosnący) segment do napisów
QUIET_AFTER_SEC = 2.0      # tyle ciągłej przerwy w mowie = sygnał "schowaj napisy"
MIN_READ_SEC = 3.0         # minimalny czas wyświetlania napisu przed schowaniem

BROWSERS = ["chrome.exe", "msedge.exe", "firefox.exe", "opera.exe",
            "opera_gx.exe", "brave.exe", "vivaldi.exe", "librewolf.exe"]


def resample_to_16k(audio: np.ndarray, src_sr: int) -> np.ndarray:
    if src_sr == TARGET_SR:
        return audio.astype(np.float32)
    n_out = int(len(audio) * TARGET_SR / src_sr)
    x_out = np.linspace(0, len(audio) - 1, n_out)
    return np.interp(x_out, np.arange(len(audio)), audio).astype(np.float32)


class SpeechSegmenter:
    """Zbiera bloki audio i tnie je na wypowiedzi. Mowę wykrywa neuronowy
    model Silero VAD - w odróżnieniu od zwykłego progu głośności odróżnia
    mowę od muzyki, szumu i efektów dźwiękowych w filmach."""

    FRAME = 512  # Silero VAD pracuje na ramkach 512 próbek (32 ms przy 16 kHz)

    def __init__(self, sample_rate: int, segment_queue: queue.Queue):
        import torch
        from silero_vad import load_silero_vad

        self.torch = torch
        self.vad = load_silero_vad()
        self.sr = sample_rate
        self.segment_queue = segment_queue

        self.pending = np.empty(0, dtype=np.float32)  # 16 kHz, czeka na pełną ramkę
        self.frames: list[np.ndarray] = []            # ramki od ostatniego cięcia
        self.flags: list[bool] = []                   # czy ramka zawierała mowę
        self.buffered_sec = 0.0
        self.speech_sec = 0.0
        self.silence_sec = 0.0
        self.had_speech = False
        self.last_partial_sec = 0.0
        self.silence_run = 0.0     # ciągła przerwa (nie resetowana przez cięcie segmentów)
        self.quiet_sent = False

    def feed(self, block: np.ndarray):
        """block: mono float32 w częstotliwości self.sr."""
        self.pending = np.concatenate([self.pending, resample_to_16k(block, self.sr)])
        while len(self.pending) >= self.FRAME:
            frame = np.ascontiguousarray(self.pending[:self.FRAME])
            self.pending = self.pending[self.FRAME:]
            with self.torch.no_grad():
                prob = self.vad(self.torch.from_numpy(frame), TARGET_SR).item()
            self._push(frame, prob >= VAD_THRESHOLD)

    def feed_silence(self, seconds: float):
        self.feed(np.zeros(int(seconds * self.sr), dtype=np.float32))

    def _push(self, frame: np.ndarray, is_speech: bool):
        dur = self.FRAME / TARGET_SR
        self.frames.append(frame)
        self.flags.append(is_speech)
        self.buffered_sec += dur

        if is_speech:
            self.silence_sec = 0.0
            self.silence_run = 0.0
            self.quiet_sent = False
            self.had_speech = True
            self.speech_sec += dur
        else:
            self.silence_sec += dur
            self.silence_run += dur
            if self.silence_run >= QUIET_AFTER_SEC and not self.quiet_sent:
                self.quiet_sent = True
                self.segment_queue.put(("quiet",))

        end_of_speech = self.had_speech and self.silence_sec >= SILENCE_CUT_SEC
        too_long = self.had_speech and self.buffered_sec >= MAX_SEGMENT_SEC

        if end_of_speech or too_long:
            if too_long and not end_of_speech:
                # ciągła mowa: tnij w ostatniej mikropauzie (do 2 s wstecz),
                # a nie w pół słowa; resztę zostaw na początek nowego segmentu
                cut = len(self.frames)
                min_keep = int(1.5 / dur)   # co najmniej 1,5 s zostaje w segmencie
                for i in range(len(self.flags) - 1, min_keep, -1):
                    if not self.flags[i]:
                        cut = i + 1
                        break
                    if len(self.flags) - i > int(2.0 / dur):
                        break
            else:
                cut = len(self.frames)

            audio = np.concatenate(self.frames[:cut])
            rest_frames, rest_flags = self.frames[cut:], self.flags[cut:]
            enough_speech = self.speech_sec >= MIN_SEGMENT_SEC

            self.frames, self.flags = rest_frames, rest_flags
            self.buffered_sec = len(rest_frames) * dur
            self.speech_sec = sum(rest_flags) * dur
            self.silence_sec = 0.0
            self.had_speech = any(rest_flags)
            self.last_partial_sec = 0.0
            if enough_speech:
                self.segment_queue.put(("final", audio))
        elif self.had_speech and self.buffered_sec - self.last_partial_sec >= PARTIAL_EVERY_SEC:
            # mowa trwa - wysyłamy wstępną wersję, żeby napisy pojawiły się od razu
            self.last_partial_sec = self.buffered_sec
            self.segment_queue.put(("partial", np.concatenate(self.frames)))
        elif not self.had_speech and self.buffered_sec > 1.0:
            # brak mowy (cisza/muzyka) - trzymamy tylko krótką końcówkę,
            # żeby nie ucinać pierwszego słowa następnej wypowiedzi
            self.frames = self.frames[-10:]
            self.flags = self.flags[-10:]
            self.buffered_sec = len(self.frames) * dur


def find_app_pid(app_names: list[str]) -> tuple[int | None, str | None]:
    """Zwraca PID głównego procesu aplikacji (tego, którego rodzic nazywa się
    inaczej) - przechwytywanie obejmuje wtedy całe drzewo procesów."""
    import psutil

    procs: dict[int, tuple[str, int]] = {}
    for p in psutil.process_iter(["pid", "name", "ppid"]):
        name = (p.info["name"] or "").lower()
        if name in app_names:
            procs[p.info["pid"]] = (name, p.info["ppid"])

    for wanted in app_names:  # kolejność listy = priorytet
        for pid, (name, ppid) in procs.items():
            if name != wanted:
                continue
            parent = procs.get(ppid)
            if parent is None or parent[0] != name:
                return pid, name
    return None, None


class BrowserCapture(threading.Thread):
    """Nagrywa dźwięk tylko z przeglądarki (WASAPI process loopback)."""

    def __init__(self, app_names: list[str], segment_queue: queue.Queue,
                 stop_event: threading.Event):
        super().__init__(daemon=True)
        self.app_names = app_names
        self.segment_queue = segment_queue
        self.stop_event = stop_event

    def run(self):
        try:
            import psutil
            from proctap import ProcessAudioCapture

            while not self.stop_event.is_set():
                pid, name = find_app_pid(self.app_names)
                if pid is None:
                    self.segment_queue.put(
                        ("status", "Nie widzę uruchomionej przeglądarki - otwórz ją, czekam..."))
                    time.sleep(2)
                    continue

                tap = ProcessAudioCapture(pid)
                tap.start()
                fmt = tap.get_format()
                sr = int(fmt.get("sample_rate", 48000))
                channels = int(fmt.get("channels", 2))

                self.segment_queue.put(
                    ("status", f"Słucham dźwięku z: {name} (PID {pid})..."))
                segmenter = SpeechSegmenter(sr, self.segment_queue)

                try:
                    while not self.stop_event.is_set():
                        chunk = tap.read(timeout=0.25)
                        if chunk:
                            block = np.frombuffer(chunk, dtype=np.float32)
                            if channels > 1:
                                block = block.reshape(-1, channels).mean(axis=1)
                            segmenter.feed(block)
                        else:
                            # brak danych = przeglądarka nic nie odtwarza
                            segmenter.feed_silence(0.25)
                            if not psutil.pid_exists(pid):
                                self.segment_queue.put(
                                    ("status", "Przeglądarka została zamknięta, szukam ponownie..."))
                                break
                finally:
                    tap.close()
        except Exception as exc:
            self.segment_queue.put(exc)


class SystemCapture(threading.Thread):
    """Nagrywa cały dźwięk systemowy (WASAPI loopback) - tryb --system."""

    BLOCK_SEC = 0.05

    def __init__(self, segment_queue: queue.Queue, stop_event: threading.Event):
        super().__init__(daemon=True)
        self.segment_queue = segment_queue
        self.stop_event = stop_event

    def run(self):
        import pyaudiowpatch as pyaudio

        pa = pyaudio.PyAudio()
        try:
            wasapi = pa.get_host_api_info_by_type(pyaudio.paWASAPI)
            speakers = pa.get_device_info_by_index(wasapi["defaultOutputDevice"])
            if not speakers.get("isLoopbackDevice"):
                for lb in pa.get_loopback_device_info_generator():
                    if speakers["name"] in lb["name"]:
                        speakers = lb
                        break
                else:
                    raise RuntimeError("Nie znaleziono urządzenia loopback dla głośników.")

            device_sr = int(speakers["defaultSampleRate"])
            channels = max(1, int(speakers["maxInputChannels"]))
            frames_per_block = int(device_sr * self.BLOCK_SEC)

            stream = pa.open(
                format=pyaudio.paInt16,
                channels=channels,
                rate=device_sr,
                input=True,
                input_device_index=speakers["index"],
                frames_per_buffer=frames_per_block,
            )
            self.segment_queue.put(("status", "Słucham całego dźwięku systemowego..."))
            segmenter = SpeechSegmenter(device_sr, self.segment_queue)

            while not self.stop_event.is_set():
                raw = stream.read(frames_per_block, exception_on_overflow=False)
                block = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                if channels > 1:
                    block = block.reshape(-1, channels).mean(axis=1)
                segmenter.feed(block)

            stream.stop_stream()
            stream.close()
        except Exception as exc:
            self.segment_queue.put(exc)
        finally:
            pa.terminate()


class Translator(threading.Thread):
    """Whisper (rosyjski -> tekst) + Opus-MT (rosyjski -> polski)."""

    def __init__(self, model_size: str, segment_queue: queue.Queue,
                 ui_queue: queue.Queue, stop_event: threading.Event):
        super().__init__(daemon=True)
        self.model_size = model_size
        self.segment_queue = segment_queue
        self.ui_queue = ui_queue
        self.stop_event = stop_event

    def run(self):
        try:
            self.ui_queue.put(("status", f"Ładowanie modelu Whisper ({self.model_size})... "
                                         "(za pierwszym razem pobiera się z internetu)"))
            from faster_whisper import WhisperModel
            try:
                whisper = WhisperModel(self.model_size, device="cuda", compute_type="int8_float16")
            except Exception:
                self.ui_queue.put(("status", "GPU niedostępne - używam procesora (wolniej)..."))
                whisper = WhisperModel(self.model_size, device="cpu", compute_type="int8")

            self.ui_queue.put(("status", "Ładowanie modelu tłumaczącego (NLLB, ru->pl)..."))
            import re

            import ctranslate2
            from transformers import AutoTokenizer

            mt_name = "facebook/nllb-200-distilled-1.3B"
            mt_dir = Path(__file__).parent / "models" / "nllb-1.3b-ct2"
            if not mt_dir.is_dir():
                self.ui_queue.put(("status", "Pierwsze uruchomienie: pobieram i konwertuję "
                                             "model tłumaczący (kilka minut)..."))
                from ctranslate2.converters import TransformersConverter
                TransformersConverter(mt_name).convert(str(mt_dir), quantization="int8_float16")

            mt_tokenizer = AutoTokenizer.from_pretrained(mt_name, src_lang="rus_Cyrl")
            try:
                mt = ctranslate2.Translator(str(mt_dir), device="cuda", compute_type="int8_float16")
            except Exception:
                mt = ctranslate2.Translator(str(mt_dir), device="cpu", compute_type="int8")

            def split_long(sentence: str, limit: int = 28) -> list[str]:
                """Tnie bardzo długie "zdania" (ciągła mowa bez interpunkcji)
                na porcje ~28 słów, najchętniej na przecinku - model tłumaczący
                traci jakość na zbyt długich wejściach."""
                words = sentence.split()
                out = []
                while len(words) > limit:
                    cut = limit
                    for i in range(limit, max(8, limit - 14), -1):
                        if words[i - 1].endswith((",", ";", ":", "-")):
                            cut = i
                            break
                    out.append(" ".join(words[:cut]))
                    words = words[cut:]
                if words:
                    out.append(" ".join(words))
                return out

            def translate(text: str, beam_size: int = 1) -> str:
                # Model tłumaczy pojedyncze zdania, więc dzielimy tekst i tłumaczymy paczką.
                sentences = [s.strip() for s in re.split(r"(?<=[.!?…])\s+", text) if s.strip()]
                sentences = [c for s in sentences for c in split_long(s)]
                if not sentences:
                    return ""
                batch = [mt_tokenizer.convert_ids_to_tokens(
                            mt_tokenizer.encode(s, truncation=True, max_length=256))
                         for s in sentences]
                results = mt.translate_batch(batch, beam_size=beam_size,
                                             max_decoding_length=256,
                                             no_repeat_ngram_size=4,
                                             target_prefix=[["pol_Latn"]] * len(batch))
                out = []
                for r in results:
                    tokens = [t for t in r.hypotheses[0] if t != "pol_Latn"]
                    ids = mt_tokenizer.convert_tokens_to_ids(tokens)
                    out.append(mt_tokenizer.decode(ids, skip_special_tokens=True))
                return collapse_repeats(" ".join(out))

            self.ui_queue.put(("status", "Gotowe. Czekam na dźwięk..."))

            def norm(t: str) -> str:
                """Do porównywania transkrypcji: bez wielkości liter i interpunkcji."""
                return re.sub(r"[^\wа-яё]+", "", t.lower())

            # typowe halucynacje Whispera na ciszy/muzyce (frazy z napisów końcowych,
            # na których model był uczony) - wycinamy je z transkrypcji
            HALLUCINATIONS = (
                "продолжениеследует", "субтитр", "спасибозапросмотр",
                "подписывайтесьнаканал", "ставьтелайк", "доновыхвстреч",
                "редактор", "переводчик", "корректор",
            )
            # frazy wycinane tylko, gdy stanowią CAŁY fragment (samotne "Спасибо."
            # na ciszy to halucynacja; w środku zdania to normalne słowo)
            EXACT_HALLUCINATIONS = {
                "спасибо", "благодарю", "всемпока", "довстречи", "пока",
                "конец", "конецфильма", "аплодисменты", "музыка", "тишина",
            }

            def clean_segments(segments) -> str:
                parts = []
                for s in segments:
                    # heurystyka Whispera: wysokie no_speech + niska pewność = cisza
                    if s.no_speech_prob > 0.6 and s.avg_logprob < -1.0:
                        continue
                    t = s.text.strip()
                    n = norm(t)
                    if any(h in n for h in HALLUCINATIONS):
                        continue
                    if n in EXACT_HALLUCINATIONS:
                        continue
                    # fragment będący samym powtarzanym "не"/"нет" = halucynacja
                    if re.fullmatch(r"(?:(?:не|нет)[\s,.!?…-]*){2,}", t.lower()):
                        continue
                    parts.append(t)
                return " ".join(parts).strip()

            def collapse_repeats(t: str) -> str:
                """Skraca zapętlenia ("nie nie nie nie...") do jednego wystąpienia."""
                # pojedyncze słowo powtórzone 3+ razy z rzędu
                t = re.sub(r"(?iu)\b([\w'-]+)(?:[\s,.!?…-]+\1\b){2,}", r"\1", t)
                # krótka fraza (2-4 słowa) powtórzona 3+ razy z rzędu
                t = re.sub(r"(?iu)(\b[\w'\s,-]{3,40}?)(?:[\s,.!?…-]+\1){2,}", r"\1", t)
                return t

            last_partial: tuple[str, str] | None = None  # (norm(ru), pl)
            prev_partial_words: list[str] = []  # słowa poprzedniej wersji wstępnej
            shown_word_count = 0                # ile "zatwierdzonych" słów już pokazano
            stall_count = 0                     # ile cykli z rzędu bez postępu

            while not self.stop_event.is_set():
                try:
                    items = [self.segment_queue.get(timeout=0.3)]
                except queue.Empty:
                    continue
                # zgarniamy wszystko, co czeka w kolejce, żeby nie narastało opóźnienie
                while True:
                    try:
                        items.append(self.segment_queue.get_nowait())
                    except queue.Empty:
                        break

                finals: list[np.ndarray] = []
                partial: np.ndarray | None = None
                for item in items:
                    if isinstance(item, Exception):
                        self.ui_queue.put(("status", f"Błąd audio: {item}"))
                    elif item[0] in ("status", "quiet"):
                        self.ui_queue.put(item)
                    elif item[0] == "final":
                        finals.append(item[1])
                        partial = None  # final zastępuje wcześniejsze wersje wstępne
                    else:  # "partial" - liczy się tylko najnowszy
                        partial = item[1]

                if finals:
                    audio, is_partial = np.concatenate(finals), False
                elif partial is not None:
                    audio, is_partial = partial, True
                else:
                    continue

                # wersje wstępne: szybko; ostateczne: maksymalna jakość
                segments, _ = whisper.transcribe(
                    audio, language="ru", beam_size=1 if is_partial else 4,
                    vad_filter=True, condition_on_previous_text=False,
                )
                ru_text = collapse_repeats(clean_segments(segments))
                if not ru_text or len(ru_text) < 2:
                    continue

                if is_partial:
                    # "local agreement": pokazujemy tylko słowa, które dwie kolejne
                    # analizy rozpoznały tak samo - napis rośnie, nie przepisuje się
                    cur_words = ru_text.split()
                    if not prev_partial_words:
                        agreed = cur_words  # pierwsza wersja - pokaż od razu
                    else:
                        agreed = []
                        for a, b in zip(prev_partial_words, cur_words):
                            if norm(a) == norm(b):
                                agreed.append(b)
                            else:
                                break
                    prev_partial_words = cur_words

                    if len(agreed) > shown_word_count:
                        stall_count = 0
                        shown_word_count = len(agreed)
                        ru_shown = " ".join(agreed)
                    else:
                        # brak postępu; po 2 cyklach pokazujemy bieżącą wersję,
                        # żeby napisy nie stały w miejscu
                        stall_count += 1
                        if stall_count < 2 or len(cur_words) <= shown_word_count:
                            continue
                        stall_count = 0
                        ru_shown = ru_text

                    pl_text = translate(ru_shown, beam_size=1)
                    last_partial = (norm(ru_shown), pl_text)
                    self.ui_queue.put(("subtitle", ru_shown, pl_text + " …"))
                else:
                    # jeśli ostateczna transkrypcja brzmi tak samo jak już pokazana
                    # wersja wstępna, nie zmieniamy napisu (żeby się nie "poprawiał")
                    if last_partial and last_partial[0] == norm(ru_text):
                        pl_text = last_partial[1]
                    else:
                        pl_text = translate(ru_text, beam_size=4)
                    last_partial = None
                    prev_partial_words = []
                    shown_word_count = 0
                    stall_count = 0
                    self.ui_queue.put(("subtitle", ru_text, pl_text))
        except Exception as exc:
            self.ui_queue.put(("status", f"Błąd: {exc}"))


class SubtitleWindow:
    """Napisy jak w kinie: sam tekst z czarną obwódką, bez tła.
    Przezroczyste miejsca przepuszczają kliknięcia do filmu pod spodem;
    złapać i przeciągnąć można sam tekst."""

    TRANSPARENT = "#010101"   # kolor traktowany jako "szyba" (nie czysta czerń,
                              # bo czerni używamy do obwódki liter)

    def __init__(self, ui_queue: queue.Queue, stop_event: threading.Event, show_ru: bool):
        import tkinter as tk
        import tkinter.font as tkfont

        self.ui_queue = ui_queue
        self.stop_event = stop_event
        self.show_ru = show_ru

        self.root = tk.Tk()
        self.root.title("Napisy PL")
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)
        self.root.attributes("-transparentcolor", self.TRANSPARENT)
        self.root.configure(bg=self.TRANSPARENT)

        sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        self.width = int(sw * 0.75)
        self.x = (sw - self.width) // 2
        self.bottom = sh - 60          # dolna krawędź napisów (stała przy zmianie wysokości)
        self.root.geometry(f"{self.width}x60+{self.x}+{self.bottom - 60}")

        self.canvas = tk.Canvas(self.root, bg=self.TRANSPARENT, highlightthickness=0, bd=0)
        self.canvas.pack(fill="both", expand=True)

        self.pl_font = tkfont.Font(family="Segoe UI", size=21, weight="bold")
        self.ru_font = tkfont.Font(family="Segoe UI", size=12)

        self.canvas.bind("<Button-1>", self._drag_start)
        self.canvas.bind("<B1-Motion>", self._drag_move)
        self.canvas.bind("<Double-Button-1>", lambda e: self.close())
        self.root.bind("<Escape>", lambda e: self.close())
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        self._last_subtitle_time = 0.0
        self._hide_at: float | None = None
        self._render("", "Uruchamianie...", color="#e0c060")
        self._poll()

    # --- rysowanie ------------------------------------------------------

    def _outlined_text(self, x, y, text, font, fill, anchor="n", width=None):
        """Tekst z czarną obwódką (rysowany 8x na czarno + 1x w kolorze)."""
        kw = dict(text=text, font=font, anchor=anchor)
        if width:
            kw["width"] = width
        for dx, dy in ((-2, 0), (2, 0), (0, -2), (0, 2),
                       (-1, -1), (1, -1), (-1, 1), (1, 1)):
            self.canvas.create_text(x + dx, y + dy, fill="black", justify="center", **kw)
        return self.canvas.create_text(x, y, fill=fill, justify="center", **kw)

    def _render(self, ru: str, pl: str, color: str = "white"):
        self.canvas.delete("all")
        if not pl and not ru:
            self.root.geometry(f"{self.width}x1+{self.x}+{self.bottom - 1}")
            return

        cx = self.width // 2
        y = 4
        if ru and self.show_ru:
            item = self._outlined_text(cx, y, ru, self.ru_font, "#c8c8c8",
                                       width=self.width - 40)
            y = self.canvas.bbox(item)[3] + 4
        if pl:
            self._outlined_text(cx, y, pl, self.pl_font, color, width=self.width - 40)

        bbox = self.canvas.bbox("all")
        h = (bbox[3] if bbox else 20) + 6
        self.root.geometry(f"{self.width}x{h}+{self.x}+{self.bottom - h}")

    # --- przeciąganie ---------------------------------------------------

    def _drag_start(self, event):
        self._dx, self._dy = event.x_root - self.root.winfo_x(), event.y_root - self.root.winfo_y()

    def _drag_move(self, event):
        self.x = event.x_root - self._dx
        y = event.y_root - self._dy
        self.bottom = y + self.root.winfo_height()
        self.root.geometry(f"+{self.x}+{y}")

    # --- pętla UI ---------------------------------------------------------

    def _poll(self):
        now = time.time()
        try:
            while True:
                msg = self.ui_queue.get_nowait()
                if msg[0] == "status":
                    self._render("", msg[1], color="#e0c060")
                    self._hide_at = None
                elif msg[0] == "quiet":
                    # cisza - schowaj napis, ale dopiero gdy dało się go przeczytać
                    if self._last_subtitle_time:
                        self._hide_at = max(now, self._last_subtitle_time + MIN_READ_SEC)
                else:
                    _, ru, pl = msg
                    self._render(ru, pl)
                    self._last_subtitle_time = now
                    self._hide_at = now + 10.0  # awaryjnie, gdyby sygnał ciszy nie doszedł
        except queue.Empty:
            pass
        if self._hide_at and now >= self._hide_at:
            self._hide_at = None
            self._last_subtitle_time = 0.0
            self._render("", "")
        self.root.after(100, self._poll)

    def close(self):
        self.stop_event.set()
        self.root.destroy()

    def run(self):
        self.root.mainloop()


def acquire_single_instance_lock():
    """Pozwala działać tylko jednej kopii aplikacji naraz (blokada na porcie)."""
    import socket

    lock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        lock.bind(("127.0.0.1", 54217))
        return lock  # trzymamy otwarty socket przez cały czas życia aplikacji
    except OSError:
        return None


def main():
    parser = argparse.ArgumentParser(description="Tłumacz na żywo RU -> PL (napisy)")
    parser.add_argument("--model", default="large-v3-turbo",
                        help="model Whisper: tiny/small/medium/large-v3-turbo/large-v3 "
                             "(domyślnie large-v3-turbo)")
    parser.add_argument("--system", action="store_true",
                        help="słuchaj całego dźwięku systemowego zamiast samej przeglądarki")
    parser.add_argument("--app", default=None, metavar="NAZWA.EXE",
                        help="słuchaj konkretnego programu, np. --app vlc.exe")
    parser.add_argument("--rosyjski", action="store_true",
                        help="pokazuj też oryginalny tekst rosyjski nad polskim")
    args = parser.parse_args()

    lock = acquire_single_instance_lock()
    if lock is None:
        # druga kopia już działa - nie uruchamiamy kolejnych napisów
        import tkinter.messagebox as mb
        mb.showinfo("Napisy PL", "Tłumacz już działa.\n\n"
                    "Napisy są na ekranie (podwójny klik na tekście je zamyka).")
        sys.exit(0)

    stop_event = threading.Event()
    segment_queue: queue.Queue = queue.Queue()
    ui_queue: queue.Queue = queue.Queue()

    if args.system:
        SystemCapture(segment_queue, stop_event).start()
    else:
        app_names = [args.app.lower()] if args.app else BROWSERS
        BrowserCapture(app_names, segment_queue, stop_event).start()

    Translator(args.model, segment_queue, ui_queue, stop_event).start()

    window = SubtitleWindow(ui_queue, stop_event, show_ru=args.rosyjski)
    window.run()


if __name__ == "__main__":
    main()
