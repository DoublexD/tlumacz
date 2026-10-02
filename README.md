# Tłumacz na żywo: rosyjski → polskie napisy

Aplikacja słucha dźwięku **tylko z przeglądarki** (Chrome, Edge, Firefox, Opera,
Brave — wykrywana automatycznie), rozpoznaje mowę rosyjską i wyświetla polskie
napisy w okienku na wierzchu ekranu. Inne dźwięki (Discord, muzyka, powiadomienia)
nie przeszkadzają w tłumaczeniu.
Wszystko działa lokalnie na Twojej karcie graficznej — za darmo i bez internetu
(internet potrzebny był tylko raz, do pobrania modeli).

## Jak używać

1. Kliknij dwa razy **START.bat** (albo w terminalu: `venv\Scripts\python.exe app.py`).
2. Poczekaj ok. 20–30 sekund, aż w okienku pojawi się „Słucham dźwięku z: chrome.exe...".
3. Włącz stream/film po rosyjsku w przeglądarce — napisy pojawiają się
   już w trakcie mówienia (po ok. 1 sekundzie) i aktualizują się na bieżąco.
   Napis zakończony „…" to wersja wstępna, która za chwilę się doprecyzuje.

## Obsługa napisów

Napisy wyglądają jak w kinie — sam biały tekst z czarną obwódką, bez tła,
więc nie zasłaniają filmu. Miejsca bez tekstu przepuszczają kliknięcia
do tego, co jest pod spodem.

- **Przeciągnij tekst myszką** — przesuwasz napisy w dowolne miejsce.
- **Podwójny klik na tekście** — zamyka aplikację.

## Opcje (uruchamianie z terminala)

```
venv\Scripts\python.exe app.py --model small        # szybciej, mniej dokładnie
venv\Scripts\python.exe app.py --model large-v3     # maksymalna dokładność, wolniej
venv\Scripts\python.exe app.py --rosyjski           # pokazuj też oryginał rosyjski
venv\Scripts\python.exe app.py --system             # słuchaj całego komputera
venv\Scripts\python.exe app.py --app vlc.exe        # słuchaj innego programu
```

Domyślny model to `large-v3-turbo` — dokładność blisko największego Whispera,
a na RTX 4060 działa szybciej niż `medium`.

## Jak to działa

1. **proc-tap** (WASAPI process loopback) przechwytuje dźwięk tylko z procesu
   przeglądarki — inne aplikacje nie są słyszane.
2. **Silero VAD** (neuronowy detektor mowy) tnie audio na wypowiedzi —
   odróżnia mowę od muzyki, szumu i efektów dźwiękowych w filmach.
3. **faster-whisper** (model `large-v3-turbo`, na GPU) zamienia rosyjską mowę na tekst.
4. **NLLB-200** (`facebook/nllb-200-distilled-1.3B`, na GPU) tłumaczy tekst na polski.
5. Napisy rysowane są w przezroczystym oknie „zawsze na wierzchu" (tkinter).

## Rozwiązywanie problemów

- **„Nie widzę uruchomionej przeglądarki"** — otwórz przeglądarkę; aplikacja
  sama ją znajdzie po kilku sekundach. Obsługiwane: Chrome, Edge, Firefox,
  Opera (i GX), Brave, Vivaldi. Inną aplikację wskażesz przez `--app nazwa.exe`.
- **Napisy mocno opóźnione** — użyj `--model small`.
- **Dziwne/urwane tłumaczenia** — to normalne przy muzyce, hałasie
  lub kilku osobach mówiących naraz.
