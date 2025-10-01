# Yeti2 — AI Code Generator (yeti.py)

Poniższy dokument szczegółowo opisuje, co robi aplikacja zawarta w pliku `yeti.py`, do czego może być wykorzystana oraz jakie ulepszenia warto wprowadzić, zachowując obecną logikę działania.


**Co to jest**
- **Cel:** narzędzie do budowy i trenowania modelu generującego tekst (m.in. kod w Pythonie) na bazie własnych plików tekstowych i zewnętrznych zbiorów (Hugging Face). Program uruchamia prosty interfejs CLI.
- **Architektura:** hybrydowy model sekwencyjny łączący w jednym pipeline: embeddingi GloVe, konwolucje 1D, warstwy BiGRU i BiLSTM, oraz własny blok transformera (wielogłowa atencja + pozycjonowanie + FFN).
- **Tokenizacja:** wykorzystuje tokenizer GPT‑2 (Hugging Face) do zamiany tekstu na identyfikatory tokenów.
- **Embeddingi:** macierz embeddingów budowana z GloVe (100d); dla tokenów bez wektora stosowana jest średnia embeddingów (fallback).
- **Trening i logowanie:** dane pakowane do `tf.data.Dataset` i trenowane z callbackami TensorBoard (histogramy, metadane) oraz EarlyStopping i Checkpoint.


**Główne komponenty**
- **Wejście danych:**
  - Katalog `test/` z plikami `.txt` (lokalne dane treningowe).
  - Zewnętrzny dataset „maximedb/natural_questions” (Hugging Face), pobierany i zapisywany lokalnie do `./data/natural_questions`.
- **Wstępne przetwarzanie:**
  - `preprocess_text`: tokenizacja NLTK (słowa) z rozpoznaniem wejść jako string/bytes/tensor/ndarray.
  - Dla plików: każda linia tekstu jest tokenizowana i rozbijana na słowa.
- **Tokenizacja i słownik:**
  - `AutoTokenizer` (GPT‑2) z dodanym tokenem PAD.
  - Rozmiar słownika liczony na podstawie mapy `get_vocab()`.
- **Embeddingi:**
  - `create_embedding_matrix`: macierz `(vocab_size, 100)` tworzona z GloVe; brakujące tokeny otrzymują średni wektor wszystkich embeddingów GloVe.
- **Sekwencje i dane:**
  - `generate_sequences`: koduje tekst z GPT‑2 i dopasowuje/podcina do `input_sequence_length`.
  - `create_X_y`: z sekwencji tworzy pary (X, y) dla przewidywania kolejnego tokenu.
  - `split_data`: dzieli na train/val (domyślnie 80/20).
  - `create_datasets`: buduje `tf.data.Dataset` z buforowaniem, batchingiem i prefetch.
- **Model:**
  - Embedding (z wagami GloVe), następnie: `Conv1D → MaxPool1D → Conv1D → BiGRU → Dropout → LayerNorm → BatchNorm → BiGRU → Dropout → LayerNorm → BatchNorm → BiLSTM → Dropout → LayerNorm → BatchNorm → ACompleteTransformerBlock → GlobalAveragePooling1D → Dense → Dense → Dense(softmax)`.
  - Blok transformera: pozycjonowanie sinus/cosinus, `MultiHeadAttention`, FFN, LayerNorm, Dropout.
  - Optymalizator Adam, stratę `CategoricalCrossentropy` i metrykę accuracy.
- **Callbacki:**
  - `TensorBoard` z histogramami wag bloku transformera i metadanymi słownika.
  - `EarlyStopping` i `ModelCheckpoint` (zapisy `.ckpt` i `.h5` w `logs/<model_name>/`).
  - `TextGenerationCallback`: okresowo generuje tekst w trakcie treningu.
- **Zapisy/Wczytywanie:**
  - Model i tokenizer zapisywane w katalogu `mozgi/` (`.h5` oraz `_tokenizer.pkl`).
- **CLI:**
  - Opcje: 1) Trenowanie nowego modelu, 2) Wczytanie istniejącego, 3) Generowanie kodu, 4) Ewaluacja, 5) Wyjście.


**Do czego może być wykorzystany**
- **Generowanie tekstu i kodu:** tworzenie szkiców odpowiedzi, krótkich fragmentów kodu Python lub tekstu technicznego na bazie dostarczonych danych.
- **Prototypowanie modeli NLP:** szybkie eksperymenty z architekturą łączącą RNN/Conv/Transformer nad własnym korpusem.
- **Wzbogacanie korpusu:** łączenie danych lokalnych z publicznymi zbiorami (np. QA) w jednym procesie.
- **Edukacja:** demonstracja pipeline’u: tokenizacja HF, embeddingi GloVe, `tf.data`, callbacki TensorBoard i wczesne zatrzymanie.


**Wymagania i uruchomienie**
- **Środowisko:** Python 3.9+ z zainstalowanymi m.in.: `tensorflow`, `transformers`, `datasets`, `huggingface_hub`, `nltk`, `gensim`, `scikit-learn`, `psutil`, `tqdm`.
- **GPU (opcjonalnie):** kod aktywnie wykrywa GPU i włącza „memory growth”.
- **Dane lokalne:** umieść pliki `.txt` w katalogu `test/`.
- **Dostęp do sieci:** wymagany do pobrania GloVe i datasetu z Hugging Face.
- **Uruchomienie:** `python yeti.py` i wybór opcji w menu CLI.

Uwaga: bieżąca wersja zawiera błędy konsystencji (patrz „Znane problemy”). Aby realnie uruchomić trening/generowanie, konieczne są poprawki opisane niżej (sekcja „Ulepszenia”).


**Znane problemy (stan pliku yeti.py)**
- **Mieszanie tokenizera i embeddingów:** GPT‑2 tokenizer (BPE) vs. embeddingi GloVe (słowa). Mapowanie token→embedding przez słowa jest częściowo przypadkowe; brak spójności semantycznej dla niektórych tokenów.
- **Rozjechane interfejsy funkcji:**
  - `create_sequences` zwraca 7 elementów, a `create_and_train_model` rozpakowuje 8 (z różnymi nazwami), co przerwie wykonanie.
  - `prepare_data` używa `batch_size`, którego nie przyjmuje jako argumentu ani nie widzi w zasięgu.
- **Wywołania funkcji i zasięgi:**
  - W `main()` wywołania typu `get_available_models()`, `load_model_and_tokenizer()`, `generate_code_ai()` i `evaluate_model1()` nie odnoszą się do obiektu `TextProcessor` i/lub nie istnieją (`evaluate_model1`).
  - `get_available_models` i `generate_text` są zdefiniowane bez `self`, ale wewnątrz klasy — to spowoduje błędy (brak dekoratora `@staticmethod`).
  - `load_model_and_tokenizer` korzysta z nieistniejącego `self.load_tokenizer`.
- **Warstwa wyjściowa i targety:**
  - Ostatnia warstwa ma rozmiar `input_sequence_length` z aktywacją `softmax`, podczas gdy typowe przewidywanie kolejnego tokenu wymaga rozmiaru `vocab_size` oraz targetów one‑hot lub indeksów tokenów z odpowiednią stratą.
- **Foldery i ścieżki:**
  - Zapisy do `mozgi/` i `logs/` nie tworzą katalogów, jeśli nie istnieją.
- **Bezpieczeństwo:** hard‑codowane `use_auth_token` (Hugging Face) w kodzie grozi wyciekiem sekretu.
- **Sieć:** pobieranie GloVe i datasetów może się nie udać w środowiskach bez dostępu do Internetu.


**Ulepszenia (zachowując logikę)**
- **Spójny interfejs funkcji:**
  - Ujednolicić wartości zwracane przez `create_sequences` i podpis `create_and_train_model`. Przekazywać i zwracać: `(train_dataset, val_dataset, embedding_matrix, vocab_size, embedding_dim, batch_size, dodatkowe_metadane)`; zmienną `embedding_dim` brać z `embedding_matrix.shape[1]`.
  - Dodać argument `batch_size` do `prepare_data` albo zrezygnować z jej wywołania na rzecz bezpośredniego budowania datasetów tam, gdzie znany jest `batch_size`.
- **Korekta zakresów i metod:**
  - Oznaczyć metody narzędziowe (`get_available_models`, `generate_text` wariant bez `self`) jako `@staticmethod` lub przenieść je poza klasę. W `main()` wywoływać metody na instancji `processor`.
  - Zaimplementować `load_tokenizer` (symetrycznie do `save_tokenizer`) lub używać `AutoTokenizer.from_pretrained(...)` i zapisu przez `tokenizer.save_pretrained(...)`/`from_pretrained(...)`.
- **Wyjście modelu i targety:**
  - Zachowując obecną architekturę, zmienić ostatnią warstwę Dense na rozmiar `vocab_size` i dopasować targety (np. `sparse_categorical_crossentropy` dla indeksów tokenów) — logika „przewiduj następny token” pozostaje ta sama, jedynie wymiar jest poprawny.
- **Tokeny a embeddingi:**
  - Pozostać przy pomyśle GPT‑2 tokenizer + GloVe, ale jawnie obsłużyć mapowanie: dla tokenów BPE, które nie mają reprezentacji słownej w GloVe, używać obecnego fallbacku (średni embedding) lub uproszczonej normalizacji (np. stripowanie `Ġ`/`Ċ`) przed lookupiem.
- **Foldery i zapisy:**
  - Przed zapisami `os.makedirs('mozgi', exist_ok=True)` oraz `os.makedirs('logs/<model_name>', exist_ok=True)`.
- **Konfiguracja i bezpieczeństwo:**
  - Usunąć hard‑codowane tokeny; używać `HUGGINGFACE_TOKEN` z env i przekazywać go tylko, gdy jest ustawiony.
- **Obsługa braku sieci:**
  - Guardy wokół pobierania GloVe i datasets; możliwość pracy tylko na danych lokalnych.
- **Walidacja i ewaluacja:**
  - Dodać metryki dopasowane do zadań generatywnych (np. perplexity); aktualnie `accuracy` na tokenach bywa myląca.
- **Stabilność callbacku generatora:**
  - `TextGenerationCallback` używa atrybutów `Tokenizer` niezgodnych z `AutoTokenizer` (np. `word_index`). Zmienić na `convert_ids_to_tokens`/`decode` zgodnie z HF albo w callbacku używać prostych próbek z datasetu i `tokenizer.decode`.
- **Drobne porządki:**
  - Usunąć nieużywane importy, nazwy w PL/EN ujednolicić, dodać `requirements.txt`, opisać zmienne konfiguracyjne i parametry modelu.


**Szybki scenariusz użycia (po poprawkach)**
- Przygotuj dane tekstowe w `test/*.txt`.
- Ustaw opcjonalnie `HUGGINGFACE_TOKEN` (jeśli prywatne modele/zbiory) i zapewnij dostęp do sieci (dla GloVe i datasetu QA), albo wyłącz zewnętrzne pobieranie.
- Uruchom: `python yeti.py`, wybierz „1. Utwórz nowy model”, podaj nazwę modelu, poczekaj na trening. Checkpointy i logi trafią do `logs/<model_name>/`, model i tokenizer do `mozgi/`.
- Po treningu użyj „2/3” do wczytania i generacji.


**Struktura repo (istotne elementy)**
- `yeti.py` — główny plik aplikacji (CLI + pipeline NLP + model).
- `test/` — przykładowe dane wejściowe (`extended_training_data.txt`).
- `logs/` — katalog logów TensorBoard (tworzony podczas działania; per model).
- `mozgi/` — katalog na modele/tokenizery (wymaga utworzenia przed zapisem).


**Zastrzeżenia**
- Kod w obecnej postaci ma niespójności, które wymagają wdrożenia poprawek z sekcji „Ulepszenia”, aby móc przeprowadzić pełny trening i generację bez błędów runtime.
- Pobieranie zewnętrznych zasobów wymaga dostępu do Internetu i poszanowania odpowiednich licencji.

