# Plan Implementacji Wersji v2.1.0 (Metis & Momus Edition)

## 1. Analiza Metis (Analiza Architektoniczna i Wymagania Przed-wdrożeniowe)

### Cel wdrożenia
Rozszerzenie serwera `deepseek-search-everywhere-mcp` o pełną zgodność ze specyfikacją DeepSeek Search & Fetch API, mechanizmy bezpieczeństwa przeciw Prompt Injection, ergonomiczne stronicowanie dokumentów w `fetch`, jawne operatory wyszukiwania, bezpośrednie metryki kosztów oraz opcjonalny lokalny dzienny limit budżetowy.

### Inwarianty projektu (zgodnie z `AGENTS.md`)
1. **Zamrożone Public API**: Nazwy narzędzi i istniejące kody błędów nie mogą ulec zmianie.
2. **Nowe parametry narzędzi**: Wyłącznie keyword-only (`*, param: type = default`).
3. **SSRF**: Filtry URL pozostają czysto statyczne (brak lokalnego rozwiązywania DNS).
4. **Brak cachowania błędów**: Błędy i limity budżetowe nigdy nie trafiają do `TTLCache`.
5. **Brak zewnętrznej telemetrii**: Licznik budżetu i kosztów jest w 100% lokalny (pamięć lub plik stanu).
6. **Automatyzacja wydań**: Wersje podnosi wyłącznie `release-please` na podstawie Conventional Commits (`feat:` -> minor `2.1.0`).

---

## 2. Krytyka Momusa (Adversarial Review / Blocker-Finding)

Momus przeprowadził analizę ryzyka i wskazał następujące punkty krytyczne do rozwiązania przed rozpoczęciem kodowania:

| ID | Zidentyfikowane Ryzyko | Potencjalny Blocker | Wymagana Mitygacja w Architekturze |
|:---|:-----------------------|:--------------------|:-----------------------------------|
| **M1** | Łączenie `site:` przez `OR` w `filters.py` a budżet znaków | Dodanie ` OR ` dodaje 4 znaki na każdą kolejną domenę. Może przekroczyć `MAX_FILTER_OPS_CHARS = 96`. | Rygorystyczna kontrola długości w `compile_filters` z kodem `filters_too_long`. Uaktualnienie testów weryfikujących format. |
| **M2** | Przepełnienie budżetu `MAX_QUERY = 128` przy `exclude_terms` i `exact_phrases` | Połączenie zapytania naturalnego z wieloma wykluczeniami może przekroczyć 128 znaków. | `server.py` oblicza budżet `MAX_QUERY - ops_len`. Query naturalne jest bezpiecznie docinane z notatką `trimmed`. |
| **M3** | Wyścigi (Race conditions) w pliku budżetu przy wielu workerach | Równoległe procesy HTTP mogą nadpisać plik `budget.json` lub spowodować błąd parsowania JSON. | Zastosowanie `asyncio.Lock` wewnątrz procesu oraz atomowego zapisu pliku (`atomic write` przez plik tymczasowy i `os.replace`). Przy błędzie odczytu: bezpieczny fallback bez crasha. |
| **M4** | Out-of-bounds przy `offset` i `limit` w `deepseek_fetch` | `offset` większy niż długość strony mógłby rzucić `IndexError` lub wygenerować mylący stan. | Bezpieczne cięcie `content[start:start+limit]`, gdzie dla `offset >= len(content)` zwracany jest pusty string `""`, `chars=0`, `has_more=False`. |
| **M5** | Kolizja `find_text` i wielkości liter | Użytkownik szuka "API", w tekście jest "api". | Wyszukiwanie `find_text` musi być domyślnie case-insensitive (`content.lower().find(find_text.lower())`). |

**Werdykt Momusa:** `OKAY WITH MITIGATIONS APPLIED`.

---

## 3. Fazy Implementacji

### Faza 1: Filtry Zapytań i Specyfikacja Search (Bugfix & Compliance)
* **Zadanie 1.1:** Poprawka operatora `OR` dla wielu domen `include_domains` w `src/deepseek_search_everywhere_mcp/filters.py`.
  - Format: `site:domain1 OR site:domain2`.
* **Zadanie 1.2:** Dodanie obsługi `exclude_terms: list[str] | None` (`-term`) i `exact_phrases: list[str] | None` (`"phrase"`) w `compile_filters`.
* **Zadanie 1.3:** Dodanie parametru `localization: str | None = None` do `deepseek_search`, `deepseek_web_search` i przekazanie go do payloadu `_build_search_payload`.
* **Zadanie 1.4:** Testy jednostkowe dla filtrów (`tests/test_filters.py`) i parametru `localization` (`tests/test_search.py`).

### Faza 2: Stronicowanie i Wyszukiwanie w `deepseek_fetch` (Ergonomia i Oszczędność Tokenów)
* **Zadanie 2.1:** Rozszerzenie sygnatur `deepseek_fetch` o keyword-only: `offset: int | None = None`, `limit: int | None = None`, `find_text: str | None = None`.
* **Zadanie 2.2:** Implementacja logiki okna tekstowego w `_fetch`:
  - Walidacja: `offset >= 0`, `limit > 0`.
  - Obsługa `find_text`: wyszukanie indeksu i automatyczne pozycjonowanie `offset`.
  - Zwracanie w kopercie: `offset`, `total_chars`, `has_more`.
* **Zadanie 2.3:** Propagacja opcjonalnych parametrów w `deepseek_fetch_many` (`urls` lub słowniki opcji).
* **Zadanie 2.4:** Testy jednostkowe stronicowania i `find_text` (`tests/test_fetch.py`).

### Faza 3: Bezpieczeństwo i Śledzenie Kosztów (Prompt Injection & Meta Cost)
* **Zadanie 3.1:** Dodanie flagi `untrusted_content: True` w kopertach `deepseek_fetch` i `deepseek_search` (oraz wewnątrz `_meta`).
* **Zadanie 3.2:** Dodanie obiektu `_meta: {"cost_usd": ..., "untrusted_content": True}` oraz pola `cost_usd`:
  - `0.0` przy trafieniu w cache (`cached=True`).
  - `0.00075` przy wywołaniu upstream.
* **Zadanie 3.3:** Uaktualnienie `deepseek_fetch_many` o sumaryczny koszt upstream.
* **Zadanie 3.4:** Testy weryfikujące poprawność metadanych (`tests/test_metadata.py` / aktualizacja `test_search.py`).

### Faza 4: Lokalny Dzienny Limit Budżetowy (Kill-Switch Ochronny)
* **Zadanie 4.1:** Nowy moduł `src/deepseek_search_everywhere_mcp/budget.py`:
  - Klasa `DailyBudgetTracker`.
  - Zmienna `DEEPSEEK_DAILY_BUDGET_USD` (float).
  - Trwały zapis w `$DEEPSEEK_STATE_DIR/budget.json` lub fallback do katalogu cache użytkownika.
  - Resetowanie o 00:00 UTC (klucz po dacie `YYYY-MM-DD`).
* **Zadanie 4.2:** Integracja z `_limited_request` / `server.py`:
  - Sprawdzenie przed zapytaniem upstream.
  - Zwrócenie błędu `_fail("budget_exceeded", ..., status=429)` bez cachowania.
* **Zadanie 4.3:** Testy jednostkowe budżetu (mock daty, przekroczenie limitu, odporność na błędy IO).

### Faza 5: Weryfikacja Jakości, Testy Regresji i Przygotowanie Wydania
* **Zadanie 5.1:** Pełne wykonanie testów regresyjnych (`.venv/bin/pytest -q`).
* **Zadanie 5.2:** Weryfikacja lintera i formatera (`.venv/bin/ruff check . && .venv/bin/ruff format --check .`).
* **Zadanie 5.3:** Sprawdzenie spójności dokumentacji `README.md` oraz `server.json`.
* **Zadanie 5.4:** Commity zgodne z Conventional Commits (`feat(tools): ...`, `fix(filters): ...`).
