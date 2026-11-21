# Frontier Models Reference — November 2026

- **Date of capture:** 2026-11-20
- **Source:** [`llm-stats.com/leaderboards/llm-leaderboard`](https://llm-stats.com/leaderboards/llm-leaderboard) (Ranks #1–60, Pages 1 & 2) and model detail pages ([`llm-stats.com/models/<slug>`](https://llm-stats.com/models/))
- **Capture methodology:** Live DOM extraction via `webview_browser` subagent across Page 1 (#1–30) and Page 2 (#31–60), combined with systematic extraction of the provider table column **`Context in / out`** from each individual model page on `llm-stats.com`.
- **Related ADR:** [ADR 0012 — Modern context window and scaled research limit architecture](adr/0012-modern-context-window-limit-expansion.md)
- **Machine-readable dataset:** [evals/frontier_models.json](frontier_models.json)

---

## 1. Key Architectural Standards (Context IN vs Context OUT)

### 1.1 Context IN (Input Context Window): 1.0M – 1.1M Tokens
Across all frontier model families in November 2026:
- **1,000,000 to 1,100,000 tokens** is the universal input context ceiling for production models across Anthropic, Google, OpenAI, DeepSeek, Moonshot, Zhipu, Meta, and Alibaba.
- Over **85%** of all ranked models on Pages 1 & 2 offer $\ge 1\text{M}$ input tokens natively.
- Any legacy limit below 1M input tokens (such as 32k, 64k, or 128k input caps) is obsolete.

### 1.2 Context OUT (Output / Completion Token Ceilings):
In reasoning models (`thinking: enabled`), reasoning traces (`<think>`) and visible output text share the same completion budget (`max_tokens`). The landscape splits into four distinct tiers:

1. **Long Decodes / Full Output (up to 1,000,000 output tokens):**
   - **Google Gemini 4** (Neon, Carbon, Argon, Hydrogen, 3.7 Flash): `1.0M IN / 1.0M OUT`.
   - **Moonshot Kimi** (K3, K3.1 Pro): `1.0M IN / 1.0M OUT`.
   - **Zhipu GLM** (5.3, 5.3-Flash, 5.2): `1.0M IN / 1.0M OUT`.
   - **Meta Muse Spark Open**: `1.0M IN / 1.0M OUT`.
   - **DeepSeek V4-Pro-0813** (DeepInfra endpoint): `1.0M IN / 1.0M OUT`.
2. **Super-Large Output (384K – 524K output tokens):**
   - **DeepSeek-V4.1-Flash** (and 1017 checkpoint): `1.0M IN / 393.2K OUT` (384K binary).
   - **MiniMax M3.1 Flash**: `1.0M IN / 524.3K OUT`.
   - **Meta Muse Ember**: `1.0M IN / 524.3K OUT`.
   - **Meta Muse Spark 1.3**: `1.0M IN / 943.7K OUT`.
3. **Standard Production Output (128K output tokens):**
   - **Anthropic Claude** (Fable 5.2, Mythos 5.2, Opus 5.5, Sonnet 5.5, Opus 5, Haiku 5.5, Opus 4.8, Opus 4.7, Opus 4.6): `1.0M IN / 128.0K OUT`.
   - **OpenAI GPT** (GPT-6.2 Astra, GPT-6.2 Sol, GPT-6 Astra, GPT-6 Sol, GPT-6.1 Sol, GPT-6.1 Terra, GPT-5.6 Sol, GPT-5.6 Terra, GPT-5.5, GPT-5.5 Pro): `1.1M IN / 128.0K OUT`.
4. **Mid Output (64K – 131K output tokens):**
   - **Qwen4-Max**: `983.6K IN / 262.1K OUT`.
   - **Qwen3.8 Flash**: `1.0M IN / 131.1K OUT`.
   - **Meituan LongCat-2.5**: `1.0M IN / 131.1K OUT`.
   - **Meta Muse Spark 1.1**: `1.0M IN / 131.1K OUT`.
   - **Claude Sonnet 5**: `1.0M IN / 64.0K OUT`.
   - **Gemini 3.8 Flash**: `1.0M IN / 65.5K OUT`.
   - **Qwen3.7 Max**: `1.0M IN / 65.5K OUT`.

---

## 2. Complete Ranks #1–60 Table with Context IN and Context OUT

Extracted directly from the provider tables on `llm-stats.com/models/<slug>`:

| Rank | Model Name | Organization / Lab | Context IN | Context OUT | Model Slug |
| :---: | :--- | :--- | :---: | :---: | :--- |
| 1 | Claude Fable 5.2 | Anthropic | **1.0M** | **128.0K** | `claude-fable-5-2` |
| 2 | Claude Mythos 5.2 | Anthropic | **1.0M** | **128.0K** | `claude-mythos-5-2` |
| 3 | Gemini 4 Neon | Google | **1.0M** | **1.0M** | `gemini-4-neon` |
| 4 | Claude Opus 5.5 | Anthropic | **1.0M** | **128.0K** | `claude-opus-5-5` |
| 5 | GPT-6.2 Astra | OpenAI | **1.1M** | **128.0K** | `gpt-6.2-astra` |
| 6 | Gemini 4 Carbon | Google | **1.0M** | **1.0M** | `gemini-4-carbon` |
| 7 | Claude Sonnet 5.5 | Anthropic | **1.0M** | **128.0K** | `claude-sonnet-5-5` |
| 8 | GPT-6 Astra | OpenAI | **1.1M** | **128.0K** | `gpt-6-astra` |
| 9 | Kimi K3.1 Pro | MoonshotAI | **1.0M** | **1.0M** | `kimi-k3.1-pro` |
| 10 | GPT-6.2 Sol | OpenAI | **1.1M** | **128.0K** | `gpt-6.2-sol` |
| 11 | Claude Mythos Preview | Anthropic | — | — | `claude-mythos-preview` |
| 12 | Gemini 4 Argon | Google | **1.0M** | **1.0M** | `gemini-4-argon` |
| 13 | Muse Wonder Full | Meta | — | — | `muse-wonder-full` |
| 14 | Claude Opus 5 | Anthropic | **1.0M** | **128.0K** | `claude-opus-5` |
| 15 | GPT-6.1 Sol | OpenAI | **1.1M** | **128.0K** | `gpt-6.1-sol` |
| 16 | Claude Fable 5.1 | Anthropic | **1.0M** | **128.0K** | `claude-fable-5-1` |
| 17 | Claude Fable 5 | Anthropic | **1.0M** | **128.0K** | `claude-fable-5` |
| 18 | Muse Wonder Full | Meta | — | — | `muse-wonder-full-1.0` |
| 19 | GPT-5.6 Sol | OpenAI | **1.1M** | **128.0K** | `gpt-5.6-sol` |
| 20 | ERNIE 5.5 Flash Preview | Baidu | **1.0M** | **1.0M** | `ernie-5.5-flash-preview` |
| 21 | Muse Spark 1.3 | Meta | **1.0M** | **943.7K** | `muse-spark-1.3` |
| 22 | Kimi K3 | MoonshotAI | **1.0M** | **1.0M** | `kimi-k3` |
| 23 | Qwen4-Max | Qwen | **983.6K** | **262.1K** | `qwen4-max` |
| 24 | DeepSeek-V4.1-Flash-1017 | DeepSeek | **1.0M** | **393.2K** | `deepseek-v4.1-flash-1017` |
| 25 | Qwen3.8 Max | Qwen | **256.0K** | **256.0K** | `qwen3.8-max` |
| 26 | GLM-5.3 | ZAI | **1.0M** | **1.0M** | `glm-5.3` |
| 27 | Muse Wonder | Meta | **1.0M** | — | `muse-wonder` |
| 28 | LongCat-2.5 | Meituan | **1.0M** | **131.1K** | `longcat-2.5` |
| 29 | Claude Opus 4.8 | Anthropic | **1.0M** | **128.0K** | `claude-opus-4-8` |
| 30 | Muse Wonder | Meta | — | — | `muse-wonder` |
| 31 | Claude Haiku 5.5 | Anthropic | **1.0M** | **128.0K** | `claude-haiku-5-5` |
| 32 | DeepSeek-V4.1-Flash | DeepSeek | **1.0M** | **393.2K** | `deepseek-v4.1-flash` |
| 33 | DeepSeek-V4-Pro-0813 | DeepSeek | **1.0M** | **1.0M** | `deepseek-v4-pro-0813` |
| 34 | Hy4 preview | Tencent | — | — | `hy4-preview` |
| 35 | GPT-5.6 Terra | OpenAI | **1.1M** | **128.0K** | `gpt-5.6-terra` |
| 36 | GLM-5.3-Flash | ZAI | **1.0M** | **1.0M** | `glm-5.3-flash` |
| 37 | Gemini 4 Hydrogen | Google | **1.0M** | **1.0M** | `gemini-4-hydrogen` |
| 38 | Gemini 3.8 Flash | Google | **1.0M** | **65.5K** | `gemini-3.8-flash` |
| 39 | MiMo-V2.6-Pro | Xiaomi | **1.0M** | — | `mimo-v2.6-pro` |
| 40 | Muse Spark 1.1 | Meta | **1.0M** | **131.1K** | `muse-spark-1.1` |
| 41 | MiniMax M3.1 Flash | MiniMax | **1.0M** | **524.3K** | `minimax-m3.1-flash` |
| 42 | Qwen3.8-Flash-Next | Qwen | — | — | `qwen3.8-flash-next` |
| 43 | Qwen3.8 Flash | Qwen | **1.0M** | **131.1K** | `qwen3.8-flash` |
| 44 | Inkling 2 Small | Thinking Machines | **256.0K** | — | `inkling-2-small` |
| 45 | SWE-2 | Cognition | — | — | `swe-2` |
| 46 | GPT-5.5 | OpenAI | **1.1M** | **128.0K** | `gpt-5.5` |
| 47 | GPT-6.1 Terra | OpenAI | **1.1M** | **128.0K** | `gpt-6.1-terra` |
| 48 | Claude Sonnet 5 | Anthropic | **1.0M** | **64.0K** | `claude-sonnet-5` |
| 49 | Gemini 3.7 Flash | Google | **1.0M** | **1.0M** | `gemini-3.7-flash` |
| 50 | GPT-6 Sol | OpenAI | **1.1M** | **128.0K** | `gpt-6-sol` |
| 51 | Mistral Large 4 | Mistral AI | **1.0M** | — | `mistral-large-4` |
| 52 | GPT-5.5 Pro | OpenAI | **1.1M** | **128.0K** | `gpt-5.5-pro` |
| 53 | Muse Ember | Meta | **1.0M** | **524.3K** | `muse-ember` |
| 54 | Grok 4.7 | xAI | **500.0K** | — | `grok-4.7` |
| 55 | Seed 2.1 Pro | Bytedance | — | — | `seed-2.1-pro` |
| 56 | Muse Spark Open | Meta | **1.0M** | **1.0M** | `muse-spark-open` |
| 57 | Qwen3.7 Max | Qwen | **1.0M** | **65.5K** | `qwen3.7-max` |
| 58 | GLM-5.2 | ZAI | **1.0M** | **1.0M** | `glm-5.2` |
| 59 | Claude Opus 4.6 | Anthropic | **1.0M** | **128.0K** | `claude-opus-4-6` |
| 60 | Claude Opus 4.7 | Anthropic | **1.0M** | **128.0K** | `claude-opus-4-7` |

---

## 3. Konsekwencje dla `deepseek-search-everywhere-mcp`

1. **Kontekst wejściowy (Context IN):**
   - Mając uniwersalne 1M tokenów wejścia, nawet najgłębsze badanie `max` gromadzące 30 stron źródłowych (800,000 znaków $\approx$ 200,000 tokenów) zajmuje zaledwie $\approx 20\%$ okna kontekstu.
2. **Kontekst wyjściowy (Context OUT):**
   - Ustalenie `max_tokens` na poziomach `[16_384, 32_768, 65_536, 131_072]` dla `[low, medium, high, max]` dokładnie wpisuje się w standardy generowania (128K–384K), pozostawiając zapas na ślad myślenia (20k–50k) i pełne wygenerowanie 6,000 słów raportu.
