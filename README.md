# 🛡️ CoverWise AI — Coverage Insight and Risk Intelligence System

**AI-powered Insurance Policy Decoder, Coverage Comparison and Risk Explanation Assistant**
GUVI | HCL Capstone · Domain: InsurTech / Health Insurance Analytics / Policy Intelligence / Consumer Protection / Corporate Employee Benefits / Financial Decision Support

> ⚠️ **Responsible AI notice** — CoverWise AI is a *policy-understanding and decision-support* assistant. It does **not** provide financial, legal, medical or insurance advice and never tells anyone to buy, switch or cancel a policy. The final decision always stays with the user and a licensed insurance advisor / insurer representative.

| | |
|---|---|
| 🔗 **Live demo** | `https://huggingface.co/spaces/<your-username>/coverwise-ai` *(add after deployment)* |
| 🎥 **Video walkthrough** | `<add link>` |
| 📦 **Dataset resource pack** | https://drive.google.com/file/d/1csdcu8VjGlw_4hiRhCP9T51tBxysIzaC/view?usp=drive_link |
| 🧪 **Extraction accuracy** | **88 / 88 fields (100%)** on 5 PDFs incl. 3 real Indian policy documents (offline and with LLM) |
| ⚖️ **RAG Q&A (LLM-as-judge)** | groundedness **4.88/5** · clause accuracy **4.94/5** · hallucination **0%** · unsafe advice **0%** |
| 🤖 **Best risk model** | Logistic Regression — CV macro-F1 **0.83**, test accuracy **0.84** |

---

## 📑 Table of Contents
1. [Problem Statement](#1-problem-statement)
2. [What CoverWise Does](#2-what-coverwise-does)
3. [Business Use Cases](#3-business-use-cases)
4. [Tech Stack](#4-tech-stack)
5. [System Architecture](#5-system-architecture)
6. [Project Structure](#6-project-structure)
7. [Datasets](#7-datasets)
8. [Dataset Analysis (EDA findings)](#8-dataset-analysis-eda-findings)
9. [Pipeline — Step by Step](#9-pipeline--step-by-step)
10. [Coverage Clarity / Risk Scoring Model](#10-coverage-clarity--risk-scoring-model)
11. [RAG-based Policy Q&A](#11-rag-based-policy-qa)
12. [Installation & Setup](#12-installation--setup)
13. [Running the Application](#13-running-the-application)
14. [Using the Dashboard (with real policies)](#14-using-the-dashboard-with-real-policies)
15. [Command-Line Usage](#15-command-line-usage)
16. [Evaluation & Results](#16-evaluation--results)
17. [Real Policy Analysis Results](#17-real-policy-analysis-results)
18. [Output Report](#18-output-report)
19. [Deployment (Hugging Face / AWS EC2 / Docker)](#19-deployment)
20. [Responsible AI, Privacy & Security](#20-responsible-ai-privacy--security)
21. [Configuration Reference](#21-configuration-reference)
22. [Testing](#22-testing)
23. [Troubleshooting](#23-troubleshooting)
24. [Limitations & Future Work](#24-limitations--future-work)
25. [Project Deliverables Checklist](#25-project-deliverables-checklist)
26. [Skills Demonstrated](#26-skills-demonstrated)
27. [Disclaimer](#27-disclaimer)

---

## 1. Problem Statement

People buy, renew or switch health insurance policies by comparing only the **premium** and **sum insured**. The clauses that decide what is actually paid at claim time — waiting periods, co-payment, room-rent limits, ICU caps, disease-wise sub-limits, exclusions, restoration benefit and renewal conditions — are buried in long, legalistic policy wordings (often 40–60 pages).

As a result, users often do **not** know:
- what is covered and what is excluded,
- whether co-payment applies and when,
- whether room rent / ICU charges are capped,
- how long the pre-existing disease (PED) waiting period is,
- whether specific treatments have sub-limits,
- whether maternity, newborn, child or family cover is included,
- whether a premium increase is justified by better benefits,
- whether switching creates new waiting-period risk,
- which clauses are unclear and must be confirmed, and
- what exact questions to ask before buying or switching.

**CoverWise AI** lets a user upload one or two policy PDFs and get the key clauses extracted with page references, explained in plain English, compared side by side, scored for coverage clarity / risk, and summarised in a downloadable report — together with the questions to ask an advisor.

## 2. What CoverWise Does

| Capability | Description |
|---|---|
| 📄 **PDF ingestion** | Digital PDFs via PyMuPDF (pdfplumber fallback); scanned pages via Tesseract OCR |
| 🧹 **Cleaning & structuring** | Removes repeated headers/footers and page numbers, masks personal data, splits into sections (coverage, exclusions, waiting period, claims, renewal, definitions…) |
| 🔍 **Clause extraction** | 50+ fields: insurer, plan, UIN, sum insured, premium, pre/post-hospitalisation days, waiting periods, co-pay, room rent, ICU, sub-limits, exclusions, restoration, NCB, maternity, newborn, child, AYUSH, OPD, claim timelines, grace period, portability… each with **page evidence** |
| ✅ **Validation** | LLM output and rule-based output are cross-checked; disagreements shown as *Validation notes* |
| ⚖️ **Comparison** | Current vs proposed: improved / worse / same / unclear for 20 aspects, premium vs coverage note, switching risks, profile-specific cautions |
| 📊 **Scoring** | Coverage Clarity Score (0–100), risk label, model confidence, top reasons (transparent points) |
| 🚩 **Risk flags** | 10 model flags + profile flags (age, pre-existing condition, child planning) + "verify" flags for missing values |
| 💬 **RAG Q&A** | Answers only from the uploaded document with `[S#]` source citations; says *"Not clearly found"* instead of guessing; refuses buy/switch verdicts |
| ❓ **Advisor checklist** | 8–12 concrete questions to ask the insurer / agent |
| 🧾 **PDF report** | All 8 required report sections, downloadable |
| 🔌 **Works offline** | Every LLM step has a rule-based fallback — runs without an API key |

## 3. Business Use Cases

1. **Insurance Policy Decoder** – long policy wording → simple explanation.
2. **Coverage and Restriction Analyzer** – benefits, exclusions, co-pay, waiting periods, room-rent limits, sub-limits.
3. **Current vs Proposed Policy Comparison** – what improves, what gets worse, what is unclear.
4. **Premium vs Coverage Understanding** – is the extra premium linked to meaningful improvements?
5. **Family / Child Coverage Checker** – dependent child, newborn, maternity, floater, age limits.
6. **Claim-time Risk Awareness** – clauses that can reduce or block a claim.

## 4. Tech Stack

| Layer | Tools |
|---|---|
| Language | Python 3.10+ |
| PDF & OCR | PyMuPDF, pdfplumber, pytesseract + Tesseract, Pillow |
| NLP / RAG | sentence-transformers (`all-MiniLM-L6-v2`), FAISS, scikit-learn TF-IDF (hybrid retrieval) |
| LLM | **Groq** (`openai/gpt-oss-120b`, default) or **Google Gemini** (`gemini-2.5-flash`) |
| ML | scikit-learn — Logistic Regression, Decision Tree, Random Forest, Gradient Boosting |
| UI | Streamlit |
| Reports | ReportLab |
| Analysis | pandas, NumPy, matplotlib, seaborn, Jupyter |
| Deployment | Docker, Hugging Face Spaces, AWS EC2 (optional) |
| Testing | pytest (with a fake LLM for offline tests) |

## 5. System Architecture

```
                ┌─────────────────────────── Streamlit Dashboard / CLI ───────────────────────────┐
                │ upload current PDF · upload proposed PDF · user profile · premium / SI override │
                └───────────────────────────────────────┬─────────────────────────────────────────┘
                                                        ▼
 PDF ─► extract_policy_text (PyMuPDF / OCR) ─► clean_pages (+ PII masking) ─► split_sections ─► chunk_sections
                                                        │                                         │
                                                        ▼                                         ▼
                               regex/rule extractor (full text, page evidence)        build_rag_index (FAISS + TF-IDF)
                                                        │                                         │
                                                        │      LLM extractor (Groq / Gemini) ◄────┤ retrieved, page-tagged chunks
                                                        ▼                 │                       │
                                       merge + validate (types, ranges, cross-check)              │
                                                        ▼                                         │
                   build_features ─► score_policy (rules + ML) ─► risk flags ─► advisor questions │
                                                        ▼                                         ▼
               summary · compare_policies · switching risks · RAG Q&A with citations · generate_report (PDF)
```

**Key design decisions**
1. **Two extractors, one truth table** – the LLM reads retrieved clauses; the rule extractor reads the whole document. Disagreements on co-pay, PED waiting, room rent, ICU, restoration, maternity and pre/post days are surfaced, not hidden.
2. **Training–serving consistency** – the `risk_flags_count` feature is generated at inference with the same 10 rules that reproduce it in the training data.
3. **No label leakage** – `coverage_clarity_score` is never an input when predicting `risk_label`.
4. **Conservative defaults** – multi-option waiting periods use the longest option; missing values are imputed only for scoring and always shown as *Verify* flags.
5. **Grounded answers** – citations are mandatory; unsupported questions get *"Not clearly found in the uploaded document."*

## 6. Project Structure

```
CoverWise_AI/
├── app/
│   └── streamlit_app.py            # dashboard (7 tabs)
├── coverwise/                      # core library
│   ├── config.py                   # paths, env vars, labels, disclaimer
│   ├── pdf_extract.py              # extract_policy_text (PyMuPDF / pdfplumber / OCR)
│   ├── text_clean.py               # cleaning, PII masking, section detection, sentence splitter
│   ├── rag.py                      # chunk_sections, build_rag_index (FAISS + TF-IDF hybrid)
│   ├── llm.py                      # Groq / Gemini client with retries + JSON parsing
│   ├── prompts.py                  # prompt loader
│   ├── extraction.py               # extract_clauses: rule + LLM extraction, validation, merge
│   ├── features.py                 # build_features + 10 risk-flag rules
│   ├── scoring.py                  # rule-based score, train_models, score_policy
│   ├── analysis.py                 # risk flags, compare_policies, advisor questions, guardrails
│   ├── qa.py                       # answer_question (RAG), policy summary
│   ├── report.py                   # generate_report (PDF)
│   └── pipeline.py                 # analyze_policy / analyze_pair orchestration
├── prompts/                        # system_guardrails, extraction, rag_qa, summary, comparison,
│                                   # advisor_questions, judge (LLM-as-judge)
├── scripts/
│   ├── train_model.py              # train & compare models
│   ├── evaluate_extraction.py      # field accuracy vs ground truth
│   ├── evaluate_qa.py              # RAG Q&A evaluation + LLM-as-judge
│   ├── run_analysis.py             # CLI analysis + PDF report
│   ├── check_llm.py                # verifies the Groq / Gemini API call
│   └── make_sample_policies.py     # generates the 2 fictional sample PDFs
├── notebooks/
│   ├── 01_eda_and_model_comparison.ipynb
│   └── 02_optional_premium_practice.ipynb
├── data/
│   ├── raw/                        # resource-pack CSVs (500-row dataset, 60-question test set)
│   ├── sample_policies/            # fictional sample PDFs (+ put real public PDFs here)
│   └── eval/                       # ground truth JSON + policy-specific Q&A CSV
├── models/                         # coverage_risk_model.joblib, model_metrics.json
├── reports/                        # sample PDF reports + evaluation CSVs
├── docs/                           # DATA_SOURCES.md, DEPLOYMENT.md, EVALUATION.md
├── tests/test_pipeline.py
├── .streamlit/config.toml
├── Dockerfile
├── requirements.txt
├── .env.example
└── .gitignore
```

## 7. Datasets

### Dataset 1 — Policy documents (PDFs)
Public policy wordings, brochures, prospectus and Customer Information Sheets (CIS).

**Tested with this project** (download and place in `data/sample_policies/`):

| Role in demo | Document | Link |
|---|---|---|
| Current policy | IFFCO-Tokio **Arogya Sanjeevani** – policy wording (37 pages) | https://www.iffcotokio.co.in/content/dam/iffcotokio/iffco-pdf/policy-wordings-arogya-sanjeevani-policy.pdf |
| Proposed policy | HDFC ERGO **my: Optima Secure** – policy wording (53 pages) | https://www.hdfcergo.com/docs/default-source/downloads/policy-wordings/health/optima-secure-revision-pw.pdf |
| Short-form test | HDFC ERGO **my: Optima Secure** – CIS (11 pages) | https://www.hdfcergo.com/docs/default-source/downloads/cis/cis---myoptimasecure.pdf |

**Bundled (no download needed):** two **fictional, anonymized** policies from an invented insurer — `current_policy_nilgiri_arogya_basic.pdf` and `proposed_policy_nilgiri_arogya_plus.pdf` — with known ground truth.

Other public sources from the brief: IRDAI health products (https://irdai.gov.in/health-insurance-products), HDFC ERGO wordings / brochures / prospectus / CIS, Star Health (https://www.starhealth.in/downloads/), Care Health, Niva Bupa — full list in `docs/DATA_SOURCES.md`.

### Dataset 2 — `coverwise_synthetic_coverage_risk_dataset_500.csv` (main ML dataset)
500 rows × 21 columns: `policy_id, policy_type, sum_insured, annual_premium, premium_to_coverage_ratio, waiting_period_months, pre_existing_disease_waiting_period_months, copayment_percentage, room_rent_limit_present, icu_limit_present, disease_sub_limit_count, exclusion_count, restoration_benefit_present, no_claim_bonus_present, child_coverage_present, maternity_coverage_present, claim_process_clarity_score, ambiguous_clause_count, risk_flags_count, coverage_clarity_score, risk_label`.

### Dataset 3 — `coverwise_policy_qa_risk_testset_60.csv` (Q&A behaviour test set)
60 rows: `test_id, question_category, user_question, target_clause_or_feature, expected_answer_guidance, must_include_citation, unsafe_or_overclaim_to_avoid, score_dimension`.
Complemented by **`data/eval/sample_policy_qa.csv`** — 20 policy-specific questions with expected answers, keywords and pages (including unanswerable and "should I buy?" questions).

### Optional
- Premium-prediction dataset (~1.2 M rows, ~219 MB) — practice only, `notebooks/02_*`. Save as `data/raw/acko_dataset_modified.csv` (git-ignored). Link: https://drive.google.com/file/d/1USvXMotqZctBbA1euFEw5Gyx9gyut0_G/view?usp=drive_link
- Bitext insurance chatbot dataset: https://huggingface.co/datasets/bitext/Bitext-insurance-llm-chatbot-training-dataset
- InsuranceQA-v2: https://huggingface.co/datasets/deccan-ai/insuranceQA-v2

## 8. Dataset Analysis (EDA findings)

Full analysis: `notebooks/01_eda_and_model_comparison.ipynb`.

| Finding | Evidence | What we did |
|---|---|---|
| **Imbalanced labels** | Moderate 181 · Strong 176 · Caution 110 · **High Risk 33** | Stratified splits, `class_weight="balanced"`, macro-F1 as the main metric |
| **Overlapping score bands** | Strong 78–100, Moderate 58–87, Caution 38–57, High Risk ≤ 37 | Label predicted with a classifier, not a fixed threshold |
| **Leakage risk** | `coverage_clarity_score` is derived from the same information as the label | Excluded from model inputs |
| **Score is nearly linear** | Linear regression explains **96%** (R² = 0.96) of `coverage_clarity_score` | Rounded coefficients = transparent rule-based points |
| **`risk_flags_count` is rule-based** | Count of 10 simple rules reproduces it (corr **0.95**) | Same 10 rules used on real PDFs (training/serving consistency) |
| **Premium dataset not suitable** | 1.2 M rows, no clause fields, ~35% missing target | Kept as optional practice only |
| **Q&A set has repeats** | 12 unique questions × 5 | Evaluate unique questions + added 20 policy-specific Q&A |

**The 10 risk-flag rules**

| # | Rule | Why it matters |
|---|---|---|
| 1 | PED waiting ≥ 36 months | Existing conditions not covered for years |
| 2 | Co-payment ≥ 10% | You pay that share of every claim |
| 3 | Room rent capped | Proportionate deduction on the whole bill |
| 4 | ICU capped | Long ICU stays exceed the cap |
| 5 | Disease sub-limits ≥ 3 | Common surgeries paid only up to fixed amounts |
| 6 | Exclusions ≥ 15 | More non-payable situations |
| 7 | Ambiguous clauses ≥ 3 | Room for claim disputes |
| 8 | No restoration benefit | No top-up if sum insured is exhausted |
| 9 | Claim-process clarity < 60 | Higher chance of claim delays |
| 10 | Specified-disease waiting ≥ 24 months | Listed illnesses not covered for 2+ years |

## 9. Pipeline — Step by Step

| Step | Function | Details |
|---|---|---|
| 1. Extract text | `extract_policy_text` | Page-wise text; OCR when a page has < 40 characters |
| 2. Clean | `clean_pages` | De-hyphenation, header/footer removal (lines repeated on ≥ 50% of pages), page numbers, PII masking |
| 3. Sections | `split_sections` | Numbered / uppercase headings, IRDAI codes (Excl01–03 → waiting period, Excl04+ → exclusions), list items not mistaken for headings |
| 4. Chunk | `chunk_sections` | ~1,000-char chunks, 150 overlap, never crossing a section; metadata = policy, page range, section title, category |
| 5. Index | `build_rag_index` | FAISS (cosine) + TF-IDF hybrid; insurance synonym expansion ("copay" → co-payment, cost sharing…) |
| 6. Extract | `extract_clauses` | Rule extractor + LLM JSON extraction in 4 groups (overview, coverage, restrictions, claims/renewal) |
| 7. Validate | `validate_extraction` | Type coercion ("4 years" → 48 months), plausibility ranges, LLM vs rule cross-check |
| 8. Features | `build_features` | 17 numerical features + policy type; clipped to training range; missing values flagged |
| 9. Score | `score_policy` | ML label + probability + clarity score; rule-based points as reasons |
| 10. Flags & questions | `generate_risk_flags`, `advisor_questions` | Model, profile and verification flags; concrete advisor questions |
| 11. Compare | `compare_policies` | 20 aspects, premium vs coverage, switching risks |
| 12. Explain | `policy_summary`, `answer_question` | 10–15 line summary, grounded Q&A with citations |
| 13. Report | `generate_report` | Downloadable PDF |

## 10. Coverage Clarity / Risk Scoring Model

**Input:** 17 features — `sum_insured, annual_premium, premium_to_coverage_ratio, waiting_period_months, pre_existing_disease_waiting_period_months, copayment_percentage, room_rent_limit_present, icu_limit_present, disease_sub_limit_count, exclusion_count, restoration_benefit_present, no_claim_bonus_present, child_coverage_present, maternity_coverage_present, claim_process_clarity_score, ambiguous_clause_count, risk_flags_count` + one-hot `policy_type`.

**Target:** `risk_label` → Strong Coverage · Moderate Coverage · Caution Required · High Risk / Needs Advisor Review.

**Output:** Coverage Clarity Score /100 · Risk Level · prediction probability · top reasons.

**Transparent rule-based score** (derived from the regression):
```
score = 103 − 7 × risk_flags − 0.75 × co-pay% − 0.25 × PED months − 2 × ambiguous clauses
        + 10 (restoration) + 5 (no-claim bonus) + 1 (maternity) + 0.5 (child cover)      → clipped 0–100
```

**Model comparison** (80/20 stratified split, 5-fold CV on training data):

| Model | CV macro-F1 | Test accuracy | Test macro-F1 |
|---|---|---|---|
| **Logistic Regression ✅ selected** | **0.829 ± 0.039** | **0.84** | 0.77 |
| Gradient Boosting | 0.751 ± 0.072 | 0.79 | 0.72 |
| Decision Tree | 0.660 ± 0.052 | 0.73 | 0.66 |
| Random Forest | 0.641 ± 0.085 | 0.81 | 0.67 |
| Transparent rule baseline | – | 0.85 | 0.79 |

Per-class F1 (test): Strong **0.90** · Moderate **0.88** · Caution **0.79** · High Risk **0.53** (only 33 training samples).
Clarity-score regressor (Gradient Boosting): **MAE 3.18 points, R² 0.957**.

Confusion matrix (rows = true, cols = predicted; Strong, Moderate, Caution, High Risk):
```
[[31,  4,  0,  0],
 [ 3, 32,  1,  0],
 [ 0,  1, 17,  4],
 [ 0,  0,  3,  4]]
```
Interpretation: the synthetic score is close to linear, so the simple linear model and the transparent rule baseline perform as well as tree ensembles — a strong argument for **explainable scoring** in insurance. The weak High Risk class is why every High Risk result is labelled *Needs Advisor Review* and shows its reasons. The model auto-retrains if the saved file is missing or incompatible with your scikit-learn version.

## 11. RAG-based Policy Q&A

- Retrieves the top-5 chunks (hybrid FAISS + TF-IDF, definitions down-weighted unless the user asks "what does X mean").
- The LLM answers **only from the retrieved context**, cites `[S1]`, `[S2]`… and ends with a *Verify:* line.
- If retrieval is weak or a key term is absent → **"Not clearly found in the uploaded document."**
- "Should I buy / switch?" questions → factual *strengths / cautions / questions to ask*, never a verdict.
- Output is sanitised for phrases such as "you should buy".
- Offline mode returns the most relevant cited sentences from the document.

Example questions: *Is co-payment applicable? · What is the PED waiting period? · Is room rent capped? · Does the policy cover a newborn or dependent child? · What are the major exclusions? · What should I verify before switching?*

## 12. Installation & Setup

### Prerequisites
- Python 3.10+ (tested on 3.11 / 3.12), Git
- Optional: Tesseract OCR for scanned PDFs (`sudo apt install tesseract-ocr`, or the Windows installer)
- A Groq or Gemini API key (optional — the app runs offline without one)

### Steps
```bash
git clone https://github.com/<your-username>/CoverWise_AI.git
cd CoverWise_AI

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
cp .env.example .env               # Windows: copy .env.example .env
```

### Add your API key (never commit it)
Edit `.env`:
```
LLM_PROVIDER=groq
GROQ_API_KEY=<your-groq-key>
GROQ_MODEL=openai/gpt-oss-120b
```
or Gemini:
```
LLM_PROVIDER=gemini
GEMINI_API_KEY=<your-gemini-key>
GEMINI_MODEL=gemini-2.5-flash
```
Get keys: Groq → https://console.groq.com/keys · Gemini → https://aistudio.google.com/apikey

> 🔐 Keep keys only in `.env` (git-ignored) or in deployment *Secrets*. Never paste keys into code, chats, screenshots or GitHub. If a key is ever exposed, revoke it immediately and create a new one.

### Verify the API call
```bash
python scripts/check_llm.py
```
Expected output (the key is never printed in full):
```
.env found       : True
Provider         : groq
Key loaded       : yes (gsk_...abcd)
Client           : groq:openai/gpt-oss-120b
Plain call       : 'COVERWISE OK'
JSON call        : {'status': 'ok', 'copayment_percentage': 5}
API call works - the app will use the LLM.
```
The script explains common failures (wrong file name, invalid / revoked key, unavailable model, rate limit, network).

### Train the model
```bash
python scripts/train_model.py      # ~5 seconds → models/coverage_risk_model.joblib + model_metrics.json
```

## 13. Running the Application

```bash
streamlit run app/streamlit_app.py
```
Open http://localhost:8501.

Low RAM / slow internet? Set `USE_EMBEDDINGS=0` in `.env` to skip the embedding-model download (TF-IDF retrieval is used instead).

## 14. Using the Dashboard (with real policies)

**Sidebar**
1. **Language model** – provider (Groq / Gemini / offline), API key (session only), model name.
2. **Your profile** – age, family members, pre-existing condition, planning a child, adding a child, main concern.
3. **Premium / sum insured** – fill when the document does not state *your* values (wordings list options, not your schedule). Example: Arogya SI = 500000, Optima SI = 1000000, plus premiums from your quotes.

**Main page**
1. Upload **Current policy PDF** → `policy-wordings-arogya-sanjeevani-policy.pdf`
2. Upload **Proposed policy PDF** → `optima-secure-revision-pw.pdf`
3. Untick *Use the bundled fictional sample policies* (tick it for an instant offline demo).
4. Click **Analyze policy**.

**Tabs**

| Tab | What you see |
|---|---|
| Summary | Score stamp + 10–15 line plain-English summary per policy |
| Clauses | Every extracted field with page number and evidence quote, exclusions list, unclear clauses, validation notes |
| Score & risk flags | Score, label, class probabilities, points table (reasons), features, flags with severity and source page |
| Comparison | Colour-coded current vs proposed table, stronger / caution / unclear lists, premium vs coverage, switching risks, AI explanation |
| Ask the policy | Chat with citations and expandable sources |
| Advisor questions | Questions to ask before deciding |
| Report | Download the full PDF report |

> Tip: the CIS (`cis---myoptimasecure.pdf`) is a summary of the Optima wording — use it for single-policy analysis or Q&A, not as a "different" policy in the comparison.

## 15. Command-Line Usage

```bash
# Full analysis + PDF report
python scripts/run_analysis.py \
  --current  data/sample_policies/policy-wordings-arogya-sanjeevani-policy.pdf \
  --proposed data/sample_policies/optima-secure-revision-pw.pdf \
  --current-si 500000 --proposed-si 1000000 \
  --current-premium <amount> --proposed-premium <amount> \
  --age 34 --planning-child \
  --out reports/my_report.pdf

# Same without an LLM
python scripts/run_analysis.py --current data/sample_policies/current_policy_nilgiri_arogya_basic.pdf --offline

# Regenerate the fictional sample PDFs + ground truth
python scripts/make_sample_policies.py
```
CLI flags: `--current`, `--proposed`, `--age`, `--family`, `--has-ped`, `--planning-child`, `--current-premium`, `--proposed-premium`, `--current-si`, `--proposed-si`, `--offline`, `--out`.

## 16. Evaluation & Results

```bash
python -m pytest -q                                   # tests (offline, fake LLM included)
python scripts/evaluate_extraction.py [--offline]     # clause extraction accuracy
python scripts/evaluate_qa.py [--offline]             # RAG Q&A + LLM-as-judge (with a key)
python scripts/evaluate_qa.py --policy <pdf> --generic-only
```
Outputs go to `reports/*.csv`; detailed write-up in `docs/EVALUATION.md`.

### Clause extraction accuracy (rule-based, offline)

| Document | Pages | Fields checked | Correct |
|---|---|---|---|
| Sample current policy (fictional) | 4 | 23 | 23 |
| Sample proposed policy (fictional) | 4 | 24 | 24 |
| IFFCO-Tokio Arogya Sanjeevani – wording | 37 | 15 | 15 |
| HDFC ERGO my: Optima Secure – wording | 53 | 15 | 15 |
| HDFC ERGO my: Optima Secure – CIS | 11 | 11 | 11 |
| **Total** | | **88** | **88 (100%)** |

Bugs found on real documents and fixed during testing: a day-care procedure ("restoration of skin continuity") mistaken for restoration benefit; UIN / product name present only in page footers; "At Actuals" room rent read as capped; maternity / OPD read from optional add-ons; CIS table cells interleaved; a travel clause read as the initial waiting period.

### RAG Q&A (offline mode, 32 questions = 12 unique generic + 20 policy-specific)

| Metric | Result |
|---|---|
| Answers with source citation | **100%** |
| Unsafe buy / switch advice | **0%** |
| Keyword recall on answerable questions | **94%** |
| Correct "not found" on unanswerable questions | **100%** |
| False "not found" on answerable questions | **0%** |

### With LLM (Groq `openai/gpt-oss-20b`, hybrid LLM + rules)

| Metric | Result |
|---|---|
| Extraction accuracy (LLM + rule cross-check) | **100% (88 / 88 fields, 5 PDFs)** |
| LLM-as-judge – groundedness (1–5) | **4.88** |
| LLM-as-judge – clause accuracy (1–5) | **4.94** |
| LLM-as-judge – simplicity (1–5) | **4.94** |
| LLM-as-judge – risk caution (1–5) | **4.88** |
| LLM-as-judge – citation (1–5) | **4.84** |
| Judge-flagged hallucination | **0%** |
| Unsafe buy / switch advice | **0%** |
| Correct "not found" on unanswerable questions | **100%** |
| False "not found" on answerable questions | **0%** |
| Keyword recall on answerable questions (formatting-normalised) | **91%** |
| Answers with `[S#]` source citation | **94%** (2 / 32 without a tag) |
| Processing time per policy (LLM mode) | **69 – 160 s** (4–53 pages) |

Notes for interpretation:
* Extraction accuracy is the **hybrid** system's accuracy: on 13 safety-critical fields the verified rule value wins when the LLM disagrees, and every conflict is shown as a *Validation note*.
* Keyword recall uses formatting-only normalisation (spaces, ₹, thousand separators, Unicode) — `python scripts/evaluate_qa.py --rescore reports/qa_eval_groq.csv` reproduces it without LLM calls. Raw strict-match recall was 62% because the LLM paraphrases (`2 %`, `₹40,000`, `newborn`).
* Remaining misses: SP-12 answered "not capped" (correct, but the keyword expected "no capping"); SP-09 described the claim process without the 24-hour / 30-day timelines (a genuine gap).
* Evaluations ran on `gpt-oss-20b` to stay within the Groq free-tier daily quota; the demo uses `gpt-oss-120b`.

### Mapping to the brief's evaluation metrics

| Brief metric | How it is measured here |
|---|---|
| Accuracy of the scoring model; precision / recall / F1 | `models/model_metrics.json`, notebook 01 |
| Relevance / accuracy of extracted clauses (co-pay, waiting period, room rent, exclusions) | `evaluate_extraction.py` vs ground truth |
| Quality of comparison | Comparison table + `sample_report_real_policies_offline.pdf` |
| Risk explanation & advisor questions | Report sections + judge "risk_caution" |
| Citation correctness / grounding | Citation rate + judge "groundedness", "citation" |
| Hallucination control & refusal | Correct-refusal rate + judge "hallucination" |
| Dashboard usability / report readability | Streamlit app, PDF report |
| Processing time | `processing_seconds` per policy: ~0.3–2 s offline, 69–160 s with the LLM (11–53 pages) |
| Privacy & anonymization | PII masking in `text_clean.anonymize`, unit-tested |

## 17. Real Policy Analysis Results

Offline analysis with user-entered sum insured (₹5 L current, ₹10 L proposed):

| Aspect | Arogya Sanjeevani (current) | my: Optima Secure (proposed) |
|---|---|---|
| Coverage Clarity Score | **36 / 100 – High Risk / Needs Advisor Review** | **69 / 100 – Moderate Coverage** |
| Co-payment | 5% on all claims (p.13) | none found |
| Room rent | 2% of SI, max ₹5,000/day (p.6) | At Actuals unless the schedule says otherwise (p.11) |
| ICU | 5% of SI, max ₹10,000/day (p.6) | At Actuals (p.11) |
| Pre-existing disease waiting | 36 months (p.8) | 36 months (p.28) |
| Specified-disease waiting | 24/36 months tiers (48 for joint replacement) → 36 used (p.8) | 24 months (p.28) |
| Initial waiting | 30 days | 30 days |
| Pre / post hospitalisation | 30 / 60 days | 60 / 180 days |
| Restoration | not mentioned | yes |
| Cumulative bonus | 5% per claim-free year, max 50% | yes |
| Maternity | excluded (Excl18) | excluded (Excl18) |
| Domiciliary / OPD | excluded | domiciliary covered / OPD excluded |
| Claim settlement | within 30 days | within 15 days |

Switching risks flagged: waiting periods restart (portability credit only up to the previous SI), waiting periods apply afresh on the increased SI, possible loss of accumulated bonus, keep continuity until the new policy is issued. *This is factual decision support — not a recommendation.*

## 18. Output Report

`generate_report()` produces a PDF containing:
1. Easy-to-read 10–20 line policy summary
2. Coverage and restriction report (with page references)
3. Current vs proposed comparison report
4. Coverage Clarity Score and risk level (with reasons)
5. Risk flags and caution points
6. Questions to ask the advisor / agent / insurer
7. User-specific policy understanding (profile + your Q&A)
8. Responsible-AI disclaimer and limitations

Samples: `reports/sample_report_offline.pdf` (fictional policies) and `reports/sample_report_real_policies_offline.pdf` (Arogya vs Optima).

## 19. Deployment

### A. Hugging Face Spaces (Docker) — deployed demo link
1. Create a new Space → SDK **Docker** → blank.
2. The Space's `README.md` must start with this header (add it in the Space repo only):
   ```yaml
   ---
   title: CoverWise AI
   emoji: 🛡️
   colorFrom: blue
   colorTo: green
   sdk: docker
   app_port: 7860
   ---
   ```
3. Push the code:
   ```bash
   git remote add space https://huggingface.co/spaces/<your-username>/coverwise-ai
   git push space main
   ```
4. Space → Settings → **Variables and secrets**: secret `GROQ_API_KEY`, variable `LLM_PROVIDER=groq` (optional `USE_EMBEDDINGS=0` for a lighter build).
5. The `Dockerfile` installs Tesseract, trains the model at build time and serves Streamlit on port 7860.
6. Demo only with the fictional samples or public policy wordings.

### B. AWS EC2 (optional extension)
```bash
# Ubuntu 22.04/24.04 · t3.small+ · security group: 22 (your IP), 8501
sudo apt update && sudo apt install -y python3-venv tesseract-ocr git
git clone https://github.com/<your-username>/CoverWise_AI.git && cd CoverWise_AI
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env && nano .env        # or AWS Secrets Manager / SSM Parameter Store
python scripts/train_model.py
nohup streamlit run app/streamlit_app.py --server.port 8501 --server.address 0.0.0.0 > app.log 2>&1 &
```
Production touches: Nginx reverse proxy + HTTPS (certbot), a `systemd` service, secrets in AWS Secrets Manager, and stop the instance when idle.

### C. Docker anywhere
```bash
docker build -t coverwise .
docker run -p 7860:7860 --env-file .env coverwise     # open http://localhost:7860
```

## 20. Responsible AI, Privacy & Security

- **No advice** – the system explains; it never says buy / don't buy / switch / cancel. Advice-seeking questions get a factual strengths / cautions / questions answer.
- **Grounding** – every answer cites the document; unsupported questions are refused.
- **Transparency** – every score comes with points-based reasons and page references; uncertain values are shown as *Verify*.
- **Human review** – reports end with "confirm with the insurer or a licensed advisor".
- **Privacy** – emails, phone numbers, PAN, Aadhaar, policy numbers, names after "Insured name:" and dates of birth are masked before indexing or sending text to an LLM. Use only public or anonymized documents in public demos.
- **Secrets** – API keys only in `.env` / deployment secrets; `.env` is git-ignored.
- **Known bias / data limits** – the risk model is trained on synthetic data; the score measures clarity / restrictiveness of wording, not insurer quality or claim-settlement ratio.

## 21. Configuration Reference

| Variable | Default | Meaning |
|---|---|---|
| `LLM_PROVIDER` | `groq` | `groq`, `gemini` or `none` |
| `GROQ_API_KEY` / `GROQ_MODEL` | – / `openai/gpt-oss-120b` | Groq settings |
| `GEMINI_API_KEY` / `GEMINI_MODEL` | – / `gemini-2.5-flash` | Gemini settings |
| `LLM_TEMPERATURE` | `0.1` | Low for factual extraction |
| `USE_EMBEDDINGS` | `1` | `0` = TF-IDF only |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Dense retriever |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1000` / `150` | Characters |
| `TOP_K` | `5` | Retrieved chunks per question |
| `MIN_RETRIEVAL_SCORE` | `0.12` | Below this → "not found" |
| `ENABLE_OCR` | `1` | OCR for scanned pages |

Upload limit: 25 MB (`.streamlit/config.toml`).

## 22. Testing

```bash
python -m pytest -q
```
Covers: sentence splitting, PII masking, JSON parsing, advice detection and sanitising, score bounds, the full offline pipeline + report, and every LLM code path via a `FakeLLM` (extraction merge, unit conversion, comparison narrative, advisor questions, guardrails).

## 23. Troubleshooting

| Problem | Fix |
|---|---|
| "offline (rule-based)" shown although a key is set | Run `python scripts/check_llm.py`; the file must be named `.env` (not `.env.example` / `.env.txt`) in the project root, `LLM_PROVIDER` must match the key, then restart Streamlit |
| `401 invalid_api_key` | Key revoked or mistyped - create a new key, paste after `=` with no quotes/spaces |
| Groq `404 model_not_found` / *decommissioned* | Set `GROQ_MODEL=openai/gpt-oss-120b` (or `openai/gpt-oss-20b`); check https://console.groq.com/docs/models |
| `429 rate limit` | Wait a minute; the client retries automatically; try a smaller model |
| Slow first run | The embedding model downloads once (~90 MB); or set `USE_EMBEDDINGS=0` |
| "almost no text could be extracted" | Scanned PDF → install Tesseract and keep `ENABLE_OCR=1` |
| Model load warning / sklearn version mismatch | It retrains automatically; or run `python scripts/train_model.py` |
| Sum insured / premium "Not provided" | Wordings list options — enter your values in the sidebar |
| Torch install too large (HF / laptop) | Remove `sentence-transformers` from requirements and set `USE_EMBEDDINGS=0` |

## 24. Limitations & Future Work

**Limitations**
- Synthetic training data for the risk model; scores are indicative.
- Policy wordings rarely contain *your* sum insured / premium.
- Plan-dependent limits ("unless specified in the Policy Schedule") need the actual schedule.
- Multi-option waiting periods use the longest option (conservative).
- Complex tables / multi-column brochures and poor scans reduce extraction quality.

**Future work**
- Policy-schedule upload to resolve plan-dependent limits.
- Table-aware extraction (PyMuPDF `find_tables`) for brochures.
- Multilingual explanations (Tamil, Hindi).
- Larger labelled set of real policies to train the risk model.
- Claim-scenario simulator ("₹3 L surgery in a ₹8,000/day room — how much would I pay?").

## 25. Project Deliverables Checklist

| # | Deliverable | Status |
|---|---|---|
| 1 | Python codebase (GitHub) | ✅ code ready – push to GitHub |
| 2 | Streamlit dashboard | ✅ `app/streamlit_app.py` |
| 3 | Deployed demo link (Hugging Face Spaces) | ⏳ deploy and add link at the top |
| 4 | Documentation & README | ✅ this file + `docs/` |
| 5 | Video walkthrough | ⏳ record and add link |
| 6 | Sample anonymized policy documents / inputs | ✅ `data/sample_policies/` |
| 7 | Sample output reports | ✅ `reports/` |
| 8 | Presentation deck (optional) | ⏳ optional |
| 9 | Dataset resource link section | ✅ section 7 + `docs/DATA_SOURCES.md` |

Project guidelines followed: Git version control · separate folders for app / notebooks / data / prompts / models / reports · PEP 8, modular functions (`extract_policy_text`, `build_rag_index`, `extract_clauses`, `compare_policies`, `generate_report`) · keys in environment variables · anonymized samples only · disclaimer everywhere · tested on unseen real documents · grounding checks · human review step · no buy / do-not-buy recommendations · evaluation sheet with test questions and expected answers.

## 26. Skills Demonstrated

Insurance PDF ingestion & OCR · clause extraction from policy wording · coverage, exclusion, waiting-period and co-payment detection · room-rent, ICU and sub-limit analysis · current vs proposed comparison · RAG with source references · LLM plain-English explanation · risk flags and advisor checklist · ML / rule-based risk scoring · Streamlit dashboard · PDF report generation · responsible-AI framing · evaluation (LLM-as-judge, ground-truth accuracy) · deployment.

**Technical tags:** NLP · PDF Processing · OCR · Insurance Clause Extraction · Policy Comparison · RAG · Embeddings · Vector Database (FAISS) · Prompt Engineering · LLM · Groq API · Gemini API · Machine Learning · Classification · Risk Scoring · Feature Engineering · Scikit-learn · Streamlit · Report Generation · Responsible AI

## 27. Disclaimer

CoverWise AI is for **policy understanding and decision support only**. It is not financial, legal, medical or insurance advice. AI extraction can be incomplete or wrong — always verify every point with the official policy wording, the insurer, or a licensed insurance advisor before making a decision. Policy documents referenced belong to their respective insurers; the bundled "Nilgiri Health Insurance" documents are fictional.
