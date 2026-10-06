# Evaluation results

Re-run with the commands in the README; numbers below are from the offline (no-LLM) mode so they are
reproducible. Run again with your Groq/Gemini key to add LLM extraction and LLM-as-judge scores.

## 1. Coverage Clarity / Risk model (synthetic 500-row dataset, 80/20 stratified split)
| Model | 5-fold CV macro-F1 | Test accuracy | Test macro-F1 |
|---|---|---|---|
| **Logistic Regression (selected)** | **0.829 ± 0.039** | 0.84 | 0.77 |
| Gradient Boosting | 0.751 ± 0.072 | 0.79 | 0.72 |
| Decision Tree | 0.660 ± 0.052 | 0.73 | 0.66 |
| Random Forest | 0.641 ± 0.085 | 0.81 | 0.67 |
| Transparent rule baseline | - | 0.85 | 0.79 |

Per-class F1 (selected model, test): Strong 0.90 · Moderate 0.88 · Caution 0.79 · **High Risk 0.53** (only 33 training rows).
Clarity-score regressor (Gradient Boosting): MAE 3.18 points, R² 0.957.

Interpretation: the synthetic score is almost linear in the features, so the linear model and the
transparent rule baseline perform equally well - a good argument for transparency in insurance AI.
The weak High Risk class is why every High Risk result is labelled *Needs Advisor Review* with reasons.

## 2. Clause extraction accuracy (rule extractor, offline)
| Document | Pages | Fields checked | Correct |
|---|---|---|---|
| Sample current policy (fictional) | 4 | 23 | 23 |
| Sample proposed policy (fictional) | 4 | 24 | 24 |
| IFFCO-Tokio Arogya Sanjeevani - policy wording | 37 | 15 | 15 |
| HDFC ERGO my: Optima Secure - policy wording | 53 | 15 | 15 |
| HDFC ERGO my: Optima Secure - CIS | 11 | 11 | 11 |
| **Total** | | **88** | **88 (100%)** |

Ground truth is in `data/eval/sample_policy_ground_truth.json`; values for the real documents were
taken from the printed wording (page references in the extraction output). Errors found and fixed
while testing on the real PDFs: restoration matched an unrelated day-care procedure, UIN/product name
lived only in page headers, "At Actuals" room rent read as capped, maternity/OPD read from optional
add-ons, table cells in the CIS interleaved, and a travel clause read as the initial waiting period.

Analysis of the real documents (offline, user-entered sum insured 5 L vs 10 L):
Arogya Sanjeevani 36/100 (High Risk - 5% co-pay, room rent 2% of SI / ICU 5% of SI caps, no
restoration, 24/36/48-month specified-disease tiers) vs my: Optima Secure 69/100 (Moderate - no
co-pay, room/ICU at actuals unless scheduled, restoration and cumulative bonus; maternity excluded).
Sample report: `reports/sample_report_real_policies_offline.pdf`.

## 3. RAG Q&A (offline mode, 32 questions)
| Metric | Result |
|---|---|
| Answers with source citation | 100% |
| Unsafe buy/switch advice | 0% |
| Keyword recall on answerable policy questions | 94% |
| Correct "not found" on unanswerable questions | 100% |
| False "not found" on answerable questions | 0% |

With an LLM key, `scripts/evaluate_qa.py` adds LLM-as-judge scores (groundedness, clause accuracy,
simplicity, risk caution, citation, hallucination flag) for each answer.

## 4. With the LLM (Groq openai/gpt-oss-20b)
* Extraction (hybrid LLM + rules): **100% (88/88)** on 5 PDFs; 69-160 s per policy.
* RAG Q&A (32 questions): LLM-as-judge groundedness 4.88, clause accuracy 4.94, simplicity 4.94,
  risk caution 4.88, citation 4.84 (1-5); judge-flagged hallucination 0%; unsafe advice 0%;
  correct refusal 100%; false "not found" 0%; keyword recall 91% (formatting-normalised; 62% strict);
  citation tags 94%.
* Known gaps: SP-09 claim-process answer omitted the 24 h / 30 day timelines; SP-12 correct but
  phrased differently from the expected keyword.
