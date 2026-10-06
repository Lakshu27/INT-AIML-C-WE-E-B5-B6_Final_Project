# Data sources and what to download

## Already inside this project
| File | Use |
|---|---|
| `data/raw/coverwise_synthetic_coverage_risk_dataset_500.csv` | Training / testing the risk model (from the resource pack) |
| `data/raw/coverwise_policy_qa_risk_testset_60.csv` | Generic Q&A behaviour tests (from the resource pack) |
| `data/sample_policies/*.pdf` | Two **fictional** policies (Nilgiri Health Insurance - invented) for an offline demo |
| `data/eval/sample_policy_ground_truth.json` | Expected clause values for the sample PDFs |
| `data/eval/sample_policy_qa.csv` | 20 policy-specific questions with expected answers |

Resource pack: https://drive.google.com/file/d/1csdcu8VjGlw_4hiRhCP9T51tBxysIzaC/view?usp=drive_link

## Recommended real policy PDFs for the demo (public documents - already tested with this project)
Download into `data/sample_policies/` (they are public product documents, not customer data).

| Role in demo | Document | Link |
|---|---|---|
| Current policy (basic, restrictive) | IFFCO-Tokio Arogya Sanjeevani - policy wording | https://www.iffcotokio.co.in/content/dam/iffcotokio/iffco-pdf/policy-wordings-arogya-sanjeevani-policy.pdf |
| Proposed policy (comprehensive) | HDFC ERGO my: Optima Secure - policy wording | https://www.hdfcergo.com/docs/default-source/downloads/policy-wordings/health/optima-secure-revision-pw.pdf |
| Short-form test | HDFC ERGO my: Optima Secure - Customer Information Sheet | https://www.hdfcergo.com/docs/default-source/downloads/cis/cis---myoptimasecure.pdf |

If a link moves, use the insurer download pages from the brief:
* IRDAI health products: https://irdai.gov.in/health-insurance-products
* HDFC ERGO wordings / brochures / prospectus / CIS: https://www.hdfcergo.com/download/policy-wordings/health · https://www.hdfcergo.com/download/brochure/health · https://www.hdfcergo.com/download/prospectus/health · https://www.hdfcergo.com/download/cis/health
* Star Health: https://www.starhealth.in/downloads/
* Care Health: https://www.careinsurance.com/other-downloads.html · https://www.careinsurance.com/health-insurance-brochure.html · https://www.careinsurance.com/customer-information-sheet.html
* Niva Bupa: https://transaction.nivabupa.com/pages/downloads.aspx

## Optional datasets (not used by the core model)
* Premium prediction practice (no clause fields): https://drive.google.com/file/d/1USvXMotqZctBbA1euFEw5Gyx9gyut0_G/view?usp=drive_link → `notebooks/02_optional_premium_practice.ipynb`, save as `data/raw/acko_dataset_modified.csv` (git-ignored, ~219 MB).
* Bitext insurance chatbot dataset: https://huggingface.co/datasets/bitext/Bitext-insurance-llm-chatbot-training-dataset
* InsuranceQA-v2: https://huggingface.co/datasets/deccan-ai/insuranceQA-v2

## Privacy rules
* Never upload real customer policies (names, policy numbers, addresses, medical details) to GitHub, Hugging Face or any public demo.
* The app masks emails, phone numbers, PAN, Aadhaar, policy numbers, names after "Insured name:" and DOB before indexing or sending text to an LLM - but masking is best-effort, so still use public documents only.
