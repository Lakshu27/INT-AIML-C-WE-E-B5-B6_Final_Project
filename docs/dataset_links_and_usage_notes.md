# CoverWise AI - Dataset Links and Resources

Use these resources with the Dataset Preparation section in the CoverWise AI final project document.

## Dataset 1: Sample Insurance Policy Documents / Product Brochures

Primary official resource pages:

1. IRDAI Health Insurance Products
https://irdai.gov.in/health-insurance-products

2. HDFC ERGO Health Policy Wordings
https://www.hdfcergo.com/download/policy-wordings/health

3. HDFC ERGO Health Brochures / Prospectus pages
https://www.hdfcergo.com/download/brochure/health
https://www.hdfcergo.com/download/prospectus/health

4. Star Health Downloads
https://www.starhealth.in/downloads/

5. Care Health Insurance Downloads
https://www.careinsurance.com/other-downloads.html
https://www.careinsurance.com/health-insurance-brochure.html

6. Niva Bupa Download Center
https://transaction.nivabupa.com/pages/downloads.aspx

## Dataset 2: Custom / Synthetic Coverage Risk Dataset

This pack contains:
coverwise_synthetic_coverage_risk_dataset_500.csv

It follows these fields from the project document:
policy_type, sum_insured, annual_premium, premium_to_coverage_ratio, waiting_period_months, pre_existing_disease_waiting_period_months, copayment_percentage, room_rent_limit_present, icu_limit_present, disease_sub_limit_count, exclusion_count, restoration_benefit_present, no_claim_bonus_present, child_coverage_present, maternity_coverage_present, claim_process_clarity_score, ambiguous_clause_count, risk_flags_count, coverage_clarity_score, risk_label.

Optional public ML practice dataset for premium prediction only:
https://www.kaggle.com/competitions/insurance-premium-forecasting-challenge/data
https://www.kaggle.com/datasets/imtkaggleteam/health-insurance-dataset

Note: These Kaggle datasets are not policy clause datasets. Use them only for ML feature/premium prediction practice, not for clause extraction.

## Dataset 3: Synthetic Policy Q&A and Risk Test Set

This pack contains:
coverwise_policy_qa_risk_testset_60.csv

Optional public insurance QA resources:
https://huggingface.co/datasets/bitext/Bitext-insurance-llm-chatbot-training-dataset
https://huggingface.co/datasets/deccan-ai/insuranceQA-v2

Note: These are general insurance Q&A/conversation datasets. For CoverWise, students should create document-specific Q&A from the policy PDFs they use.
