"""Generate two FICTIONAL, anonymized policy wordings for demo/testing.

The insurer ("Nilgiri Health Insurance Company Limited") and products are invented.
They follow the structure of real IRDAI policy wordings (definitions, coverage,
Excl01-Excl18 exclusions, claims, renewal) so the whole pipeline can be tested
offline and the extraction can be scored against known ground truth
(data/eval/sample_policy_ground_truth.json).

Run:  python scripts/make_sample_policies.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "sample_policies"
EVAL = ROOT / "data" / "eval"

STD_EXCL = [
    ("Excl04", "Investigation & Evaluation", "Expenses related to any admission primarily for diagnostics and evaluation purposes only are excluded."),
    ("Excl05", "Rest Cure, rehabilitation and respite care", "Expenses related to any admission primarily for enforced bed rest and not for receiving treatment are excluded."),
    ("Excl06", "Obesity / weight control", "Expenses related to surgical treatment of obesity that does not fulfil the stated conditions are excluded."),
    ("Excl07", "Change-of-gender treatments", "Expenses related to any treatment to change characteristics of the body to those of the opposite sex are excluded."),
    ("Excl08", "Cosmetic or plastic surgery", "Expenses for cosmetic or plastic surgery unless for reconstruction following an accident, burn or cancer are excluded."),
    ("Excl09", "Hazardous or adventure sports", "Expenses related to treatment necessitated due to participation as a professional in hazardous or adventure sports are excluded."),
    ("Excl10", "Breach of law", "Expenses for treatment directly arising from or consequent upon any insured person committing or attempting to commit a breach of law with criminal intent are excluded."),
    ("Excl11", "Excluded providers", "Expenses incurred towards treatment in any hospital or by any medical practitioner specifically excluded by the Company are not admissible."),
    ("Excl12", "Alcoholism and drug abuse", "Treatment for alcoholism, drug or substance abuse or any addictive condition and consequences thereof are excluded."),
    ("Excl13", "Non-medical admissions", "Treatments received in health hydros, nature cure clinics, spas or similar establishments are excluded."),
    ("Excl14", "Vitamins and tonics", "Dietary supplements and substances that can be purchased without prescription are excluded unless prescribed as part of hospitalisation."),
    ("Excl15", "Refractive error", "Expenses related to the treatment for correction of eye sight due to refractive error less than 7.5 dioptres are excluded."),
    ("Excl16", "Unproven treatments", "Expenses related to any unproven treatment, services and supplies are excluded."),
    ("Excl17", "Sterility and infertility", "Expenses related to sterility and infertility including assisted reproduction services are excluded."),
]
MATERNITY_EXCL = ("Excl18", "Maternity", "Medical treatment expenses traceable to childbirth (including complicated deliveries and caesarean sections) and expenses towards miscarriage or termination of pregnancy are excluded. New born baby expenses are not covered.")

COMMON_DEFS = [
    "<b>Co-payment</b> means a cost sharing requirement under a health insurance policy that provides that the policyholder/insured will bear a specified percentage of the admissible claims amount. A co-payment does not reduce the Sum Insured.",
    "<b>Pre-existing Disease</b> means any condition, ailment, injury or disease that is diagnosed by a physician within 48 months prior to the effective date of the policy issued by the insurer or its reinstatement.",
    "<b>Room Rent</b> means the amount charged by a hospital towards Room and Boarding expenses and shall include the associated medical expenses.",
    "<b>Grace Period</b> means the specified period of time immediately following the premium due date during which a payment can be made to renew or continue a policy in force without loss of continuity benefits.",
    "<b>Day Care Treatment</b> means medical treatment and/or surgical procedure which is undertaken under general or local anaesthesia in a hospital/day care centre in less than 24 hrs because of technological advancement.",
]

POLICIES = {
    "current_policy_nilgiri_arogya_basic.pdf": {
        "name": "Nilgiri Arogya Basic Health Policy",
        "uin": "NILHLIP21001V012122",
        "schedule": "Sum Insured: Rs. 5,00,000 on family floater basis (2 adults + 1 child). Annual premium: Rs. 14,800 (inclusive of GST). Policy period: 1 year.",
        "eligibility": "Entry age for adults is 18 to 65 years. Dependent children aged 91 days to 25 years can be covered under the floater along with at least one parent.",
        "coverage": [
            ("1.1 In-patient Hospitalisation", "The Company shall indemnify medical expenses incurred for in-patient hospitalisation of the insured person for a minimum period of 24 consecutive hours, up to the Sum Insured."),
            ("1.2 Room Rent and ICU Charges", "Room rent, boarding and nursing expenses are payable up to 1% of the Sum Insured subject to a maximum of Rs. 5,000 per day. Intensive Care Unit (ICU) / ICCU expenses are payable up to 2% of the Sum Insured subject to a maximum of Rs. 10,000 per day. If the insured person opts for a room with rent higher than the eligible limit, all associated medical expenses shall be reduced proportionately."),
            ("1.3 Pre-hospitalisation Expenses", "Medical expenses incurred during the 30 days immediately before the date of admission are covered, provided the in-patient claim is admissible."),
            ("1.4 Post-hospitalisation Expenses", "Medical expenses incurred during the 60 days immediately after discharge are covered, provided the in-patient claim is admissible."),
            ("1.5 Day Care Procedures", "Day care treatment taken in a hospital or day care centre is covered up to the Sum Insured."),
            ("1.6 Road Ambulance", "Expenses for road ambulance are covered up to Rs. 2,000 per hospitalisation."),
            ("1.7 AYUSH Treatment", "In-patient AYUSH treatment taken in a government recognised AYUSH hospital is covered up to the Sum Insured."),
            ("1.8 Cumulative Bonus", "For every claim-free policy year, a cumulative bonus of 10% of the Sum Insured is added, up to a maximum of 50%. The bonus is reduced by 10% in the year following a claim."),
            ("1.9 Restoration of Sum Insured", "Restoration benefit is not available under this policy. Once the Sum Insured and Cumulative Bonus are exhausted, no further claims are payable in that policy year."),
            ("1.10 Domiciliary Hospitalisation", "Domiciliary hospitalisation is not covered under this policy."),
        ],
        "limits": [
            ("2.1 Co-payment", "Each and every claim under the policy shall be subject to a co-payment of 10% of the admissible claim amount. For insured persons aged 61 years and above at entry, the co-payment shall be 20%."),
            ("2.2 Disease-wise Sub-limits", "The following sub-limits apply per policy year: Cataract treatment is limited to 25% of the Sum Insured or Rs. 40,000 per eye, whichever is lower. Knee replacement surgery is limited to Rs. 1,50,000 per knee. Hernia surgery is limited to Rs. 50,000. Modern treatments such as robotic surgery and oral chemotherapy are limited to 50% of the Sum Insured."),
        ],
        "ped": 48, "sdw": 24, "maternity": None,
        "claims_extra": "",
        "ambig": "The Company may, at its sole discretion, seek additional documents. Payment shall be limited to reasonable and customary charges for the treatment.",
    },
    "proposed_policy_nilgiri_arogya_plus.pdf": {
        "name": "Nilgiri Arogya Plus Health Policy",
        "uin": "NILHLIP24007V022425",
        "schedule": "Sum Insured: Rs. 10,00,000 on family floater basis (2 adults + 1 child). Annual premium: Rs. 21,600 (inclusive of GST). Policy period: 1 year.",
        "eligibility": "Entry age for adults is 18 to 65 years. Dependent children aged 91 days to 25 years can be covered. A new born baby can be added as a dependent child from the 91st day.",
        "coverage": [
            ("1.1 In-patient Hospitalisation", "The Company shall indemnify medical expenses incurred for in-patient hospitalisation of the insured person for a minimum period of 24 consecutive hours, up to the Sum Insured."),
            ("1.2 Room Rent and ICU Charges", "There is no capping on room rent; the insured person may choose any room category except a suite. ICU / ICCU charges are payable at actuals up to the Sum Insured."),
            ("1.3 Pre-hospitalisation Expenses", "Medical expenses incurred during the 60 days immediately before the date of admission are covered."),
            ("1.4 Post-hospitalisation Expenses", "Medical expenses incurred during the 180 days immediately after discharge are covered."),
            ("1.5 Day Care Procedures", "All day care treatments taken in a hospital or day care centre are covered up to the Sum Insured."),
            ("1.6 Road Ambulance", "Expenses for road ambulance are covered up to Rs. 5,000 per hospitalisation."),
            ("1.7 AYUSH Treatment", "In-patient AYUSH treatment is covered up to the Sum Insured."),
            ("1.8 No Claim Bonus", "For every claim-free policy year, a no claim bonus of 50% of the base Sum Insured is added, up to a maximum of 100%."),
            ("1.9 Restoration of Sum Insured", "If the Sum Insured is exhausted during the policy year, 100% of the base Sum Insured shall be restored once in a policy year for subsequent claims for any illness or injury."),
            ("1.10 Domiciliary Hospitalisation", "Domiciliary hospitalisation for more than 3 days on medical advice is covered up to 10% of the Sum Insured."),
            ("1.11 Maternity and New Born Cover", "Maternity expenses for delivery are covered after a waiting period of 24 months, up to Rs. 50,000 for normal delivery and Rs. 75,000 for caesarean section, limited to two deliveries during the lifetime of the policy. New born baby expenses are covered from day one up to Rs. 25,000 within the maternity limit."),
            ("1.12 Out-patient Expenses", "OPD and out-patient consultation expenses are not covered under this policy."),
        ],
        "limits": [
            ("2.1 Co-payment", "There is no co-payment under this policy irrespective of the age of the insured person. However, if treatment is taken in a Zone A city when the premium was paid for Zone B, a co-payment of 15% shall apply on that claim."),
            ("2.2 Disease-wise Sub-limits", "Cataract treatment is limited to Rs. 1,00,000 per eye per policy year. No other disease-wise sub-limits apply."),
        ],
        "ped": 36, "sdw": 24, "maternity": 24,
        "claims_extra": "",
        "ambig": "Benefits for modern treatments shall be as specified in the policy schedule.",
    },
}


def build(fname: str, p: dict):
    ss = getSampleStyleSheet()
    h = ParagraphStyle("h", parent=ss["Heading2"], fontSize=12, spaceBefore=8)
    h3 = ParagraphStyle("h3", parent=ss["Heading4"], fontSize=10, spaceBefore=4)
    body = ParagraphStyle("b", parent=ss["BodyText"], fontSize=9.5, leading=12.5)
    flow = [Paragraph("NILGIRI HEALTH INSURANCE COMPANY LIMITED", ss["Title"]),
            Paragraph(f"{p['name']} - Policy Wording", ss["Heading2"]),
            Paragraph(f"UIN: {p['uin']} | Sample document generated for CoverWise AI testing - fictional insurer.", body),
            Spacer(1, 6),
            Paragraph("PREAMBLE", h),
            Paragraph("This policy is a contract of insurance issued by Nilgiri Health Insurance Company Limited to the proposer mentioned in the schedule. The Company shall indemnify medical expenses as per the terms, conditions and exclusions of this policy.", body),
            Paragraph("POLICY SCHEDULE (ILLUSTRATION)", h),
            Paragraph(p["schedule"], body),
            Paragraph("Policyholder name: [NAME]. Policy number: [POLICY_NO]. These sample values contain no personal data.", body),
            Paragraph("ELIGIBILITY", h), Paragraph(p["eligibility"], body),
            Paragraph("SECTION A - DEFINITIONS", h)]
    flow += [Paragraph(f"A.{i} {d}", body) for i, d in enumerate(COMMON_DEFS, 1)]
    flow.append(Paragraph("SECTION B - COVERAGE (WHAT IS COVERED)", h))
    for t, d in p["coverage"]:
        flow += [Paragraph(t, h3), Paragraph(d, body)]
    flow.append(Paragraph("SECTION C - CO-PAYMENT AND SUB-LIMITS", h))
    for t, d in p["limits"]:
        flow += [Paragraph(t, h3), Paragraph(d, body)]
    flow.append(Paragraph("SECTION D - WAITING PERIODS", h))
    flow += [Paragraph("D.1 Pre-Existing Diseases - Code Excl01", h3),
             Paragraph(f"Expenses related to the treatment of a pre-existing disease and its direct complications shall be excluded until the expiry of {p['ped']} months of continuous coverage after the date of inception of the first policy with us.", body),
             Paragraph("D.2 Specified Disease/Procedure Waiting Period - Code Excl02", h3),
             Paragraph(f"Expenses related to the treatment of the listed conditions, surgeries or treatments shall be excluded until the expiry of {p['sdw']} months of continuous coverage after the date of inception of the first policy with us. Listed conditions include cataract, hernia, joint replacement, kidney stones, sinusitis and benign prostatic hypertrophy.", body),
             Paragraph("D.3 First 30 Days Waiting Period - Code Excl03", h3),
             Paragraph("Expenses related to the treatment of any illness within 30 days from the first policy commencement date shall be excluded except claims arising due to an accident.", body)]
    if p["maternity"]:
        flow += [Paragraph("D.4 Maternity Waiting Period", h3),
                 Paragraph(f"Maternity benefit is payable only after {p['maternity']} months of continuous coverage of the female insured person.", body)]
    flow.append(Paragraph("SECTION E - EXCLUSIONS (WHAT IS NOT COVERED)", h))
    excl = STD_EXCL + ([MATERNITY_EXCL] if not p["maternity"] else [])
    for i, (code, title, text) in enumerate(excl, 1):
        flow += [Paragraph(f"E.{i} {title} - Code {code}", h3), Paragraph(text, body)]
    flow += [Paragraph(f"E.{len(excl)+1} Non-payable items", h3),
             Paragraph("Items listed in Annexure I (non-payable items / consumables such as gloves, masks and toiletries) are not payable.", body),
             Paragraph("SECTION F - CLAIM PROCEDURE", h),
             Paragraph("F.1 Cashless Claims", h3),
             Paragraph("Cashless facility is available only at network hospitals. The insured person must intimate the claim within 24 hours of an emergency admission and at least 48 hours before a planned admission through the TPA helpline.", body),
             Paragraph("F.2 Reimbursement Claims", h3),
             Paragraph("For reimbursement claims, the claim form along with the discharge summary, original bills, payment receipts, investigation reports and prescriptions must be submitted within 30 days of discharge.", body),
             Paragraph("F.3 Settlement of Claims", h3),
             Paragraph("The Company shall settle or reject a claim within 30 days from the date of receipt of the last necessary document.", body),
             Paragraph(p["ambig"], body),
             Paragraph("SECTION G - RENEWAL, PORTABILITY AND GENERAL CONDITIONS", h),
             Paragraph("G.1 Renewal", h3),
             Paragraph("The policy shall ordinarily be renewable for life and renewal shall not be denied on the ground that the insured person made a claim, except on grounds of fraud or misrepresentation.", body),
             Paragraph("G.2 Grace Period", h3),
             Paragraph("A grace period of 30 days is allowed for payment of renewal premium. Coverage is not available for the period for which no premium is received.", body),
             Paragraph("G.3 Portability and Migration", h3),
             Paragraph("The insured person may port the policy to another insurer by applying at least 15 days before the renewal date. Credit for waiting periods already served will be given up to the existing sum insured.", body),
             Paragraph("G.4 Moratorium Period", h3),
             Paragraph("After completion of 60 continuous months of coverage, no policy and claim shall be contestable except for proven fraud and permanent exclusions specified in the policy.", body),
             Paragraph("G.5 Grievance Redressal", h3),
             Paragraph("Grievances may be sent to the grievance cell of the Company or to the Insurance Ombudsman.", body)]

    def frame(c, doc):
        c.saveState()
        c.setFont("Helvetica", 7)
        c.drawString(15 * mm, A4[1] - 10 * mm, f"Nilgiri Health Insurance Company Limited | {p['name']} | UIN: {p['uin']}")
        c.drawRightString(A4[0] - 15 * mm, 8 * mm, f"Page {doc.page}")
        c.restoreState()

    doc = SimpleDocTemplate(str(OUT / fname), pagesize=A4, topMargin=16 * mm, bottomMargin=15 * mm,
                            leftMargin=15 * mm, rightMargin=15 * mm, title=p["name"])
    doc.build(flow, onFirstPage=frame, onLaterPages=frame)


GROUND_TRUTH = {
    "current_policy_nilgiri_arogya_basic.pdf": {
        "uin": "NILHLIP21001V012122", "sum_insured": 500000, "annual_premium": 14800,
        "pre_hospitalization_days": 30, "post_hospitalization_days": 60,
        "ped_waiting_period_months": 48, "specific_disease_waiting_period_months": 24,
        "initial_waiting_period_days": 30, "copayment_percentage": 10.0,
        "room_rent_limit_present": True, "icu_limit_present": True, "restoration_benefit": False,
        "no_claim_bonus": True, "maternity_covered": False, "child_coverage": True,
        "domiciliary_covered": False, "ayush_covered": True, "day_care_covered": True,
        "cashless_available": True, "reimbursement_available": True, "grace_period_days": 30,
        "disease_sub_limit_count": 4, "exclusion_count": 18,
    },
    "proposed_policy_nilgiri_arogya_plus.pdf": {
        "uin": "NILHLIP24007V022425", "sum_insured": 1000000, "annual_premium": 21600,
        "pre_hospitalization_days": 60, "post_hospitalization_days": 180,
        "ped_waiting_period_months": 36, "specific_disease_waiting_period_months": 24,
        "initial_waiting_period_days": 30, "copayment_percentage": 0.0,
        "room_rent_limit_present": False, "icu_limit_present": False, "restoration_benefit": True,
        "no_claim_bonus": True, "maternity_covered": True, "child_coverage": True,
        "domiciliary_covered": True, "ayush_covered": True, "day_care_covered": True,
        "opd_covered": False, "cashless_available": True, "reimbursement_available": True,
        "grace_period_days": 30, "disease_sub_limit_count": 1, "exclusion_count": 17,
    },
}

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    EVAL.mkdir(parents=True, exist_ok=True)
    for f, spec in POLICIES.items():
        build(f, spec)
        print("wrote", OUT / f)
    (EVAL / "sample_policy_ground_truth.json").write_text(json.dumps(GROUND_TRUTH, indent=2))
    print("wrote", EVAL / "sample_policy_ground_truth.json")
    sys.exit(0)
