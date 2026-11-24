"""Sanction-letter rules on made-up letter text (no real applicant)."""
from src.idcards.sanction import extract_sanction_text, parse_amount, words_to_number

LETTER = """Subject: Sanction Letter (Education Loan Application Number A0000000001)
Applicant Ravi Kumar Test
Co-Applicant(s) Suresh Test
Country of Study Canada
Course Details Master of Engineering in Civil Engineering
University University Of Example
Sanctioned Loan Amount INR 40,00,000.00 (Forty Lakh Rupees equivalent to CAD 65,000.00 approx.)
Rate of Interest 10.75 % p.a. Floating Rate
Origination Fees (Non-
Refundable)
INR 20,000.00 (inclusive of all taxes)
Repayment Term 120 monthly installments
1 - 24 months: PMII of INR 4,000.00 per month
25 - 120 months: EMI of INR 60,000.00 per month
Loan Type UNSECURED
Date of Sanction 01 June 2026
This Sanction letter is valid till 31 May, 2027
For Example Finance Limited
General Terms and Conditions
4. Insurance (if applicable) to be availed of by the Borrower.""".splitlines()


def test_reads_the_summary_fields():
    r = extract_sanction_text(LETTER, "PDF text")
    f = {k: v["value"] for k, v in r["fields"].items()}
    assert (f["university"], f["country"], f["course"]) == ("University Of Example", "Canada",
                                                          "Master of Engineering in Civil Engineering")
    assert f["loan_amount"] == 4_000_000 and f["roi"] == 10.75 and f["roi_type"] == "Floating"
    assert f["processing_fee"] == 20_000 and f["processing_fee_percent"] == 0.5
    assert f["tenure_months"] == 120 and f["moratorium_months"] == 24 and f["emi"] == 60_000
    assert f["insurance"].startswith("Mentioned (if applicable)")
    assert (f["applicant_name"], f["co_applicants"]) == ("Ravi Kumar Test", "Suresh Test")
    assert (f["sanction_date"], f["valid_until"], f["loan_type"]) == ("2026-06-01", "2027-05-31", "Unsecured")
    assert r["checks"]["amount_words_match"] and r["checks"]["instalment_periods_match_tenure"]


def test_words_and_figures_disagree_is_flagged():
    bad = [l.replace("Forty Lakh", "Fourteen Lakh") for l in LETTER]
    r = extract_sanction_text(bad, "PDF text")
    assert r["checks"]["amount_words_match"] is False and "loan_amount" in r["review"]["reasons"]


def test_amount_formats():
    assert parse_amount("Rs. 40 Lakhs")[0] == 4_000_000
    assert parse_amount("₹ 35,00,000/-")[0] == 3_500_000
    assert parse_amount("INR 1.2 Crore")[0] == 12_000_000
    assert words_to_number("One Crore Twenty Five Lakh Rupees Only") == 12_500_000


def test_not_a_sanction_letter():
    r = extract_sanction_text(["Board of Secondary Education", "SECONDARY SCHOOL CERTIFICATE"], "OCR")
    assert r["document_type"] == "not_sanction" and r["review"]["status"] == "manual_required"


def test_label_on_its_own_line_and_pursuing_sentence():
    letter = """Applicant
Ravi Kumar Test
Country of Study
UK
Purpose
For pursuing the Master of Science Data Analytics program from University Of
Example
Sanctioned Education Loan
Amount
INR 30,00,000.00 (Thirty Lakh Rupees equivalent to GBP 23,000.00 approx.)
Interest Rate
11.00% p.a. floating rate
One time Origination Fees
(Non-refundable)
INR 25,000.00 (inclusive of all taxes)
Repayment Tenure
120 monthly installments
For Example Finance Limited""".splitlines()
    f = {k: v["value"] for k, v in extract_sanction_text(letter, "PDF text")["fields"].items()}
    assert (f["university"], f["course"]) == ("University Of Example", "Master of Science Data Analytics")
    assert (f["loan_amount"], f["roi"], f["processing_fee"], f["tenure_months"]) == (3_000_000, 11.0, 25_000, 120)


def test_course_and_university_together_and_phased_tenure():
    letter = """EDUCATION LOAN SANCTION LETTER
Dated :01/06/2026
education loan of Rs. 2500000/- (INR Twenty Five lakh only) has been sanctioned
1. Details of Course & University : COVENTRY UNIVERSITY MSC DATA SCIENCE AND
ARTIFICIAL INTELLIGENCE
3. Sanctioned Amount : Rs. 2500000/-
5. Rate of Interest : 12.5% per annum
7. Repayment Period : Loan is repayable as
PMI (Partial Monthly Interest)/ Amount Rs 1,000 /- 24 Months
Monthly EMI Amount: /- 120 Months
9. Security Offered : N/A""".splitlines()
    r = extract_sanction_text(letter, "OCR")
    f = {k: v["value"] for k, v in r["fields"].items()}
    assert f["university"] == "COVENTRY UNIVERSITY"
    assert f["course"] == "MSC DATA SCIENCE AND ARTIFICIAL INTELLIGENCE"
    assert f["loan_amount"] == 2_500_000 and r["checks"]["amount_words_match"]
    assert (f["tenure_months"], f["moratorium_months"]) == (144, 24)
    assert f["country"] == "UK" and "country" in r["review"]["reasons"]      # inferred, so flagged
    assert (f["loan_type"], f["sanction_date"]) == ("Unsecured", "2026-06-01")
