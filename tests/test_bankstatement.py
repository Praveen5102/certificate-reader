"""Bank statement rules on a made-up statement (no real account)."""
from src.idcards.bankstatement import extract_statement_lines, money_tokens


def _statement():
    rows = ["Account Name : RAVI KUMAR TEST", "IFSC : SBIN0000001", "Account Number : 00000000001234",
            "Txn Date", "Description", "Debit", "Credit", "Balance", "OPENING BALANCE", "10,000.00"]
    bal = 10_000.0
    for m in range(1, 7):                       # six months: salary, EMI, rent, spending
        for day, desc, amt in [(1, "NEFT SALARY FOR THE MONTH ACME LTD", 50_000), (3, "ACH-DR-HDFC HOUSING EMI", -15_000),
                               (5, "UPI/DR/1234/ELECTRICITY TSSPDCL/Payment", -1_500),
                               (8, "UPI/DR/5678/ZOMATO/Payment", -2_000), (12, "ATM WDL CASH", -5_000)]:
            bal += amt
            rows += [f"{day:02d}/{m:02d}/2026", desc, f"{abs(amt):,.2f}", f"{bal:,.2f}"]
    return rows


def test_balance_chain_and_monthly_summary():
    r = extract_statement_lines(_statement(), "PDF text")
    c, s = r["checks"], r["summary"]
    assert c["transactions"] == 30 and c["balance_ok"] == c["balance_checked"] == 30
    assert r["account"]["bank"] == "State Bank of India" and r["account"]["holder"] == "RAVI KUMAR TEST"
    assert r["account"]["account_number"].endswith("1234") and r["account"]["account_number"].startswith("X")
    assert s["avg_monthly_income"] == 50_000 and s["avg_monthly_emi"] == 15_000
    assert s["avg_monthly_expenses"] == 8_500 and s["avg_monthly_surplus"] == 26_500
    assert s["foir"] == 0.3 and s["bounces"] == 0 and s["risk"] == "low"


def test_edited_row_is_caught():
    rows = _statement()
    i = rows.index("ATM WDL CASH")
    rows[i + 1] = "500.00"                       # amount changed, balance left as printed
    r = extract_statement_lines(rows, "PDF text")
    assert r["checks"]["balance_ok"] == 29
    assert any("do not add up" in f["text"] for f in r["flags"])


def test_bounce_and_betting_are_flagged():
    rows = _statement() + ["15/06/2026", "RTNCHG/ACH RETURN INSUFFICIENT FUNDS", "590.00", "1.00",
                           "16/06/2026", "UPI/DR/999/DREAM11/Payment", "1.00", "0.00"]
    rows[-5] = f"{float(_statement()[-1].replace(',', '')) - 590:,.2f}"
    rows[-1] = f"{float(rows[-5].replace(',', '')) - 1:,.2f}"
    r = extract_statement_lines(rows, "PDF text")
    texts = " ".join(f["text"] for f in r["flags"])
    assert "bounce" in texts and "betting" in texts and r["summary"]["risk"] == "high"


def test_money_formats():
    vals = lambda s: [t["value"] for t in money_tokens(s)]
    assert vals("462.39(Dr) 0.00(Cr)") == [462.39, 0.0]
    assert vals("1,24,342.39CR") == [124342.39]
    assert vals("12.345.67CR") == [12345.67]          # OCR read the thousands comma as a dot
    assert vals("₹4,000") == [4000.0]
    assert vals("UPI/DR/000011112222/TESTSHOP") == []   # reference numbers are not amounts
