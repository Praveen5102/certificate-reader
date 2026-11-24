"""Bank statement analysis: transactions -> monthly income, expenses, EMIs, red flags.

Every bank prints its statement differently (SBI alone has several layouts), so
transactions are not read from fixed columns. Instead, a transaction starts at a
line beginning with a date and runs to the next one; its money amounts are found
in that block. Where the statement has a running balance, the LAST amount is the
balance and the debit/credit is the amount that moves the previous balance to it:

    previous balance +/- amount == balance        (checked on every row)

That check decides debit vs credit without knowing the column, and a failure means
a misread row or an edited statement - both are reported, never silently fixed.
Statements without a balance (e.g. UPI app histories) use DEBIT/CREDIT words.

Digital PDFs are read from their text layer; scans and photos are OCR'd.
"""
from __future__ import annotations

import datetime as dt
import re
import statistics
from collections import Counter, defaultdict
from typing import Any

# ------------------------------------------------------------------ parsing helpers
_MONTHS = {m: i + 1 for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"])}
_DATE_PATS = [
    (re.compile(r"^\s*(\d{4})[/\-](\d{1,2})[/\-](\d{1,2})(?![\d])"), "ymd"),
    (re.compile(r"^\s*(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{4}|\d{2})(?![\d])"), "dmy"),
    (re.compile(r"^\s*(\d{1,2})[\s\-/]+([A-Za-z]{3,9})[\s\-/,]+(\d{4}|\d{2})(?![\d])"), "dMy"),
    (re.compile(r"^\s*([A-Za-z]{3,9})\s+(\d{1,2}),?\s+(\d{4})(?![\d])"), "Mdy"),
]
# 1,23,456.78 / 123456.78 / ₹4,000 ; optional Cr/Dr mark ; optional foreign currency code
_MONEY = re.compile(
    r"(?<![\w.,/])(?P<cur>₹|INR|Rs\.?)?\s?(?P<num>\d{1,3}(?:[.,]\d{2,3})+[.,]\d{2}(?![\d])|\d{1,3}(?:,\d{2,3})+(?:\.\d{1,2})?|\d+\.\d{2}|(?<=₹)\s?\d+)"
    r"(?![\d.,]*\d)\s?(?P<mark>\(?\s?(?:Cr|Dr|CR|DR)\b\.?\)?)?\s?(?P<fx>SAR|AED|USD|GBP|EUR|QAR|KWD|OMR)?",
)
_SKIP_TOKENS = re.compile(r"TOTAL|CARRIED\s*FORWARD|BROUGHT\s*FORWARD|\bC/F\b|\bB/F\b|CLOSING\s*BALANCE|OPENING\s*BALANCE|"
                          r"PAGE\s*\d|AVAILABLE\s*BALANCE|INTEREST\s*RATE|% ?P\.?A", re.I)
_HEADER_WORDS = {"TXN", "TRAN", "TRANSACTION", "DATE", "VALUE", "DESCRIPTION", "DEBIT", "CREDIT", "BALANCE",
                 "PARTICULARS", "CHQ", "CHEQUE", "NO", "NO.", "REF", "NARRATION", "WITHDRAWAL", "WITHDRAWALS",
                 "DEPOSIT", "DEPOSITS", "AMOUNT", "AMOUNT()", "BALANCE()", "REMARKS", "ID", "DETAILS", "TYPE",
                 "INIT.", "BR", "SR", "S.NO", "BRANCH", "CODE", "DR", "CR", "(INR)", "INR", "/", "&", "-", "OF"}
_FOOTER = re.compile(r"computer generated|page\s*(no\.?|\d+\s*of)|^page\b|this is a system generated|"
                     r"support\.|statement of account|^\d{1,3}$", re.I)


def _year(y: str) -> int:
    y = int(y)
    return y + 2000 if y < 100 else y


def line_date(line: str) -> dt.date | None:
    for pat, kind in _DATE_PATS:
        m = pat.match(line)
        if not m:
            continue
        a, b, c = m.groups()
        try:
            if kind == "ymd":
                return dt.date(int(a), int(b), int(c))
            if kind == "dmy":
                return dt.date(_year(c), int(b), int(a))
            if kind == "dMy":
                mon = _MONTHS.get(b[:3].upper())
                return dt.date(_year(c), mon, int(a)) if mon else None
            mon = _MONTHS.get(a[:3].upper())
            return dt.date(int(c), mon, int(b)) if mon else None
        except ValueError:
            return None
    return None


def money_tokens(line: str) -> list[dict]:
    if _SKIP_TOKENS.search(line):
        return []
    if re.fullmatch(r"\s*\.\d{2}\s*", line):          # Axis prints a balance under 1 rupee as '.11'
        return [{"value": float("0" + line.strip()), "mark": "", "fx": ""}]
    s = _DATE_PATS[0][0].sub(" ", line)
    for pat, _ in _DATE_PATS[1:]:
        s = pat.sub(" ", s)
    s = re.sub(r"\b\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4}\b", " ", s)      # dates inside the text
    out = []
    for m in _MONEY.finditer(s):
        raw = m.group("num").strip()
        if re.fullmatch(r"\d{1,3}(?:[.,]\d{2,3})+[.,]\d{2}", raw):
            num = re.sub(r"[.,]", "", raw[:-3]) + "." + raw[-2:]     # OCR read thousands commas as dots: 12.345.67
        else:
            num = raw.replace(",", "")
        if not m.group("cur") and "." not in num and "," not in raw:
            continue
        if float(num) >= 1e9:
            continue                                  # an account / card number, not an amount
        mark = (m.group("mark") or "").upper()
        out.append({"value": float(num), "mark": "DR" if "DR" in mark else ("CR" if "CR" in mark else ""),
                    "fx": m.group("fx") or ""})
    return out


# ------------------------------------------------------------------ segmentation
def _is_header(line: str) -> bool:
    words = re.findall(r"[^\s|]+", line.upper())
    return bool(words) and all(w.strip("():.") in _HEADER_WORDS or w in _HEADER_WORDS for w in words)


def opening_balance(lines: list[str]) -> float | None:
    for i, l in enumerate(lines):
        if re.search(r"OPENING\s*BALANCE|BALANCE\s*B/?F|BROUGHT\s*FORWARD", l, re.I):
            for cand in (re.sub(r"(?i).*?(OPENING\s*BALANCE|BALANCE\s*B/?F|BROUGHT\s*FORWARD)", "", l),
                         *(lines[i + 1:i + 3])):
                if line_date(cand):
                    break
                m = _MONEY.search(cand.replace(" ", " "))
                if m and ("." in m.group("num") or m.group("cur")):
                    v = float(m.group("num").replace(",", ""))
                    return -v if "DR" in (m.group("mark") or "").upper() else v
    return None


def segment(lines: list[str]) -> list[dict]:
    """Blocks that each hold one transaction: a dated line plus the lines after it."""
    segs: list[dict] = []
    cur: dict | None = None
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        d = line_date(line)
        if d and (cur is None or cur["tokens"]):
            cur = {"date": d, "lines": [], "tokens": []}
            segs.append(cur)
        elif d and not cur["tokens"] and not cur["lines"]:
            cur["date"] = d               # value date printed after the transaction date
        elif d and not cur["tokens"]:
            cur.update(date=d, lines=[])  # header dates (statement date, period) before the first row
        if cur is None:
            continue
        toks = money_tokens(line)
        cur["tokens"] += toks
        if not toks and not d and not _is_header(line) and not _FOOTER.search(line):
            cur["lines"].append(line)
        elif d:
            rest = line
            for pat, _ in _DATE_PATS:
                rest = pat.sub("", rest)
            rest = _MONEY.sub(" ", rest).strip(" -|")
            if rest and not line_date(rest) and not _is_header(rest):
                cur["lines"].append(rest)
        elif toks:
            rest = _MONEY.sub(" ", line).strip(" -|")
            if re.search(r"[A-Za-z]{3}", rest) and not _is_header(rest):
                cur["lines"].append(rest)
    return [s for s in segs if s["tokens"]]


_DR_WORDS = re.compile(r"\bDEBIT\b|/DR/|\bDR\b|\bWDL\b|\bWITHDRAWAL|\bTO\s+TRANSFER|\bPAID\s+TO\b|\bATM\b|\bATW\b|\bPOS\b|"
                       r"\bNACH\b|\bACH\s*-?\s*DR|\bECS\b|\bEMI\b|LOAN\s+RECOVERY|NEFTO|\bCHG|CHARGES", re.I)
_CR_WORDS = re.compile(r"\bCREDIT\b|/CR/|\bCR\b|\bDEP\b|\bDEPOSIT|\bBY\s+TRANSFER|\bRECEIVED\s+FROM\b|SALARY|"
                       r"\bINT\.?\s*PD|REFUND|CASH\s+DEP|\bCDM\b", re.I)


def _direction_hint(seg: dict, amount_tok: dict | None) -> int:
    if amount_tok and amount_tok["mark"]:
        return -1 if amount_tok["mark"] == "DR" else 1
    text = " ".join(seg["lines"])
    dr, cr = bool(_DR_WORDS.search(text)), bool(_CR_WORDS.search(text))
    if dr != cr:
        return -1 if dr else 1
    return 0


def build_transactions(lines: list[str]) -> tuple[list[dict], dict]:
    segs = segment(lines)
    info: dict[str, Any] = {"rows": len(segs)}
    if not segs:
        return [], info
    # a summary box dated far from the transactions (e.g. the print date) is not a row
    if len(segs) >= 5:
        ds = sorted(s["date"] for s in segs)
        clusters, curc = [], [ds[0]]
        for a, b in zip(ds, ds[1:]):
            if (b - a).days > 62:
                clusters.append(curc)
                curc = []
            curc.append(b)
        clusters.append(curc)
        main = max(clusters, key=len)
        segs = [s for s in segs if main[0] <= s["date"] <= main[-1]]
    has_balance = sum(1 for s in segs if len([t for t in s["tokens"] if t["value"] or True]) >= 2) >= 0.6 * len(segs)
    info["has_balance"] = has_balance
    # newest-first statements: walk them oldest-first so the balance chain runs forward
    dates = [s["date"] for s in segs]
    desc = sum(1 for a, b in zip(dates, dates[1:]) if b < a) > sum(1 for a, b in zip(dates, dates[1:]) if b > a)
    if desc:
        segs = segs[::-1]
    info["order"] = "newest first" if desc else "oldest first"
    prev = opening_balance(lines) if not desc else None
    info["opening_balance"] = prev
    txns = []
    def neighbour_amounts(i):
        """Amount cells of the rows just above / below (scans: a cell can land one row off)."""
        out = []
        for j in (i - 1, i + 1):
            if 0 <= j < len(segs):
                tk = segs[j]["tokens"]
                out += [t for t in (tk[:-1] if len(tk) >= 2 else tk) if t["value"] > 0]
        return out

    for i, s in enumerate(segs):
        toks = s["tokens"]
        bal_tok = toks[-1] if has_balance and len(toks) >= 2 else None
        bal = None
        if bal_tok:
            bal = -bal_tok["value"] if bal_tok["mark"] == "DR" and bal_tok is not None and len(toks) > 1 else bal_tok["value"]
        cands = [t for t in (toks[:-1] if bal_tok else toks[-1:]) if t["value"] > 0]
        amount, sign, check = None, 0, "n/a"
        if bal is not None and prev is not None and cands:
            delta = round(bal - prev, 2)
            hit = next((t for t in reversed(cands) if abs(abs(delta) - t["value"]) < 0.015), None)
            near = None if hit else next((t for t in neighbour_amounts(i) if abs(abs(delta) - t["value"]) < 0.015), None)
            if hit:
                amount, sign, check = hit["value"], (1 if delta > 0 else -1), "ok"
            elif near and abs(delta) >= 1:
                amount, sign, check = near["value"], (1 if delta > 0 else -1), "ok"
            else:
                tok = cands[-1]
                amount, sign, check = tok["value"], _direction_hint(s, tok), "mismatch"
        elif cands:
            tok = cands[-1]
            amount, sign = tok["value"], _direction_hint(s, tok)
        if amount is None:
            continue
        desc_text = re.sub(r"\s+", " ", " ".join(s["lines"][:4])).strip(" |-")
        txns.append({"date": s["date"].isoformat(), "description": desc_text[:160], "amount": amount,
                     "type": "credit" if sign > 0 else ("debit" if sign < 0 else "unknown"),
                     "balance": bal, "balance_check": check,
                     "currency": next((t["fx"] for t in toks if t["fx"]), "INR")})
        if bal is not None:
            prev = bal
        elif prev is not None and sign:
            prev = round(prev + sign * amount, 2)
    # rows outside the period of the verified rows are header / summary boxes, not transactions
    ok = [dt.date.fromisoformat(t["date"]) for t in txns if t["balance_check"] == "ok"]
    if len(ok) >= 5:
        lo, hi = min(ok) - dt.timedelta(days=31), max(ok) + dt.timedelta(days=31)
        dropped = [t for t in txns if t["balance_check"] != "ok" and not lo <= dt.date.fromisoformat(t["date"]) <= hi]
        txns = [t for t in txns if t not in dropped]
        info["dropped_outside_period"] = len(dropped)
    txns.sort(key=lambda t: t["date"])
    return txns, info


# ------------------------------------------------------------------ categories
CATEGORIES: list[tuple[str, str, str]] = [
    # (category, applies to: debit / credit / any, pattern)
    ("Bounce / return charges", "any", r"RTNCHG|RETURN|BOUNCE|INSUFF|DISHON|\bRTN\b|ECS\s*RTN|ACH\s*RTN"),
    ("Gambling / betting", "debit", r"DREAM\s*11|\bMPL\b|RUMMY|POKER|\bBET|1XBET|STAKE\.|PARIMATCH|WINZO|MY11CIRCLE|JUNGLEE"),
    ("Salary", "credit", r"SALARY|\bSAL\b|SAL\s*CR|PAYROLL|\bSAL/"),
    ("Interest", "credit", r"INT\.?\s*PD|INTEREST|\bINT\s*CR"),
    ("Refund", "credit", r"REFUND|REVERSAL|CASHBACK"),
    ("Cash deposit", "credit", r"\bCDM\b|CASH\s*DEP|BY\s*CASH|ATM\s*DEPOSIT|CASH\s*DEPOSIT"),
    ("EMI / loan", "debit", r"\bNACH\b|\bACH\b|\bECS\b|\bEMI\b|LOAN\s*(RECOVERY|ACCOUNT|REPAY)|\bSI\b|BAJAJ\s*FIN|PIRAMAL|"
                            r"HOUSING\s*FINANCE|LICHOUSING|SMFG|FIVE\s*STAR|HDB\s*FIN|TATA\s*CAP|MUTHOOT|MANAPPURAM|"
                            r"FULLERTON|IIFL|HOME\s*CREDIT|KREDITBEE|NAVI|MONEYVIEW|CREDILA|AVANSE"),
    ("Credit card", "debit", r"CREDIT\s*CARD|\bCC\s*PAY|\bCRED\b|CARD\s*BILL|BILLDESK.*CARD"),
    ("Insurance", "debit", r"\bLIC\b(?!\s*HOUSING)|INSURANCE|PREMIUM|HDFC\s*LIFE|SBI\s*LIFE|STAR\s*HEALTH|POLICYBAZAAR|ICICI\s*PRU"),
    ("Investments / savings", "debit", r"\bSIP\b|MUTUAL|ZERODHA|GROWW|UPSTOX|\bNSE\b|\bBSE\b|\bRD\b|\bFD\b|DEPOSIT\s*A/?C|PPF|NPS"),
    ("Bills / utilities", "debit", r"ELECTRIC|APCPDCL|TSSPDCL|TSNPDCL|APSPDCL|BESCOM|AIRTEL|\bJIO\b|VODAFONE|\bVI\b|BSNL|DTH|"
                                   r"TATA\s*PLAY|BROADBAND|\bGAS\b|WATER|RECHARGE|BILL\s*PAY|BBPS|ACT\s*FIBER"),
    ("Education", "debit", r"SCHOOL|COLLEGE|UNIVERSITY|ACADEMY|FEES|TUITION"),
    ("Shopping / food", "debit", r"ZOMATO|SWIGGY|BLINKIT|ZEPTO|AMAZON|FLIPKART|MYNTRA|BIGBASKET|DMART|AJIO|MEESHO|INSTAMART|"
                                 r"RESTAURANT|HOTEL|BAKERY|SUPERMARKET|MART\b|STORES"),
    ("Fuel / travel", "debit", r"PETROL|FUEL|HPCL|IOCL|BPCL|\bHP\s*PAY|IRCTC|UBER|OLA\b|RAPIDO|REDBUS|MAKEMYTRIP"),
    ("Medical", "debit", r"HOSPITAL|PHARMA|MEDICAL|MEDPLUS|APOLLO|CLINIC|DIAGNOSTIC"),
    ("Cash withdrawal", "debit", r"\bATM\b|\bATW\b|CASH\s*WDL|\bCWDR\b|\bNWD\b|CASH\s*WITHDRAWAL|SELF\s*CHQ|BY\s*SELF"),
    ("Bank charges", "debit", r"\bCHG|CHARGES|MIN\s*BAL|NON\s*MAINT|PENALTY|\bGST\b|SMS\s*CHG|AMC"),
]


def categorise(t: dict, holder_words: set[str]) -> str:
    d = t["description"].upper()
    words = set(re.findall(r"[A-Z]{3,}", d))
    if holder_words and len(words & holder_words) >= min(2, len(holder_words)) and re.search(r"NEFT|IMPS|RTGS|TRANSFER|UPI|NEFTO", d):
        return "Self transfer"
    for cat, side, pat in CATEGORIES:
        if side != "any" and side != t["type"]:
            continue
        if re.search(pat, d):
            return cat
    if t["type"] == "credit":
        return "Transfers in"
    if t["type"] == "debit":
        return "Transfers / UPI payments"
    return "Unknown"


_EXPENSE_EXCLUDE = {"EMI / loan", "Self transfer", "Investments / savings"}


def _key(desc: str) -> str:
    """Counterparty key: description without reference numbers."""
    words = [w for w in re.split(r"[\s/*:\-]+", desc.upper()) if w and not re.search(r"\d", w) and len(w) > 2
             and w not in {"UPI", "NEFT", "IMPS", "RTGS", "PAYMENT", "PAYMEN", "PAYME", "PAYM", "TRANSFER", "FROM",
                           "BANK", "LTD", "LIMITED", "INDIA", "STATE", "P2A", "P2M", "OF"}]
    return " ".join(words[:3])


# ------------------------------------------------------------------ analysis
def analyse(txns: list[dict], holder: str = "") -> dict[str, Any]:
    holder_words = {w for w in re.findall(r"[A-Z]{3,}", holder.upper()) if w not in {"MR", "MRS", "MS", "SMT", "SHRI"}}
    for t in txns:
        t["category"] = categorise(t, holder_words)
    # regular income: the same payer crediting >= 10,000 in 3+ different months
    credits_by_key = defaultdict(set)
    for t in txns:
        if t["type"] == "credit" and t["amount"] >= 10_000 and t["category"] == "Transfers in":
            credits_by_key[_key(t["description"])].add(t["date"][:7])
    regular = {k for k, months in credits_by_key.items() if k and len(months) >= 3}
    for t in txns:
        if t["type"] == "credit" and t["category"] == "Transfers in" and t["amount"] >= 10_000 and _key(t["description"]) in regular:
            t["category"] = "Regular income"

    months: dict[str, dict] = {}
    for t in txns:
        m = months.setdefault(t["date"][:7], {"month": t["date"][:7], "credits": 0.0, "debits": 0.0, "income": 0.0,
                                               "emi": 0.0, "expenses": 0.0, "by_category": Counter(),
                                               "balances": [], "transactions": 0})
        m["transactions"] += 1
        if t["type"] == "credit":
            m["credits"] += t["amount"]
            if t["category"] in ("Salary", "Regular income"):
                m["income"] += t["amount"]
        elif t["type"] == "debit":
            m["debits"] += t["amount"]
            if t["category"] == "EMI / loan":
                m["emi"] += t["amount"]
            elif t["category"] not in _EXPENSE_EXCLUDE:
                m["expenses"] += t["amount"]
            m["by_category"][t["category"]] += t["amount"]
        if t["balance"] is not None:
            m["balances"].append(t["balance"])
    month_rows = []
    for k in sorted(months):
        m = months[k]
        b = m.pop("balances")
        m["closing_balance"] = b[-1] if b else None
        m["min_balance"] = min(b) if b else None
        m["avg_balance"] = round(sum(b) / len(b), 2) if b else None
        m["by_category"] = {c: round(v, 2) for c, v in m["by_category"].most_common()}
        for f in ("credits", "debits", "income", "emi", "expenses"):
            m[f] = round(m[f], 2)
        month_rows.append(m)
    # averages over whole months only (drop a partial first / last month when there are enough)
    full = month_rows
    if len(month_rows) >= 4 and txns:
        first, last = dt.date.fromisoformat(txns[0]["date"]), dt.date.fromisoformat(txns[-1]["date"])
        full = [m for m in month_rows if not ((m["month"] == first.isoformat()[:7] and first.day > 7)
                                               or (m["month"] == last.isoformat()[:7] and last.day < 24))]
    n = max(1, len(full))
    avg = lambda f: round(sum(m[f] for m in full) / n, 2)
    income, expenses, emi = avg("income"), avg("expenses"), avg("emi")
    bals = [t["balance"] for t in txns if t["balance"] is not None]

    flags = []
    bounces = [t for t in txns if t["category"] == "Bounce / return charges"]
    if bounces:
        flags.append({"level": "high", "text": f"{len(bounces)} bounce / return entr{'y' if len(bounces) == 1 else 'ies'} "
                                               "(cheque or EMI returned)", "items": [_brief(t) for t in bounces[:6]]})
    gamb = [t for t in txns if t["category"] == "Gambling / betting"]
    if gamb:
        flags.append({"level": "high", "text": f"{len(gamb)} payments to betting / gaming apps",
                      "items": [_brief(t) for t in gamb[:6]]})
    credits = sorted((t for t in txns if t["type"] == "credit"), key=lambda t: -t["amount"])
    typical = max(income, statistics.median([m["credits"] for m in full]) if full else 0)
    big = [t for t in credits if t["amount"] >= max(100_000, 2 * typical) and t["category"] not in ("Salary", "Regular income")]
    if big and txns:
        end = dt.date.fromisoformat(txns[-1]["date"])
        recent = [t for t in big if (end - dt.date.fromisoformat(t["date"])).days <= 45]
        flags.append({"level": "medium" if recent else "low",
                      "text": f"{len(big)} unusually large credit{'s' if len(big) > 1 else ''}"
                              + (f", {len(recent)} in the last 45 days (check the source of funds)" if recent else ""),
                      "items": [_brief(t) for t in big[:6]]})
    cash_in = sum(t["amount"] for t in txns if t["category"] == "Cash deposit")
    tot_in = sum(t["amount"] for t in txns if t["type"] == "credit")
    if tot_in and cash_in / tot_in > 0.3:
        flags.append({"level": "medium", "text": f"{100 * cash_in / tot_in:.0f}% of the money coming in is cash deposits"})
    if income == 0 and txns:
        flags.append({"level": "medium", "text": "no salary or regular income found"})
    elif full:
        missing = [m["month"] for m in full if m["income"] == 0]
        if missing:
            flags.append({"level": "medium", "text": f"no salary / regular income in {', '.join(missing)}"})
    charges = [t for t in txns if t["category"] == "Bank charges" and re.search(r"MIN|NON\s*MAINT|PENALTY", t["description"], re.I)]
    if charges:
        flags.append({"level": "low", "text": f"{len(charges)} minimum-balance / penalty charges"})
    foir = round(emi / income, 2) if income else None
    if foir is not None and foir > 0.5:
        flags.append({"level": "high" if foir > 0.65 else "medium",
                      "text": f"existing EMIs take {foir * 100:.0f}% of the monthly income (FOIR)"})

    summary = {
        "months_covered": len(month_rows), "months_averaged": len(full),
        "avg_monthly_income": income, "avg_monthly_expenses": expenses, "avg_monthly_emi": emi,
        "avg_monthly_surplus": round(income - expenses - emi, 2),
        "avg_monthly_credits": avg("credits"), "avg_monthly_debits": avg("debits"),
        "foir": foir,
        "avg_balance": round(sum(bals) / len(bals), 2) if bals else None,
        "min_balance": min(bals) if bals else None,
        "bounces": len(bounces),
        "expense_breakdown": dict(Counter({c: round(sum(m["by_category"].get(c, 0) for m in full) / n, 2)
                                           for c in {c for m in full for c in m["by_category"]}
                                           if c not in _EXPENSE_EXCLUDE}).most_common()),
    }
    levels = [f["level"] for f in flags]
    summary["risk"] = "high" if "high" in levels else ("medium" if "medium" in levels else "low")
    return {"months": month_rows, "summary": summary, "flags": flags}


def _brief(t: dict) -> str:
    return f"{t['date']}  {'+' if t['type'] == 'credit' else '-'}{t['amount']:,.2f}  {t['description'][:60]}"


# ------------------------------------------------------------------ account details
_IFSC_BANK = {"SBIN": "State Bank of India", "UTIB": "Axis Bank", "UBIN": "Union Bank of India", "HDFC": "HDFC Bank",
              "ICIC": "ICICI Bank", "KKBK": "Kotak Mahindra Bank", "PUNB": "Punjab National Bank", "CNRB": "Canara Bank",
              "BARB": "Bank of Baroda", "IOBA": "Indian Overseas Bank", "IDIB": "Indian Bank", "IDFB": "IDFC FIRST Bank",
              "YESB": "Yes Bank", "FDRL": "Federal Bank", "BKID": "Bank of India", "CBIN": "Central Bank of India",
              "IBKL": "IDBI Bank", "KVBL": "Karur Vysya Bank", "TMBL": "Tamilnad Mercantile Bank", "CIUB": "City Union Bank",
              "SIBL": "South Indian Bank", "KARB": "Karnataka Bank", "MAHB": "Bank of Maharashtra", "UCBA": "UCO Bank",
              "PSIB": "Punjab & Sind Bank", "APGB": "Andhra Pragathi Grameena Bank", "APGV": "Andhra Pradesh Grameena Vikas Bank",
              "TGMB": "Telangana Grameena Bank", "AUBL": "AU Small Finance Bank", "ESFB": "Equitas Small Finance Bank",
              "IPOS": "India Post Payments Bank"}
_BANK_WORDS = [("PhonePe (UPI transaction history)", r"PHONEPE"), ("Google Pay (UPI history)", r"GOOGLE\s*PAY"),
               ("Paytm (UPI history)", r"PAYTM\s+STATEMENT|ONE97"), ("Al Rajhi Bank", r"AL\s*RAJHI|RJHI|SA\d{2}8000"),
               ("State Bank of India", r"STATE\s*BANK\s*OF\s*INDIA"), ("Axis Bank", r"AXIS\s*BANK|STATEMENT\s+OF\s+AXIS"),
               ("HDFC Bank", r"HDFC\s*BANK\s*(LTD|LIMITED)?\b(?!.*UPI)"), ("ICICI Bank", r"ICICI\s*BANK"),
               ("Union Bank of India", r"UNION\s*BANK"), ("Kotak Mahindra Bank", r"KOTAK\s*MAHINDRA"),
               ("Canara Bank", r"CANARA\s*BANK"), ("Bank of Baroda", r"BANK\s*OF\s*BARODA"),
               ("Punjab National Bank", r"PUNJAB\s*NATIONAL"), ("Indian Bank", r"\bINDIAN\s*BANK\b")]


def account_details(lines: list[str]) -> dict[str, str]:
    head = lines[:80]
    text = "\n".join(head)
    out = {"bank": "", "holder": "", "account_number": "", "ifsc": "", "currency": "INR"}
    ifsc = re.search(r"\b([A-Z]{4})0[A-Z0-9]{6}\b", text)
    rrb = re.search(r"([A-Z][A-Za-z ]{3,40}?GRAMEENA\s+(?:VIKAS\s+)?BANK)", text, re.I)
    if rrb:                                   # regional rural banks use their sponsor bank's IFSC
        out["bank"] = re.sub(r"\s+", " ", rrb.group(1)).title()
    if ifsc:
        out["ifsc"] = ifsc.group(0)
        out["bank"] = out["bank"] or _IFSC_BANK.get(ifsc.group(1), "")
    if not out["bank"]:
        up = "\n".join(lines[:200]).upper()
        out["bank"] = next((n for n, p in _BANK_WORDS if re.search(p, up)), "")
    pats = [r"(?:account\s*name|customer\s*name|name\s*of\s*(?:the\s*)?(?:account\s*holder|customer))\s*[:\-]?\s*(.*)",
            r"^statement\s+of\s+(?:mr\.?|mrs\.?|ms\.?)?\s*([A-Z][A-Za-z .]+?)\s*\(", r"^(?:welcome\s*:?|name)\s*[:\-]?\s*(.*)"]
    for p in pats:
        for i, l in enumerate(head):
            m = re.search(p, l.strip(), re.I)
            if m:
                v = m.group(1).strip(" :-")
                if not v and i + 1 < len(head):
                    v = head[i + 1].strip()
                v = re.sub(r"(?i)^(mr|mrs|ms|smt|shri)\.?\s+", "", re.sub(r"\s+", " ", v))
                v = re.sub(r"[؀-ۿﭐ-﷿ﹰ-﻿].*$", "", v)
                v = re.sub(r"(\s+-)+\s+", " ", v).strip(" -")
                if re.fullmatch(r"[A-Za-z .\-]{4,60}", v) and not re.search(
                        r"STATEMENT|ACCOUNT|BANK|BRANCH|ADDRESS|CUSTOMER|DETAILS|PERIOD", v, re.I):
                    out["holder"] = v.upper()
                    break
        if out["holder"]:
            break
    if not out["holder"]:            # name printed alone near the top (Axis: first line)
        for l in head[:6]:
            v = l.strip()
            if re.fullmatch(r"[A-Z][A-Z .]{3,40}", v) and len(v.split()) >= 2:
                out["holder"] = re.sub(r"\s+", " ", v)
                break
    acc = re.search(r"(?:account\s*(?:number|no\.?)|a/c\s*(?:no\.?)?)\s*[:\-]?\s*\n?\s*([X*\d][X*\d\- ]{6,24}\d)", text, re.I)
    if acc:
        a = re.sub(r"[\s\-]", "", acc.group(1))
        out["account_number"] = ("X" * (len(a) - 4) + a[-4:]) if len(a) > 4 else a        # masked
    fx = re.search(r"\b(SAR|AED|USD|GBP|EUR|QAR|KWD|OMR)\b", text)
    if fx:
        out["currency"] = fx.group(1)
    return out


# ------------------------------------------------------------------ text sources
def lines_from_pdf(path, password: str = "") -> tuple[list[str] | None, str]:
    """(lines, status). status: 'ok' | 'password' | 'scan' | 'error'."""
    try:
        import fitz
        doc = fitz.open(str(path))
        if doc.needs_pass and not doc.authenticate(password or ""):
            return None, "password"
        pages = [p.get_text("text") for p in doc]
    except ImportError:
        try:
            import pypdfium2 as pdfium
            pdf = pdfium.PdfDocument(str(path), password=password or None)
            pages = [pg.get_textpage().get_text_range() for pg in pdf]
        except Exception as e:
            return None, "password" if "password" in str(e).lower() else "error"
    except Exception:
        return None, "error"
    text = "\n".join(pages)
    if len(re.sub(r"\s", "", text)) < 200 * max(1, len(pages)) * 0.2:
        return None, "scan"
    return text.splitlines(), "ok"


def _date_rows(lines: list, width: float) -> list[str] | None:
    """Scanned tables: each transaction row is anchored on its date (left side of the
    page); every other cell joins the date line nearest to it after removing the page
    tilt. Plain row grouping fails here: dense rows, and amount columns printed a
    little lower than the dates. None when the page has too few date rows."""
    # stamps / footers printed across a row ('** End of Statement **') are not cells of it
    lines = [l for l in lines if not re.search(r"END\s*OF\s*STATEMENT|PAID\s*UP\s*CAPITAL", l.text, re.I)]
    anchors = [l for l in lines if line_date(l.text) and l.x0 < 0.4 * width]
    if len(anchors) < 4:
        return None
    anchors.sort(key=lambda l: l.cy)
    gaps = [b.cy - a.cy for a, b in zip(anchors, anchors[1:]) if b.cy - a.cy > 0.3 * a.h]
    pitch = statistics.median(gaps) if gaps else anchors[0].h * 1.5
    cells = [l for l in lines if l not in anchors and money_tokens(l.text)]

    def ycorr(l, k):
        return l.cy - k * ((l.x0 + l.x1) / 2)

    def score(k):
        ay = [ycorr(a, k) for a in anchors]
        return sum(1 for c in cells if min(abs(ycorr(c, k) - y) for y in ay) < 0.25 * pitch)
    k = max((score(s / 1000), -abs(s), s / 1000) for s in range(-40, 41, 2))[2]
    ay = [ycorr(a, k) for a in anchors]
    rows: list[list] = [[a] for a in anchors]
    head, tail = [], []
    for l in lines:
        if l in anchors:
            continue
        y = ycorr(l, k)
        # a cell belongs to the closest date row at or above it (amounts sit slightly lower)
        best = min(range(len(ay)), key=lambda i: abs(y - ay[i] - 0.25 * pitch))
        if y < ay[0] - 0.6 * pitch:
            head.append(l)
        elif y > ay[-1] + 1.5 * pitch:
            tail.append(l)
        else:
            rows[best].append(l)
    from .ssc import _rows
    out = [r.text for r in _rows(head)]
    out += [" ".join([r[0].text] + [c.text for c in sorted(r[1:], key=lambda c: c.x0)]) for r in rows]
    out += [r.text for r in _rows(tail)]
    return out


def lines_from_ocr(ocr_pages: list[dict]) -> list[str]:
    from .common import lines_of
    from .ssc import _rows
    out = []
    for p in ocr_pages:
        ls = lines_of([p])
        rows = _date_rows(ls, p.get("width") or max((l.x1 for l in ls), default=1))
        out += rows if rows is not None else [r.text for r in _rows(ls)]
    return out


# ------------------------------------------------------------------ entry points
def extract_statement_lines(lines: list[str], source: str) -> dict[str, Any]:
    acct = account_details(lines)
    txns, info = build_transactions(lines)
    checked = [t for t in txns if t["balance_check"] in ("ok", "mismatch")]
    ok = sum(1 for t in checked if t["balance_check"] == "ok")
    analysis = analyse(txns, acct["holder"]) if txns else {"months": [], "summary": {}, "flags": []}
    flags = analysis["flags"]
    rec = round(100 * ok / len(checked), 1) if checked else None
    if checked and ok < len(checked):
        bad = [t for t in checked if t["balance_check"] == "mismatch"]
        flags.insert(0, {"level": "high" if len(bad) > max(2, 0.02 * len(checked)) else "medium",
                         "text": f"{len(bad)} of {len(checked)} rows do not add up to the printed balance "
                                 "(misread row, or the statement may have been edited)",
                         "items": [_brief(t) for t in bad[:6]]})
    unknown = [t for t in txns if t["type"] == "unknown"]
    if unknown:
        flags.append({"level": "low", "text": f"{len(unknown)} transaction(s) could not be classed as debit or credit",
                      "items": [_brief(t) for t in unknown[:4]]})
    if analysis["summary"]:
        levels = [f["level"] for f in flags]
        analysis["summary"]["risk"] = "high" if "high" in levels else ("medium" if "medium" in levels else "low")
    status = "manual_required" if not txns else ("review_recommended" if flags else "auto")
    return {
        "document_type": "bank_statement" if txns else "unknown",
        "source": source, "account": acct,
        "period": {"from": min(t["date"] for t in txns), "to": max(t["date"] for t in txns)} if txns else {},
        "checks": {"transactions": len(txns), "balance_checked": len(checked), "balance_ok": ok,
                   "balance_reconciled_percent": rec, "has_running_balance": info.get("has_balance"),
                   "order": info.get("order"), "opening_balance": info.get("opening_balance")},
        "summary": analysis["summary"], "months": analysis["months"], "flags": flags,
        "transactions": txns,
        "review": {"status": status, "fields_requiring_review": [f"flag{i}" for i in range(len(flags))],
                   "reasons": {f"flag{i}": f["text"] for i, f in enumerate(flags)}},
        "notes": "" if txns else "No transactions could be read from this file.",
    }


MAX_OCR_PAGES = 60        # a scanned 6-month statement is rarely longer; OCR is ~3-9 s per page


def _decrypted_copy(path, password: str):
    """Unlocked copy next to the upload (same temporary folder, deleted with it)."""
    from pathlib import Path
    out = Path(path).with_name(Path(path).stem + "_unlocked.pdf")
    try:
        import fitz
        doc = fitz.open(str(path))
        if doc.needs_pass and doc.authenticate(password):
            doc.save(str(out))
            return out
    except ImportError:
        import pypdfium2 as pdfium
        pdfium.PdfDocument(str(path), password=password).save(str(out))
        return out
    except Exception:
        pass
    return path


def extract_statement_files(paths: list, images_of, ocr_engine, password: str = "") -> dict[str, Any]:
    """images_of(path) renders a file to page images (used for scans and photos)."""
    from .common import ocr_upright
    all_lines: list[str] = []
    sources = set()
    for p in paths:
        lines, status = (lines_from_pdf(p, password) if str(p).lower().endswith(".pdf") else (None, "scan"))
        if status == "password":
            return {"document_type": "password_protected", "account": {}, "summary": {}, "months": [], "flags": [],
                    "transactions": [], "checks": {}, "period": {},
                    "review": {"status": "manual_required", "fields_requiring_review": ["password"],
                               "reasons": {"password": "this PDF is password protected"}},
                    "notes": "This PDF is password protected. Enter the PDF password and try again."
                             + (" The password given was not accepted." if password else "")}
        if lines:
            sources.add("PDF text")
        else:
            if password and str(p).lower().endswith(".pdf"):
                p = _decrypted_copy(p, password)        # scanned + password protected: render an unlocked copy
            pages = [ocr_upright(img, ocr_engine, i + 1, ["BALANCE", "DEBIT", "CREDIT", "DATE", "TRANSACTION"])
                     for i, img in enumerate(images_of(p)[:MAX_OCR_PAGES])]
            lines = lines_from_ocr(pages)
            sources.add("OCR")
        all_lines += lines
    r = extract_statement_lines(all_lines, " + ".join(sorted(sources)))
    r["pages"] = len(paths)
    return r


def flat(result: dict) -> dict:
    s = result.get("summary", {})
    return {"document_type": result["document_type"], **{f"account_{k}": v for k, v in result.get("account", {}).items()},
            "period_from": result.get("period", {}).get("from"), "period_to": result.get("period", {}).get("to"),
            **s, "months": [{k: v for k, v in m.items() if k != "by_category"} for m in result.get("months", [])],
            "flags": [f["text"] for f in result.get("flags", [])]}
