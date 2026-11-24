"""Education-loan sanction letter extraction.

Sanction letters come from many lenders (Credila, Avanse, Auxilo, InCred, SBI,
Union Bank, ...) and each words its summary differently: 'Processing Fee' vs
'Origination Fees', 'Tenure' vs 'Repayment Term'. Each field therefore has a
list of label variants; the value is the rest of the label's line, or the next
line(s) when the label wraps ('Origination Fees (Non-' / 'Refundable)').

Most letters are system-generated PDFs with a text layer: their text is read
directly (exact, no OCR). Scans and photos are OCR'd and rebuilt into rows.

Checks (flag only): amount in figures vs words ('Fifty Lakh'), instalment
periods adding up to the tenure, plausible interest rate, fee as % of amount.
Contact details and bank account numbers in the letter are not extracted.
"""
from __future__ import annotations

import re
from typing import Any

from .common import field, finish

LENDERS = [
    ("Credila Financial Services Limited", r"CREDILA"),
    ("Avanse Financial Services", r"AVANSE"),
    ("Auxilo Finserve", r"AUXILO"),
    ("InCred Finance", r"INCRED"),
    ("Tata Capital", r"TATA\s*CAPITAL"),
    ("Poonawalla Fincorp", r"POONAWALLA"),
    ("Propelld", r"PROPELLD"),
    ("Prodigy Finance", r"PRODIGY\s*FINANCE"),
    ("MPOWER Financing", r"MPOWER"),
    ("Leap Finance", r"LEAP\s*FINANCE"),
    ("State Bank of India", r"STATE\s*BANK\s*OF\s*INDIA|\bSBI\b"),
    ("Union Bank of India", r"UNION\s*BANK\s*OF\s*INDIA"),
    ("Bank of Baroda", r"BANK\s*OF\s*BARODA"),
    ("Bank of India", r"(?<!UNION )(?<!STATE )BANK\s*OF\s*INDIA"),
    ("Punjab National Bank", r"PUNJAB\s*NATIONAL\s*BANK"),
    ("Canara Bank", r"CANARA\s*BANK"),
    ("Indian Bank", r"\bINDIAN\s*BANK\b"),
    ("Central Bank of India", r"CENTRAL\s*BANK\s*OF\s*INDIA"),
    ("Axis Bank", r"AXIS\s*BANK"),
    ("ICICI Bank", r"ICICI\s*BANK"),
    ("IDFC FIRST Bank", r"IDFC\s*FIRST"),
    ("HDFC Bank", r"HDFC\s*BANK"),
    ("Yes Bank", r"\bYES\s*BANK"),
    ("Kotak Mahindra Bank", r"KOTAK"),
]

# label variants per field, matched at the start of a line (after any '1.' numbering)
LABELS: dict[str, str] = {
    "applicant_name": r"(?:name\s+of\s+(?:the\s+)?(?:applicant|student|borrower)|applicant'?s?\s+name|student'?s?\s+name|borrower'?s?\s+name|applicant|borrower|student)(?!\s*\()",
    "co_applicants": r"co[\s-]*(?:applicant|borrower)s?(?:\s*\(s\))?(?:\s*name\(?s?\)?)?|guarantor(?:\s*\(s\)|s)?",
    "country": r"(?:country\s+of\s+(?:study|education)|study\s+country|destination(?:\s+country)?|country)",
    "course": r"(?:name\s+of\s+(?:the\s+)?(?:course|program(?:me)?)|course(?:\s+(?:details|name|of\s+study))?|program(?:me)?(?:\s+name)?)",
    "university": r"(?:name\s+of\s+(?:the\s+)?(?:university|institute|institution|college)(?:\s*/\s*[a-z]+)*(?:\s*&\s*location)?|university(?:\s+name)?|institute(?:\s+name)?|institution|college(?:\s+name)?)",
    "course_university": r"(?:details\s+of\s+(?:the\s+)?)?(?:course|program(?:me)?)\s*(?:&|and)\s*(?:university|institute|college)",
    "loan_amount": r"(?:sanctioned\s+(?:education\s+)?loan(?:\s+amount)?|sanctioned\s+amount|loan\s+amount(?:\s+sanctioned)?|amount\s+(?:of\s+loan\s+)?sanctioned|sanction(?:ed)?\s+limit|total\s+loan\s+amount|loan\s+sanctioned|limit\s+sanctioned)",
    "roi": r"(?:rate\s+of\s+interest|interest\s+rate|applicable\s+(?:rate\s+of\s+)?interest|roi)(?:\s*\(roi\))?",
    "processing_fee": r"(?:(?:one[\s-]*time\s+)?origination\s+fees?|processing\s+(?:fees?|charges?)|login\s+fees?|upfront\s+fees?|administrative\s+fees?|pf)",
    "tenure": r"(?:repayment\s+(?:term|tenure|tenor|period)|loan\s+(?:tenure|tenor|term)|tenure|tenor)",
    "moratorium": r"(?:moratorium(?:\s+period)?|holiday\s+period)",
    "sanction_date": r"(?:date\s+of\s+sanction|sanction\s+date)",
    "loan_type": r"loan\s+type|type\s+of\s+loan|nature\s+of\s+security",
}
_NUM = r"\d[\d,]*(?:\.\d+)?"
_MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"
_MON = {m[:3].upper(): i + 1 for i, m in enumerate(_MONTHS.split("|"))}

WORDS = {"ZERO": 0, "ONE": 1, "TWO": 2, "THREE": 3, "FOUR": 4, "FIVE": 5, "SIX": 6, "SEVEN": 7, "EIGHT": 8,
         "NINE": 9, "TEN": 10, "ELEVEN": 11, "TWELVE": 12, "THIRTEEN": 13, "FOURTEEN": 14, "FIFTEEN": 15,
         "SIXTEEN": 16, "SEVENTEEN": 17, "EIGHTEEN": 18, "NINETEEN": 19, "TWENTY": 20, "THIRTY": 30,
         "FORTY": 40, "FIFTY": 50, "SIXTY": 60, "SEVENTY": 70, "EIGHTY": 80, "NINETY": 90}
SCALES = {"HUNDRED": 100, "THOUSAND": 1_000, "LAKH": 100_000, "LAKHS": 100_000, "LAC": 100_000,
          "LACS": 100_000, "CRORE": 10_000_000, "CRORES": 10_000_000}


def words_to_number(text: str) -> int | None:
    """'Fifty Lakh Rupees' -> 5000000; 'One Crore Twenty Lakh' -> 12000000."""
    total = cur = 0
    seen = False
    for w in re.findall(r"[A-Z]+", text.upper().replace("-", " ")):
        if w in WORDS:
            cur += WORDS[w]
            seen = True
        elif w == "HUNDRED":
            cur = max(cur, 1) * 100
        elif w in SCALES:
            total += max(cur, 1) * SCALES[w]
            cur = 0
            seen = True
        elif w in ("AND", "RUPEES", "RS", "INR", "ONLY"):
            continue
        elif seen:
            break
    return total + cur if seen else None


def parse_amount(text: str) -> tuple[float | None, str]:
    """First rupee amount in the text -> (rupees, currency shown)."""
    t = text.replace("₹", " INR ").replace("`", " INR ")
    m = re.search(rf"(INR|RS\.?|RUPEES)\s*({_NUM})\s*(LAKHS?|LACS?|CRORES?)?", t, re.I) or \
        re.search(rf"({_NUM})\s*(LAKHS?|LACS?|CRORES?)", t, re.I)
    if not m:
        m2 = re.search(rf"\b({_NUM})\s*/-", t)
        if not m2:
            return None, ""
        return float(m2.group(1).replace(",", "")), "INR"
    if m.re.groups == 3:
        num, scale = m.group(2), m.group(3)
    else:
        num, scale = m.group(1), m.group(2)
    v = float(num.replace(",", ""))
    if scale:
        v *= SCALES[scale.upper()]
    return v, "INR"


def _date(text: str) -> str:
    iso = re.search(r"\b(20\d{2})-(\d{2})-(\d{2})\b", text)          # 'Date: 2026-06-04'
    if iso:
        return f"{iso.group(1)}-{iso.group(2)}-{iso.group(3)}"
    m = re.search(rf"(\d{{1,2}})(?:st|nd|rd|th)?\s*[-\s]?\s*({_MONTHS[:3]}|{_MONTHS})[a-z]*\.?,?\s*[-\s]?\s*(\d{{4}})", text, re.I)
    if m:
        mon = _MON.get(m.group(2)[:3].upper())
        if mon:
            return f"{m.group(3)}-{mon:02d}-{int(m.group(1)):02d}"
    m = re.search(r"(\d{1,2})[/.-](\d{1,2})[/.-](\d{4})", text)
    if m:
        return f"{m.group(3)}-{int(m.group(2)):02d}-{int(m.group(1)):02d}"
    return ""


def _clean(v: str) -> str:
    return re.sub(r"\s+", " ", v).strip(" :;,.-*").strip()


# what a usable value must contain (else keep looking on the next lines)
VALID = {
    "loan_amount": lambda v: parse_amount(v)[0] is not None,
    "processing_fee": lambda v: parse_amount(v)[0] is not None or re.search(r"\d\s*%", v) is not None,
    "roi": lambda v: re.search(r"\d\s*%", v) is not None,
    "moratorium": lambda v: re.search(r"\d+\s*months?", v, re.I) is not None,
}
# rest of a label that wrapped onto its own line ('Amount', '(Non-refundable)', '/College/Institution & Location')
_REMNANT = re.compile(r"^(?:amount|\(?non-?\s*refundable\)?|refundable\)|[/&].*|in\s+(?:rs\.?|inr)\W*)$", re.I)


def _find(lines: list[str], key: str, limit: int | None = None) -> tuple[str, int] | None:
    """Value for a label: rest of its line, else the following line(s) - labels wrap and
    many letters print the value on the line after its label."""
    pat = re.compile(rf"^\s*(?:\d+\.\s*)?(?:{LABELS[key]})(?![A-Za-z])\s*[:\-–@]?\s*(.*)$", re.I)
    valid = VALID.get(key)

    def wrapped(v: str) -> bool:      # label continues on the next line: '(Non-' / 'Refundable)'
        # a word broken at a hyphen, not the '/-' that ends Indian rupee amounts ('Rs. 3200000/-')
        return bool(re.search(r"[A-Za-z]-$", v)) or (v.startswith("(") and ")" not in v)

    def unusable(v: str) -> bool:
        return not v or wrapped(v) or bool(_REMNANT.match(v)) or (valid is not None and not valid(v))

    for i, line in enumerate(lines[:limit]):
        m = pat.match(line)
        if not m:
            continue
        v = re.sub(r"\s+", " ", m.group(1)).strip(" :;,*").strip()
        j = i
        while unusable(v) and j + 1 < len(lines) and j - i < 3:
            j += 1
            nxt = re.sub(r"\s+", " ", lines[j]).strip(" :;,*")
            if wrapped(v) and re.fullmatch(r"[A-Za-z\- ]{0,20}\)", nxt):
                v = ""                # 'Refundable)' closes the label; the value is on the next line
                continue
            v = nxt
        v = _clean(v)
        if v and not unusable(v):
            return v, i
    return None


# ---------------- university / country helpers
_UNI = re.compile(r"(University\s+of\s+(?:the\s+)?[A-Z][A-Za-z\-]+(?:\s+(?!Nature|Collateral|Country|Purpose|Sanction|Interest|MSC|MS|MA|MBA|BSC|BA)[A-Z][A-Za-z\-]+)?"
                  r"|(?:[A-Z][A-Za-z.&'\-]+\s+){1,3}University(?:\s+of\s+[A-Z][a-z]+)?"
                  r"|(?:[A-Z][A-Za-z.&'\-]+\s+){1,3}College(?:\s+of\s+[A-Z][a-z]+)?)", re.I)
COUNTRY_WORDS = [("UK", r"\bUNITED\s+KINGDOM\b|\bU\.?K\.?\b|\bENGLAND\b|\bSCOTLAND\b|\bWALES\b"),
                 ("USA", r"\bUNITED\s+STATES\b|\bU\.?S\.?A\.?\b"), ("Canada", r"\bCANADA\b"),
                 ("Australia", r"\bAUSTRALIA\b"), ("Ireland", r"\bIRELAND\b"), ("Germany", r"\bGERMANY\b"),
                 ("New Zealand", r"\bNEW\s+ZEALAND\b"), ("France", r"\bFRANCE\b"), ("Singapore", r"\bSINGAPORE\b")]
CURRENCY_COUNTRY = {"GBP": "UK", "CAD": "Canada", "AUD": "Australia", "NZD": "New Zealand", "SGD": "Singapore"}
# well-known destinations of Indian students; used only when the letter does not print the country
UNI_COUNTRY = [
    ("USA", r"NEW\s+YORK|STATE\s+UNIVERSITY|NORTHEASTERN|BOSTON|PURDUE|CLEMSON|TEXAS|CALIFORNIA|ILLINOIS|ARIZONA|"
            r"FLORIDA|GEORGIA|MICHIGAN|OHIO|PENNSYLVANIA|NORTH\s+CAROLINA|VIRGINIA|COLUMBIA|STEVENS|DREXEL|"
            r"RUTGERS|SYRACUSE|CARNEGIE|STANFORD|HARVARD|MIT\b|PACE\s+UNIVERSITY|DEPAUL|KENT\s+STATE"),
    ("UK", r"COVENTRY|NORTHUMBRIA|HERTFORDSHIRE|CHESTER|GREENWICH|LEICESTER|BIRMINGHAM|MANCHESTER|LIVERPOOL|LEEDS|"
           r"SHEFFIELD|NOTTINGHAM|GLASGOW|EDINBURGH|ABERDEEN|DUNDEE|STRATHCLYDE|ULSTER|SUNDERLAND|TEESSIDE|"
           r"BEDFORDSHIRE|WOLVERHAMPTON|PORTSMOUTH|WESTMINSTER|MIDDLESEX|UNIVERSITY\s+OF\s+KENT|ESSEX|SUSSEX|"
           r"SURREY|BRUNEL|KINGSTON|ROEHAMPTON|BRISTOL|CARDIFF|SWANSEA|SOUTHAMPTON|SALFORD|HUDDERSFIELD|"
           r"BRADFORD|DERBY|ANGLIA\s+RUSKIN|DE\s+MONTFORT|NORTHAMPTON|PLYMOUTH|EXETER|DURHAM|WARWICK|"
           r"LOUGHBOROUGH|ASTON|NEWCASTLE|LANCASTER|KEELE|STAFFORDSHIRE|UNIVERSITY\s+OF\s+HULL|ROBERT\s+GORDON|"
           r"HERIOT|STIRLING|EAST\s+LONDON|WEST\s+LONDON|LONDON|CRANFIELD|SOLENT|QUEEN\s+MARY|IMPERIAL|"
           r"OXFORD|CAMBRIDGE|BANGOR|WEST\s+OF\s+SCOTLAND|LIVERPOOL\s+JOHN\s+MOORES|BATH\b|READING"),
    ("Canada", r"TORONTO|WATERLOO|MCGILL|CONCORDIA|OTTAWA|ALBERTA|BRITISH\s+COLUMBIA|DALHOUSIE|CONESTOGA|HUMBER|"
               r"SENECA|CENTENNIAL|LAMBTON|FANSHAWE|ALGONQUIN|SHERIDAN|GEORGE\s+BROWN|NIAGARA\s+COLLEGE|WINDSOR|"
               r"LAKEHEAD|YORK\s+UNIVERSITY|MANITOBA|SASKATCHEWAN|CAPE\s+BRETON"),
    ("Australia", r"MONASH|MELBOURNE|SYDNEY|DEAKIN|RMIT|QUEENSLAND|ADELAIDE|GRIFFITH\s+UNIVERSITY|MACQUARIE|"
                  r"LA\s+TROBE|SWINBURNE|CURTIN|WOLLONGONG|WESTERN\s+SYDNEY|FEDERATION\s+UNIVERSITY"),
    ("Ireland", r"TRINITY\s+COLLEGE\s+DUBLIN|DUBLIN|MAYNOOTH|GALWAY|UNIVERSITY\s+COLLEGE\s+CORK|LIMERICK|GRIFFITH\s+COLLEGE"),
    ("Germany", r"TECHNICAL\s+UNIVERSITY\s+OF\s+MUNICH|MUNICH|BERLIN|HAMBURG|AACHEN|IU\s+INTERNATIONAL"),
    ("New Zealand", r"AUCKLAND|OTAGO|WAIKATO|MASSEY|UNIVERSITY\s+OF\s+CANTERBURY"),
]


def _infer_country(text: str, university: str) -> tuple[str, str]:
    """Country when the letter has no 'Country of Study': wording, currency, then university name."""
    up = text.upper()
    for c, pat in COUNTRY_WORDS:
        if re.search(pat, up):
            return c, "mentioned in the letter"
    cur = re.search(r"\b(GBP|CAD|AUD|NZD|SGD)\b", up)
    if cur:
        return CURRENCY_COUNTRY[cur.group(1)], f"from the {cur.group(1)} amount"
    for c, pat in UNI_COUNTRY:
        if university and re.search(pat, university.upper()):
            return c, "from the university name"
    return "", ""


def _lender(text: str) -> str:
    up = text.upper()
    m = re.search(r"\bFOR\s+([A-Z][A-Za-z&.\s]{3,60}?(?:LIMITED|LTD\.?|BANK|FINANCE|FINANCING))", up)
    for name, pat in LENDERS:
        if re.search(pat, up):
            return name
    return m.group(1).title() if m else ""


def extract_sanction_text(lines: list[str], source: str) -> dict[str, Any]:
    lines = [l for l in (re.sub(r"\s+", " ", x).strip() for x in lines) if l]
    text = "\n".join(lines)
    up = text.upper()
    fields: dict[str, dict] = {}
    flags: dict[str, str] = {}
    checks: dict[str, Any] = {"text_source": source}

    if not re.search(r"SANCTION|LOAN", up):
        note = "This does not look like a loan sanction letter."
        fields["lender"] = field("", "not found")
        r = finish("not_sanction", fields, {"document": note}, [], checks, note)
        r["review"]["status"] = "manual_required"
        return r

    # summary block = before the general terms and conditions
    stop = next((i for i, l in enumerate(lines) if re.search(r"general\s+terms|terms\s+and\s+conditions\s*$", l, re.I)), None)

    def get(key, limit=stop):
        return _find(lines, key, limit) or (_find(lines, key) if limit else None)

    lender = _lender(text)
    fields["lender"] = field(lender, "printed text" if lender else "not found")

    for key in ("university", "country", "course", "applicant_name", "co_applicants"):
        hit = get(key)
        v = hit[0] if hit else ""
        if key == "university" and re.fullmatch(r"(?i)university", v):
            v = ""
        fields[key] = field(v, "printed text" if v else "not found", raw=lines[hit[1]] if hit else None)

    # 'Details of Course & University : COVENTRY UNIVERSITY MSC ...' (value may wrap to the next line)
    hit = _find(lines, "course_university", stop)
    if hit and not (fields["university"]["value"] and fields["course"]["value"]):
        v, i = hit
        if i + 1 < len(lines) and not re.match(r"^\s*\d+\.|.*:\s", lines[i + 1]) and len(lines[i + 1]) < 80:
            v = f"{v} {lines[i + 1].strip()}"
        um = _UNI.search(v)
        if um:
            uni = _clean(um.group(0))
            course = _clean((v[:um.start()] + " " + v[um.end():]).strip(" ,-&"))
            if not fields["university"]["value"]:
                fields["university"] = field(uni, "printed text ('Course & University')", raw=lines[i])
            if course and not fields["course"]["value"]:
                fields["course"] = field(course, "printed text ('Course & University')", raw=lines[i])
    # 'Purpose: For pursuing the <course> program from <University>' (Credila visa letters)
    if not (fields["university"]["value"] and fields["course"]["value"]):
        flat_text = re.sub(r"\s+", " ", " ".join(lines[:stop]))
        pm = re.search(r"pursuing\s+(?:the\s+)?(.+?)\s+(?:program(?:me)?|course)?\s*(?:from|at)\s+(.{0,120})", flat_text, re.I)
        if pm:
            um = _UNI.match(pm.group(2)) or _UNI.search(pm.group(2))
            if um and not fields["university"]["value"]:
                fields["university"] = field(_clean(um.group(0)), "printed text ('pursuing ... from')")
            if not fields["course"]["value"]:
                fields["course"] = field(_clean(pm.group(1)), "printed text ('pursuing ... from')")
    if not fields["country"]["value"]:
        c, how = _infer_country(text, fields["university"]["value"])
        if c:
            fields["country"] = field(c, f"inferred: {how}")
            if how == "from the university name":
                flags["country"] = "the letter does not print the country - inferred from the university name, please check"

    m = re.search(r"application\s*(?:number|no\.?)\s*[:\-]?\s*([A-Z0-9/-]{5,})", text, re.I) or \
        re.search(r"(?:loan\s*account|loan\s*a/?c|reference|ref\.?)\s*(?:number|no\.?)\s*[:\-]?\s*([A-Z0-9/-]{5,})", text, re.I) or \
        re.search(r"\bLAN\s*:\s*([A-Z0-9/-]{5,})", text)
    fields["application_number"] = field(m.group(1) if m else "", "printed text" if m else "not found")

    # ---- amount (figures, words cross-check, foreign-currency equivalent)
    hit = get("loan_amount")
    amt_raw = hit[0] if hit else ""
    amt, _ = parse_amount(amt_raw)
    fields["loan_amount"] = field(amt, "printed text" if amt else "not found", raw=amt_raw or None)
    wm = re.search(r"\(([A-Za-z\s-]+?)(?:Rupees|Only|equivalent|\))", amt_raw)
    if amt and not wm:            # e.g. 'Rs. 3200000/- (INR Thirty Two lakh only)' in the opening sentence
        for l in lines:
            if parse_amount(l)[0] == amt:
                wm = re.search(r"\(([A-Za-z\s-]{6,}?)(?:Rupees|Only|equivalent|\))", l, re.I)
                if wm:
                    break
    if amt and wm:
        w = words_to_number(wm.group(1))
        if w:
            ok = abs(w - amt) < 1
            checks["amount_words_match"] = ok
            fields["loan_amount"]["verified"] = ok
            if not ok:
                flags["loan_amount"] = f"figures {amt:,.0f} but words say {w:,.0f}"
    fx = re.search(r"equivalent\s+to\s+([A-Z]{3})\s*(" + _NUM + ")", amt_raw, re.I)
    fields["loan_amount_foreign"] = field(f"{fx.group(1).upper()} {fx.group(2)}" if fx else "",
                                          "printed text" if fx else "not printed")
    if not amt:
        flags["loan_amount"] = "loan amount not found"

    # ---- rate of interest
    hit = get("roi")
    roi_raw = hit[0] if hit else ""
    rm = re.search(r"(\d{1,2}(?:\.\d{1,3})?)\s*%", roi_raw)
    roi = float(rm.group(1)) if rm else None
    rtype = "Floating" if re.search(r"floating|variable|linked|benchmark|repo|rllr|mclr|eblr", roi_raw, re.I) else \
        ("Fixed" if re.search(r"fixed", roi_raw, re.I) else "")
    fields["roi"] = field(roi, "printed text" if roi is not None else "not found", raw=roi_raw or None)
    fields["roi_type"] = field(rtype, "printed text" if rtype else "not printed")
    if roi is None:
        flags["roi"] = "interest rate not found"
    elif not 4 <= roi <= 20:
        flags["roi"] = f"{roi}% is outside the usual education-loan range - please check"

    # ---- processing fee / origination fee
    hit = get("processing_fee")
    # drop the wrapped label's '(Non-' / 'Refundable)' when OCR put it on the value's row
    pf_raw = re.sub(r"\(?\s*Non-?\s*(?:Refundable\s*\)?)?", "", hit[0], flags=re.I).strip() if hit else ""
    pf_amt, _ = parse_amount(pf_raw)
    pm = re.search(r"(\d{1,2}(?:\.\d{1,2})?)\s*%", pf_raw)
    pf_pct = float(pm.group(1)) if pm else (round(100 * pf_amt / amt, 2) if (pf_amt and amt) else None)
    fields["processing_fee"] = field(pf_amt if pf_amt is not None else (f"{pm.group(1)}%" if pm else ""),
                                     "printed text" if (pf_amt is not None or pm) else "not found", raw=pf_raw or None)
    fields["processing_fee_percent"] = field(pf_pct, ("printed text" if pm else "derived: fee / loan amount") if pf_pct is not None else "not found")
    fields["processing_fee_note"] = field(_clean(re.sub(rf"(INR|RS\.?)\s*{_NUM}", "", pf_raw, flags=re.I)).strip("() "),
                                          "printed text")
    if pf_amt is None and not pm:
        flags["processing_fee"] = "no processing / origination fee is printed in this letter - confirm it with the lender"

    # ---- insurance: an amount if printed, else whether it is mentioned at all
    ins_lines = [l for l in lines if re.search(r"insurance", l, re.I)]
    ins_amt = next((parse_amount(l)[0] for l in ins_lines if parse_amount(l)[0]), None)
    if ins_amt:
        ins = ins_amt
        ins_src = "printed text"
    elif ins_lines:
        cond = re.search(r"\(if\s+applicable\)|if\s+applicable|if\s+any|optional|mandatory", " ".join(ins_lines), re.I)
        ins = f"Mentioned ({cond.group(0).strip('()')}), no amount stated" if cond else "Mentioned, no amount stated"
        ins_src = "terms and conditions"
    else:
        ins, ins_src = "Not mentioned", "not printed"
    fields["insurance"] = field(ins, ins_src, raw=ins_lines[0] if ins_lines else None)

    # ---- tenure, moratorium, EMI
    hit = get("tenure")
    ten_raw = hit[0] if hit else ""
    phase = re.compile(r"PMII?|EMI|interest|instal", re.I)
    tm = re.search(r"(\d{1,3})\s*(?:monthly|months?|emis?|instal)", ten_raw, re.I)
    ty = re.search(r"(\d{1,2})\s*years?", ten_raw, re.I)
    tenure = int(tm.group(1)) if tm and not phase.search(ten_raw) else (int(ty.group(1)) * 12 if ty else None)
    phase_months: list[tuple[int, str]] = []
    if tenure is None and hit:
        i = hit[1]
        for l in lines[i:i + 4]:
            for n in re.findall(r"(\d{1,3})\s*months?", l, re.I):
                if phase.search(l) and not re.search(r"\d{1,3}\s*[-–]\s*\d{1,3}\s*months?", l):
                    phase_months.append((int(n), l))
        if len(phase_months) >= 2:
            tenure = sum(n for n, _ in phase_months)
            ten_raw = " + ".join(f"{n} months" for n, _ in phase_months)
        elif tm:
            tenure = int(tm.group(1))
    fields["tenure_months"] = field(tenure, "printed text" if tenure else "not found", raw=ten_raw or None)
    if tenure is None:
        flags["tenure_months"] = "tenure not found"
    periods = [(int(a), int(b), l) for l in lines for a, b in re.findall(r"\b(\d{1,3})\s*[-–to]+\s*(\d{1,3})\s*months?", l, re.I)]
    mor = next((b - a + 1 for a, b, l in periods if re.search(r"PMII|interest\s+only|moratorium|simple\s+interest", l, re.I)), None)
    if mor is None and len(phase_months) >= 2:
        mor = next((n for n, l in phase_months if re.search(r"PMII?|partial|interest", l, re.I) and not re.search(r"\bEMI\b", l)), None)
    hit = get("moratorium")
    if mor is None and hit:
        mm = re.search(r"(\d{1,3})\s*months?", hit[0], re.I)
        mor = int(mm.group(1)) if mm else None
    fields["moratorium_months"] = field(mor, "printed text" if mor else "not printed")
    emi_line = next((l for l in lines if re.search(r"\bEMI\b|equated\s+monthly", l, re.I) and parse_amount(l)[0]), "")
    emi = parse_amount(emi_line)[0] if emi_line else None
    fields["emi"] = field(emi, "printed text" if emi else "not printed", raw=emi_line or None)
    if periods and tenure:
        last = max(b for _, b, _ in periods)
        checks["instalment_periods_match_tenure"] = last == tenure
        if last != tenure:
            flags["tenure_months"] = f"instalment periods run to month {last} but the tenure says {tenure}"

    hit = get("loan_type")
    lt = hit[0] if hit else ""
    if not lt:                                  # 'Security: NA' / 'Security Offered : N/A'
        sec = re.search(r"^\s*(?:\d+\.\s*)?security(?:\s+offered)?\s*:?\s*\n?\s*(N/?A|NIL|NONE)\b", text, re.I | re.M)
        lt = "unsecured" if sec else ""
    lt = "Unsecured" if re.search(r"unsecured|without\s+collateral", lt or up, re.I) else \
        ("Secured" if re.search(r"\bsecured\b|collateral|mortgage", lt, re.I) else "")
    fields["loan_type"] = field(lt, "printed text" if lt else "not found")

    hit = get("sanction_date")
    sd = _date(hit[0]) if hit else ""
    sd_src = "printed text"
    if not sd:                                  # the letter's own date at the top ('Dated :01/10/2026')
        for l in lines[:12]:
            if re.match(r"^\s*(?:dated?\s*:?\s*)?\d", l, re.I):
                sd = _date(l)
                if sd:
                    sd_src = "date of the letter"
                    break
    fields["sanction_date"] = field(sd, sd_src if sd else "not found")
    vm = re.search(r"valid\s+(?:till|upto|up\s+to|until)\s*([^\n]+)", text, re.I)
    fields["valid_until"] = field(_date(vm.group(1)) if vm else "", "printed text" if vm else "not printed")

    for k in ("university", "country", "course", "lender"):
        if not fields[k]["value"]:
            flags.setdefault(k, "not found - enter manually")
    return finish("sanction_letter", fields, flags, ["lender", "loan_amount", "roi"], checks)


def lines_from_pdf(path) -> list[str] | None:
    """Text lines of a digital PDF, or None when it is a scan (no usable text layer)."""
    try:
        import fitz  # PyMuPDF
        with fitz.open(str(path)) as doc:
            pages = [p.get_text("text") for p in doc]
    except Exception:
        try:
            from pypdf import PdfReader
            pages = [p.extract_text() or "" for p in PdfReader(str(path)).pages]
        except Exception:
            return None
    text = "\n".join(pages)
    return text.splitlines() if len(re.sub(r"\s", "", text)) > 200 else None


def lines_from_ocr(ocr_pages: list[dict]) -> list[str]:
    """OCR lines -> printed rows ('Rate of Interest 11.50 % p.a.') in reading order."""
    from .common import lines_of
    from .ssc import _rows
    lines = lines_of(ocr_pages)
    if lines:
        # logos / stamps read as a short word in a box several text lines tall would glue
        # neighbouring table rows together
        hs = sorted(l.h for l in lines)
        med = hs[len(hs) // 2]
        lines = [l for l in lines if not (l.h > 2.2 * med and len(l.text) < 8)]
    return [r.text for r in _rows(lines)]


def extract_sanction_files(paths: list, images_of, ocr_engine) -> dict[str, Any]:
    """paths: uploaded files; images_of(path) renders a file to page images (for scans)."""
    from .common import ocr_upright
    all_lines: list[str] = []
    sources = set()
    for p in paths:
        lines = lines_from_pdf(p) if str(p).lower().endswith(".pdf") else None
        if lines:
            sources.add("PDF text")
        else:
            pages = [ocr_upright(img, ocr_engine, i + 1, ["SANCTION", "LOAN", "INTEREST", "AMOUNT", "UNIVERSITY"])
                     for i, img in enumerate(images_of(p))]
            lines = lines_from_ocr(pages)
            sources.add("OCR")
        all_lines += lines
    r = extract_sanction_text(all_lines, " + ".join(sorted(sources)))
    r["pages"] = len(paths)
    return r


def flat(result: dict) -> dict:
    return {k: v["value"] for k, v in result["fields"].items()} | {"document_type": result["document_type"]}
