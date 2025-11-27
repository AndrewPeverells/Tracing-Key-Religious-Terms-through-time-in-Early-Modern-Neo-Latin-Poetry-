"""
Corpus preparation:

1. Extract clean poem lines from TEI XML files into /texts as plain .txt.
2. Build metadata.csv from TEI headers (author, year, place, printer).
3. Optionally enrich unknown places (S.l.) using *_front.xml title-page data.
4. Assign region_bin and time_bin to metadata (25-year slices, manual regions).
5. Lemmatise the /texts corpus using LatinCy.
"""

from pathlib import Path
from collections import Counter
import csv
import os
import re

import pandas as pd
import spacy
from lxml import etree

# =============================================================================
# GLOBAL PATHS / CONFIG
# =============================================================================

# --- TEI namespace / exclusion rules -----------------------------------------
NS = {"t": "http://www.tei-c.org/ns/1.0"}
EXCLUDE = {
    "note", "pb", "cb", "lb", "fw", "figure", "milestone", "ref",
    "gap", "epigraph", "argument", "table", "list", "quote", "q",
    "castList", "listBibl", "bibl", "biblStruct", "cit"
}
INCLUDE_TITLES = False       # keep False for clean corpus
WRITE_EMPTY_FILES = False    # do NOT write empty ones

# --- noise filters for line cleaning -----------------------------------------
_ROMAN = re.compile(r"^[IVXLCDM]+$", re.IGNORECASE)
NOISE_PATTERNS = [
    r"^CARM\.?\s*[IVXLCDM]+\.?$",
    r"^ELEGIA(E|E\.)?\s*[IVXLCDM]*\.?$",
    r"^ODE?A?(\.|E)?\s*[IVXLCDM]*\.?$",
    r"^EPIGRAMMA(TA)?\.?\s*[IVXLCDM]*\.?$",
    r"^LIB\.?\s*[IVXLCDM]+\.?$",
    r"^PROOEMI?UM\.?$",
    r"^PRAEFATIO\.?$",
    r"^ARGUMENTUM\.?$",
    r"^INDEX\.?$",
    r"^F\.?\s*N\.?\s*F\.?\s*$",
    r"^JANI\s+LERNUTII.*$",
    r"^MARTINI\s+BALTICI.*$",
    r"^AESTIVORUM\s+LIBER.*$",
    r"^AD\s+.+$",
    r"^IN\s+.+$",
]
NOISE_RE = [re.compile(p, re.IGNORECASE) for p in NOISE_PATTERNS]

# headings that indicate non-verse sections inside acts/scenes
HEAD_BLOCKLIST = [
    r"^PERSONAE\.?$", r"^DRAMATIS\s+PERSONAE\.?$",
    r"^AUCTORES.*$", r"^CATALOGUS.*$", r"^INDEX.*$",
    r"^AD\s+LECTOREM\.?$", r"^PRAEFATIO\.?$", r"^PROOEMI?UM\.?$",
]
HEAD_BLOCK_RE = [re.compile(p, re.IGNORECASE) for p in HEAD_BLOCKLIST]

YEAR_RE = re.compile(r"(\d{4})")


# =============================================================================
# 1. TEI UTILITY FUNCTIONS (NOISE DETECTION, HEADINGS)
# =============================================================================

def is_mostly_caps(s: str) -> bool:
    letters = [c for c in s if c.isalpha()]
    if not letters:
        return False
    uppers = sum(1 for c in letters if c.isupper())
    return uppers / len(letters) >= 0.9

def looks_like_noise_line(s: str) -> bool:
    t = s.strip()
    if not t:
        return False
    if _ROMAN.match(t.replace(".", "")):
        return True
    if is_mostly_caps(t) and len(t) <= 60:
        return True
    for rx in NOISE_RE:
        if rx.match(t):
            return True
    return False

def nearest_head_text(node, q, ns):
    """
    Find the nearest heading for this node:
    - check preceding-sibling <head> in the same parent
    - else walk up ancestors and use their child <head>
    """
    parent = node.getparent()
    if parent is not None:
        heads = parent.findall(f"./{q('head')}", namespaces=ns)
        if heads:
            text = " ".join(" ".join(heads[-1].itertext()).split())
            if text:
                return text
    for anc in node.iterancestors():
        heads = anc.findall(f"./{q('head')}", namespaces=ns)
        if heads:
            text = " ".join(" ".join(heads[-1].itertext()).split())
            if text:
                return text
    return ""


# =============================================================================
# 2. TEI → PLAIN-TEXT EXTRACTION
# =============================================================================

def extract_text_from_tei(xml_path: Path) -> str:
    """
    Extract cleaned poem lines from a TEI XML file, excluding paratext
    and headings.
    """
    parser = etree.XMLParser(recover=True, remove_comments=True)
    tree = etree.parse(str(xml_path), parser)
    root = tree.getroot()

    def q(tag: str) -> str:
        return f"t:{tag}" if root.tag.startswith("{http://www.tei-c.org/ns/1.0}") else tag

    ns = NS if root.tag.startswith("{http://www.tei-c.org/ns/1.0}") else None
    text_node = root.find(f".//{q('text')}", namespaces=ns)
    if text_node is None:
        return ""

    def prune(elem):
        for tag in EXCLUDE:
            for e in elem.findall(f".//{q(tag)}", namespaces=ns):
                p = e.getparent()
                if p is not None:
                    p.remove(e)

    lines = []

    # --- Primary: <div2|div3 type="poem">
    poem_divs = text_node.xpath(
        f".//{q('div2')}[@type='poem'] | .//{q('div3')}[@type='poem']",
        namespaces=ns
    )

    if poem_divs:
        for div in poem_divs:
            prune(div)

            if INCLUDE_TITLES:
                for h in div.findall(f"./{q('head')}", namespaces=ns):
                    ht = " ".join(" ".join(h.itertext()).split())
                    if ht and not looks_like_noise_line(ht):
                        lines.append(ht)

            for l in div.findall(f".//{q('l')}", namespaces=ns):
                s = " ".join(" ".join(l.itertext()).split())
                if s and not looks_like_noise_line(s):
                    lines.append(s)

            lines.append("")
    else:
        # --- Smarter fallback: allow acts/scenes, but skip blocklisted headings and <stage>/<argument>
        candidates = text_node.findall(f".//{q('lg')}", namespaces=ns)
        fallback = etree.Element("FALLBACK")
        for lg in candidates:
            in_stage = lg.xpath(f"boolean(ancestor::{q('stage')})", namespaces=ns)
            in_argument = lg.xpath(f"boolean(ancestor::{q('argument')})", namespaces=ns)
            if in_stage or in_argument:
                continue

            head_text = nearest_head_text(lg, q, ns)
            if head_text and any(rx.match(head_text) for rx in HEAD_BLOCK_RE):
                continue

            fallback.append(lg)

        prune(fallback)

        for l in fallback.findall(f".//{q('l')}", namespaces=ns):
            s = " ".join(" ".join(l.itertext()).split())
            if s and not looks_like_noise_line(s):
                lines.append(s)
        if lines:
            lines.append("")

    # === Speaker / role label filter (e.g., "SCA.", "POE.") ===
    cleaned_lines = []
    for line in lines:
        t = line.strip()
        if not t:
            cleaned_lines.append("")
            continue

        # remove leading "ABC." speaker markers (1–5 uppercase letters + period + space)
        t = re.sub(r"^[A-ZÆŒ]{1,5}\.\s+", "", t)

        if looks_like_noise_line(t):
            continue

        cleaned_lines.append(t)

    # normalize blank lines
    normalized, last_blank = [], False
    for line in cleaned_lines:
        if not line.strip():
            if not last_blank:
                normalized.append("")
            last_blank = True
        else:
            normalized.append(line)
            last_blank = False

    while normalized and not normalized[0].strip():
        normalized.pop(0)
    while normalized and not normalized[-1].strip():
        normalized.pop()

    return "\n".join(normalized)


def xml_to_text_corpus():
    """
    Walk the XML_INPUT_DIR, extract poem text from each TEI file,
    and write clean .txt files into TEXT_OUTPUT_DIR subfolder.
    """
    failed = []
    written = 0
    with_lines = 0
    empty_out = 0

    all_xmls = sorted(XML_INPUT_DIR.rglob("*.xml"))
    for xml_file in all_xmls:
        try:
            text = extract_text_from_tei(xml_file)
            rel = xml_file.relative_to(XML_INPUT_DIR)
            out_path = (TEXT_OUTPUT_DIR / rel).with_suffix(".txt")
            out_path.parent.mkdir(parents=True, exist_ok=True)

            if text.strip():
                with_lines += 1
                out_path.write_text(text, encoding="utf-8")
                written += 1
            else:
                empty_out += 1
                if WRITE_EMPTY_FILES:
                    out_path.write_text("", encoding="utf-8")
                    written += 1

        except Exception as e:
            failed.append((xml_file, str(e)))

    print(f"Processed {len(all_xmls)} XML files.")
    print(f"  - With poem lines: {with_lines}")
    print(f"  - Skipped empty outputs: {empty_out}")
    print(f"Wrote {written} text files (clean outputs only).")
    if failed:
        print(f"{len(failed)} files failed (showing up to 10):")
        for f, err in failed[:10]:
            print(" -", f, "→", err)


# =============================================================================
# 3. METADATA EXTRACTION FROM TEI
# =============================================================================

def get_text(node):
    """Return node text collapsed to single spaces, or '' if missing."""
    if node is None:
        return ""
    txt = "".join(node.itertext())
    return " ".join(txt.split())

def parse_imprint(imprint_raw: str):
    """
    Parse strings like:
      'Hanau: Biermann, 1613.'
      'Basileae : (Ex officina Ioannis Oporini, 1551).'
      'Koeln: Maternus Cholinus, 1562.'
      'Ingolstadt 1594'
    Returns (place, name, year) as strings.
    """
    if not imprint_raw:
        return "", "", ""

    raw = " ".join(imprint_raw.split())

    m = YEAR_RE.search(raw)
    year = m.group(1) if m else ""

    core = raw
    if year:
        idx = core.find(year)
        if idx != -1:
            core = core[:idx]
    core = core.strip(" ,.;:()[]")

    place = ""
    name  = ""

    if ":" in core:
        left, right = core.split(":", 1)
        place = left.strip(" ,.;:()[]")
        name  = right.strip(" ,.;:()[]")
    else:
        parts = core.split(",", 1)
        place = parts[0].strip(" ,.;:()[]")
        if len(parts) > 1:
            name = parts[1].strip(" ,.;:()[]")

    return place, name, year

def extract_metadata_from_file(xml_path: Path):
    """
    From a TEI XML file, extract:
      - title = filename
      - author from titleStmt/author
      - place, printer, year from sourceDesc/bibl imprint
    """
    tree = etree.parse(str(xml_path))
    root = tree.getroot()

    file_title = xml_path.name

    author_el = root.find(".//teiHeader/fileDesc/titleStmt/author")
    author = get_text(author_el)

    bibl_el = root.find(".//teiHeader/fileDesc/sourceDesc/bibl")
    imprint_raw = get_text(bibl_el)
    place, name, year = parse_imprint(imprint_raw)

    region_bin = ""
    time_bin   = ""

    return {
        "title": file_title,
        "author": author,
        "year_of_publication": year,
        "place_of_publication": place,
        "printer_editor_publisher": name,
        "region_bin": region_bin,
        "time_bin": time_bin,
    }

def build_metadata_csv():
    """
    Walk XML_INPUT_DIR and build metadata.csv with core bibliographic info.
    """
    rows = []
    xml_files = sorted(XML_INPUT_DIR.rglob("*.xml"))
    if not xml_files:
        raise SystemExit(f"No XML files found under {XML_INPUT_DIR}")

    print(f"Found {len(xml_files)} XML files under {XML_INPUT_DIR}")

    for xml_path in xml_files:
        try:
            meta = extract_metadata_from_file(xml_path)
            rows.append(meta)
        except Exception as e:
            print(f"!! Error parsing {xml_path}: {e}")

    fieldnames = [
        "title",
        "author",
        "year_of_publication",
        "place_of_publication",
        "printer_editor_publisher",
        "region_bin",
        "time_bin",
    ]

    with METADATA_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {METADATA_CSV}")


# =============================================================================
# 4. METADATA ENRICHMENT (S.L. PLACES FROM *_front.XML)
# =============================================================================

def is_sl(value: str) -> bool:
    """Return True if value looks like S.l. / S. l. / s.l. etc."""
    if not value:
        return False
    norm = value.lower().replace(" ", "").replace(".", "")
    return norm in {"sl", "sol", "sln"} or "no place" in value.lower()

def extract_front_meta(xml_path: Path):
    """
    From a *_front.xml file, try to get:
      - place: <name type="place">...</name> inside titlePage
      - printer: all <name type="person">...</name> inside titlePage, joined
    """
    tree = etree.parse(str(xml_path))
    root = tree.getroot()

    title_page = root.find(".//titlePage")
    if title_page is None:
        title_page = root

    place_el = title_page.find(".//name[@type='place']")
    place = get_text(place_el)

    persons = [
        get_text(el) for el in title_page.findall(".//name[@type='person']")
    ]
    printer = "; ".join(p for p in persons if p)

    return place, printer

def enrich_metadata_from_front():
    """
    Use *_front.xml files to fill in missing / S.l. places and printers
    in metadata.csv, writing the result to metadata_enriched.csv.
    """
    front_meta = {}
    front_files = sorted(XML_INPUT_DIR.rglob("*_front.xml"))
    print(f"Found {len(front_files)} front files")

    for fp in front_files:
        base_id = fp.name.split("_", 1)[0]
        try:
            place, printer = extract_front_meta(fp)
            if place:
                front_meta[base_id] = {
                    "place": place,
                    "printer": printer
                }
        except Exception as e:
            print(f"!! Error parsing front file {fp}: {e}")

    print(f"Collected front metadata for {len(front_meta)} base IDs")

    with METADATA_CSV.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    expected_cols = {
        "title",
        "author",
        "year_of_publication",
        "place_of_publication",
        "printer_editor_publisher",
        "region_bin",
        "time_bin",
    }
    missing = expected_cols - set(fieldnames or [])
    if missing:
        print(f"WARNING: metadata.csv missing columns: {missing}")

    updated = 0
    for row in rows:
        title = row.get("title", "")
        if not title:
            continue

        if "_front" in title or "_back" in title:
            continue

        base_id = title.split("_", 1)[0]

        if base_id not in front_meta:
            continue

        place_row = row.get("place_of_publication", "")
        printer_row = row.get("printer_editor_publisher", "")

        if is_sl(place_row):
            new_place = front_meta[base_id]["place"]
            if new_place:
                row["place_of_publication"] = new_place
                updated += 1

        if not printer_row:
            new_printer = front_meta[base_id]["printer"]
            if new_printer:
                row["printer_editor_publisher"] = new_printer

    print(f"Updated place_of_publication for {updated} rows")

    with METADATA_ENRICHED.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote enriched metadata to {METADATA_ENRICHED}")


# =============================================================================
# 5. TIME AND REGION BINS
# =============================================================================

PLACE_TO_REGION = {
    # North
    "Altenburg":       "North",
    "Braunsberg":      "North",
    "Bremen":          "North",
    "Breslau":         "North",
    "Dresden":         "North",
    "Hannover":        "North",
    "Helmstadt":       "North",
    "Jena":            "North",
    "Leipzig":         "North",
    "Lübeck":          "North",
    "Rostock":         "North",
    "Wittemberg":      "North",
    "Zwickau":         "North",

    # South
    "Amberg":                  "South",
    "Augsburg":                "South",
    "Augsburg, Ingolstadt":    "South",
    "Balingen":                "South",
    "Dillingen":               "South",
    "Ingolstadt":              "South",
    "München":                 "South",
    "Nürnberg":                "South",
    "Pforzheim":               "South",
    "Salzburg":                "South",
    "Schwäbisch Hall":         "South",
    "Stuttgart":               "South",
    "Tübingen":                "South",
    "Wien":                    "South",

    # Central / West
    "Frankfurt":       "Central/West",
    "Freiburg":        "Central/West",
    "Hanau":           "Central/West",
    "Heidelberg":      "Central/West",
    "Herborn":         "Central/West",
    "Neustadt":        "Central/West",
    "Speyer":          "Central/West",
    "Köln":            "Central/West",

    # Swiss
    "Basel":           "Swiss",
    "Zürich":          "Swiss",

    # Low Countries
    "Amsterdam":       "Low Countries",
    "Antwerpen":       "Low Countries",

    # France / Alsace
    "Paris":           "France/Alsace",
    "Strasbourg":      "France/Alsace",

    # Bohemia
    "Prague":          "Bohemia",

    # Unknown / no place
    "Sine loco":       "Unknown",
    "S.L.":            "Unknown",
    "S. L.":           "Unknown",
    "S.l":             "Unknown",
}

def assign_time_bin_25(year):
    try:
        y = int(year)
    except (TypeError, ValueError):
        return "Unknown"
    start = 1500 + ((y - 1500) // 25) * 25
    end = start + 24
    return f"{start}-{end}"

def add_time_and_region_bins(use_enriched=True):
    """
    Add region_bin and time_bin to metadata.
    """
    src = METADATA_ENRICHED if use_enriched and METADATA_ENRICHED.exists() else METADATA_CSV
    df = pd.read_csv(src, encoding="utf-8-sig")

    year_col  = next(c for c in df.columns if "year" in c.lower())
    place_col = next(c for c in df.columns if "place" in c.lower())

    df["region_bin"] = df[place_col].map(PLACE_TO_REGION).fillna("Unknown")
    df["time_bin"] = df[year_col].map(assign_time_bin_25)

    df.to_csv(src, index=False, encoding="utf-8-sig")
    print(f"Updated {src.name} with region_bin and time_bin")

# =============================================================================
# 6. LATINCY LEMMATISER (CHUNKED, BATCH-SAFE)
# =============================================================================

def run_latin_lemmatiser():
    """
    Lemmatiser for the text corpus using LatinCy:
    - Processes files in batches (FILES_PER_BATCH + BATCH_INDEX).
    - Splits very long files into character chunks to avoid spaCy limits.
    - Writes space-separated lemmas into LEMM_OUTPUT_DIR mirroring structure.
    """
    INPUT_DIR       = TEXT_OUTPUT_DIR
    OUTPUT_DIR      = LEMM_OUTPUT_DIR
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    MODEL_NAME      = "la_core_web_sm"

    # ---- file-level batching ----
    FILES_PER_BATCH = 50      # process this many files per run
    BATCH_INDEX     = 0       # 0, 1, 2, ... pick the slice you want

    # ---- spaCy throughput (keep conservative in Jupyter/WSL) ----
    N_PROCESS       = 1       # keep 1 to avoid lockups
    PIPE_BATCH_SIZE = 9       # docs/chunks per nlp.pipe batch

    # ---- long-file chunking ----
    MAX_LENGTH      = 400_000   # spaCy safety cap (chars)
    CHUNK_CHARS     = 300_000   # chunk size (<= MAX_LENGTH with margin)

    FORCE_REDO      = False     # set True to re-write outputs

    # ================== LOAD NLP ==================
    nlp = spacy.load(MODEL_NAME)
    nlp.max_length = MAX_LENGTH
    disable_comps = [c for c in ("parser", "ner", "textcat") if c in nlp.pipe_names]

    # ================== FILE LIST & BATCH SLICE ==================
    files_all = sorted(INPUT_DIR.rglob("*.txt"))
    if not files_all:
        raise SystemExit(f"No .txt files under {INPUT_DIR}")

    start = BATCH_INDEX * FILES_PER_BATCH
    end   = min(start + FILES_PER_BATCH, len(files_all))
    batch_files = files_all[start:end]
    if not batch_files:
        raise SystemExit(f"No files in slice [{start}:{end}]. Adjust BATCH_INDEX/FILES_PER_BATCH.")

    print(f"Total files: {len(files_all)}  |  Running batch {BATCH_INDEX} -> {len(batch_files)} files")

    # ================== GENERATOR OF CHUNKS ==================
    def iter_text_chunks():
        for fp in batch_files:
            rel = fp.relative_to(INPUT_DIR)
            out_fp = OUTPUT_DIR / rel
            out_fp.parent.mkdir(parents=True, exist_ok=True)

            if out_fp.exists() and out_fp.stat().st_size > 0 and not FORCE_REDO:
                continue

            part_fp = out_fp.with_suffix(out_fp.suffix + ".part")
            if part_fp.exists():
                part_fp.unlink()

            txt = fp.read_text(encoding="utf-8", errors="ignore")
            L = len(txt)

            if L <= CHUNK_CHARS:
                yield txt, (str(rel), True, True)
            else:
                start_idx = 0
                first = True
                while start_idx < L:
                    end_idx = min(start_idx + CHUNK_CHARS, L)
                    if end_idx < L:
                        ws = txt.rfind(" ", start_idx, end_idx)
                        if ws != -1 and ws > start_idx + int(0.8 * CHUNK_CHARS):
                            end_idx = ws
                    last = (end_idx >= L)
                    yield txt[start_idx:end_idx], (str(rel), first, last)
                    first = False
                    start_idx = end_idx

    # ================== PROCESS ==================
    lemma_counter = Counter()
    started_files = 0
    completed_files = 0

    with nlp.select_pipes(disable=disable_comps):
        for doc, (rel, is_first, is_last) in nlp.pipe(
            iter_text_chunks(),
            as_tuples=True,
            batch_size=PIPE_BATCH_SIZE,
            n_process=N_PROCESS,
        ):
            out_path = OUTPUT_DIR / rel
            part_fp  = out_path.with_suffix(out_path.suffix + ".part")

            if is_first:
                started_files += 1
                if started_files % 25 == 0:
                    print(f"Started {started_files} files in this batch...")

            lemmas = (t.lemma_.lower() for t in doc if t.is_alpha)
            lemma_counter.update(lemmas)

            lemmas_for_write = [t.lemma_.lower() for t in doc if t.is_alpha]
            mode = "w" if is_first else "a"
            with open(part_fp, mode, encoding="utf-8") as f:
                if mode == "a" and part_fp.stat().st_size > 0:
                    f.write(" ")
                f.write(" ".join(lemmas_for_write))

            if is_last:
                if out_path.exists():
                    out_path.unlink()
                part_fp.rename(out_path)
                completed_files += 1

    print("\n=== LEMMATIZED CORPUS (BATCH) SUMMARY ===")
    print(f"Batch index / size : {BATCH_INDEX} / {FILES_PER_BATCH}")
    print(f"Started files      : {started_files}")
    print(f"Completed files    : {completed_files}")
    print(f"Total lemma tokens : {sum(lemma_counter.values()):,}")
    print(f"Unique lemmas      : {len(lemma_counter):,}")