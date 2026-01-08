import json, re, hashlib, unicodedata
from pathlib import Path
from typing import Dict, Any, List
from unidecode import unidecode

# ================== CONFIG ==================
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
CONFIG_DIR = ROOT / "config"

TAXONOMY_PATH = CONFIG_DIR / "taxonomy.json"
TOPIC_RULES_PATH = CONFIG_DIR / "topic_rules.json"

OUT_CLEAN_JSON = PROC_DIR / "kb_clean.json"
OUT_CHUNKS_JSONL = PROC_DIR / "kb_chunks.jsonl"

PROC_DIR.mkdir(parents=True, exist_ok=True)

# ================== TEXT HELPERS ==================
def strip_html(text: str) -> str:
    text = re.sub(r"<\s*br\s*/?>", "\n", text, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    return text

def normalize_text(s: str) -> str:
    s = unidecode(s)
    s = strip_html(s)
    s = s.replace("\xa0", " ")
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n\s*\n\s*", "\n\n", s)
    return s.strip()

def stable_id(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]

def tokenize(text: str) -> set:
    return set(re.findall(r"[a-zA-Z0-9\+\-]+", text.lower()))

# ================== LOADERS ==================
def load_json_files() -> List[Dict[str, Any]]:
    items = []
    for fp in RAW_DIR.glob("*.json"):
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict):
            for v in data.values():
                if isinstance(v, list):
                    data = v
                    break

        if not isinstance(data, list):
            raise ValueError(f"{fp.name} must be a JSON array")

        for obj in data:
            obj["_source_file"] = fp.name
            items.append(obj)

    return items

def load_topic_rules():
    return json.loads(TOPIC_RULES_PATH.read_text(encoding="utf-8"))

# ================== SUBJECT FROM FILE ==================
def topic_from_filename(fname: str) -> str:
    fname = fname.lower()
    if "database" in fname or "dbms" in fname:
        return "DBMS"
    if "oops" in fname:
        return "OOPs"
    if "os" in fname:
        return "OS"
    return "uncategorized"

# In load_json_files, ensure the source file is ALWAYS set before appending
def load_json_files() -> List[Dict[str, Any]]:
    items = []
    for fp in RAW_DIR.glob("*.json"):
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Standardize data to a list
        if isinstance(data, dict):
            # Try to find the list inside the dict
            found_list = False
            for v in data.values():
                if isinstance(v, list):
                    data = v
                    found_list = True
                    break
            if not found_list:
                data = [data] # Wrap single object in list

        for obj in data:
            # Assign source file directly from the Path object
            obj["_source_file"] = str(fp.name) 
            items.append(obj)
    return items



# ================== SUBTOPIC ASSIGNMENT ==================
def assign_subtopic(q: str, a: str, topic: str, rules: List[Dict[str, Any]]) -> str:
    text = f"{q} {a}".lower()
    tokens = tokenize(text)

    best_subtopic = None
    best_score = 0.0

    for rule in rules:
        if rule["topic"] != topic:
            continue

        keywords = rule["keywords"]
        matched = sum(1 for kw in keywords if kw in text or kw in tokens)
        coverage = matched / len(keywords)

        if coverage > best_score:
            best_score = coverage
            best_subtopic = rule["subtopic"]

    # confidence threshold
    if best_score >= 0.25:
        return best_subtopic

    # safe fallback
    fallback = {
        "DBMS": "DBMS Architecture",
        "OOPs": "Classes",
        "OS": "Processes"
    }
    return fallback.get(topic)

# ================== DEADLOCK / MEMORY DISAMBIGUATION ==================
def refine_subtopic(q: str, a: str, topic: str, subtopic: str) -> str:
    text = f"{q} {a}".lower()

    if "deadlock" in text:
        if topic == "DBMS" and any(x in text for x in ["transaction", "lock"]):
            return "Deadlocks"
        if topic == "OS" and any(x in text for x in ["process", "resource"]):
            return "Deadlocks"

    if "memory" in text:
        if topic == "OOPs":
            return "Memory Management in OOP"
        if topic == "OS":
            return "Memory Management"

    return subtopic

# ================== DIFFICULTY ==================
def difficulty_heuristic(q: str, a: str) -> str:
    text = f"{q} {a}".lower()

    advanced = [
        "mvcc", "2pl", "serializable", "cap theorem", "wal",
        "b+ tree", "page replacement", "banker's algorithm",
        "vtable", "raii"
    ]

    beginner = [
        "what is", "define", "basic", "class", "object",
        "process", "thread", "primary key"
    ]

    if sum(1 for t in advanced if t in text) >= 2:
        return "Advanced"
    if sum(1 for t in beginner if t in text) >= 2:
        return "Beginner"
    return "Intermediate"

# ================== MAIN ==================
def main():
    raw_items = load_json_files()
    topic_rules = load_topic_rules()

    cleaned = []
    seen = set()

    print("🔄 Preparing KB with strict topic control...")

    for obj in raw_items:
        q = normalize_text(obj.get("question", ""))
        a = normalize_text(obj.get("answer", ""))

        if not q or not a:
            continue

        key = (q.lower(), a.lower())
        if key in seen:
            continue
        seen.add(key)

        _id = str(obj.get("id") or f"auto-{stable_id(q+a)}")
        print(obj["_source_file"])
        topic = topic_from_filename(obj["_source_file"])
        #print(topic)
        
        subtopic = assign_subtopic(q, a, topic, topic_rules)
        subtopic = refine_subtopic(q, a, topic, subtopic)

        difficulty = difficulty_heuristic(q, a)

        cleaned.append({
            "id": _id,
            "question": q,
            "answer": a,
            "topic": topic,
            "subtopic": subtopic,
            "difficulty": difficulty,
            "source": obj["_source_file"]
        })

    # -------- SAVE CLEAN JSON --------
    OUT_CLEAN_JSON.write_text(
        json.dumps(cleaned, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    # -------- BUILD CLEAN CHUNKS --------
    with OUT_CHUNKS_JSONL.open("w", encoding="utf-8") as f:
        for row in cleaned:
            chunk_text = f"Q: {row['question']}\nA: {row['answer']}"
            f.write(json.dumps({
                "id": row["id"],
                "text": chunk_text,
                "metadata": {
                    "topic": row["topic"],
                    "subtopic": row["subtopic"],
                    "difficulty": row["difficulty"],
                    "source": row["source"]
                }
            }, ensure_ascii=False) + "\n")

    print("✅ KB preparation complete")
    print(f"Clean JSON : {OUT_CLEAN_JSON}")
    print(f"Chunks     : {OUT_CHUNKS_JSONL}")

if __name__ == "__main__":
    main()
