# job_scorer.py - Score filtered jobs using two-tier Claude scoring
# ============================================================
# Reads filtered_jobs.csv, scores each job against your CV, and
# outputs scored_jobs.csv sorted by score (highest first).
#
# Two-tier scoring:
#   1. Pass 1 (Haiku 4.5): score every job cheaply
#   2. Pass 2 (Sonnet 4.6): re-score top candidates (>= RESCORE_THRESHOLD)
#      for higher-precision rankings
#
# Required environment variables:
#   ANTHROPIC_API_KEY  - your Anthropic API key
#   BASE_PATH          - folder where CSV files are read from and saved to
#
# Usage:  py -3.12 job_scorer.py
# First:  py -3.12 -m pip install anthropic
#
# IMPORTANT: You must set your Anthropic API key before running.
# Open Command Prompt and run:
#     setx ANTHROPIC_API_KEY "sk-ant-your-key-here"
# Then CLOSE and REOPEN Command Prompt before running the script.

import os
import sys
import csv
import time
from pathlib import Path
from typing import Literal

# Force UTF-8 output so special characters in job titles don't crash on Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# ── Check for API key early ──────────────────────────────────────────────────
API_KEY = os.environ.get("ANTHROPIC_API_KEY", "").strip()
if not API_KEY:
    print("=" * 60)
    print("ERROR: ANTHROPIC_API_KEY not found!")
    print()
    print("To fix this, open Command Prompt and run:")
    print('  setx ANTHROPIC_API_KEY "sk-ant-your-key-here"')
    print()
    print("Then CLOSE and REOPEN Command Prompt, and run this script again.")
    print("=" * 60)
    sys.exit(1)

# ── Check for anthropic package ──────────────────────────────────────────────
try:
    import anthropic
    from pydantic import BaseModel
except ImportError:
    print("=" * 60)
    print("ERROR: anthropic package not installed!")
    print()
    print("To fix this, run:")
    print("  py -3.12 -m pip install anthropic")
    print()
    print("Then run this script again.")
    print("=" * 60)
    sys.exit(1)

# ── Configuration ────────────────────────────────────────────────────────────
BASE_PATH = Path(os.environ.get("BASE_PATH", os.getcwd()))
INPUT_FILE = BASE_PATH / "filtered_jobs.csv"
OUTPUT_FILE = BASE_PATH / "scored_jobs.csv"
ALREADY_SCORED_FILE = BASE_PATH / "scored_jobs_history.csv"

# Two-tier model setup
HAIKU_MODEL = "claude-haiku-4-5"   # Pass 1: score every job
SONNET_MODEL = "claude-sonnet-4-6" # Pass 2: re-score top candidates only
RESCORE_THRESHOLD = 60             # Haiku score >= this → re-score with Sonnet

MAX_OUTPUT_TOKENS = 500
REQUEST_TIMEOUT = 60               # seconds per request
DELAY_BETWEEN_REQUESTS = 0.6       # ~100 RPM, well under Anthropic limits

# ── Candidate CV (system prompt — cached via prompt caching) ─────────────────
# The Anthropic API caches identical prompt prefixes for 5 minutes by default.
# Since this system prompt is the same for every job, it gets cached after the
# first request, reducing input cost by ~90% for subsequent scoring calls.
#
# CUSTOMIZE THIS: Replace the candidate profile below with your own CV details.

SYSTEM_PROMPT = """You are a job-matching expert. You will score how well a candidate matches a job posting.

## CANDIDATE PROFILE

**Location:** [Your city], Germany (open to relocation within Germany)
**Language:** English fluent. Does NOT speak German — cannot work in German-language roles.
**Education:** [Your degree], [Your university], graduating [year]
**Experience:** [X] years total

### Work History:
1. **[Job Title]** | [Company] ([years])
   - [Key achievement with metric]
   - [Key achievement with metric]

2. **[Job Title]** | [Company] ([years])
   - [Key achievement with metric]
   - [Key achievement with metric]

### Technical Skills:
[List your technical skills here]

### Soft Skills:
[List your soft skills here]

### Target Career Scopes:
- **Scope 1:** [description]
- **Scope 2:** [description]
- **Scope 3:** [description]
- **Scope 4:** [description]

## SCORING INSTRUCTIONS

You will receive a job posting. Score it 0-100 based on how well the candidate matches.

**Scoring criteria (weighted):**
- Role level fit (30%): Mid-level, associate, early manager = good. Junior/intern or Director/VP/C-suite = bad.
- Skills match (25%): How many required skills does the candidate have?
- Experience relevance (20%): How relevant is their background to this role?
- Language compatibility (15%): If the job requires German (spoken/written/native), score 0.
- Location fit (10%): Germany-based or remote-friendly = good. Other countries = bad.

**Critical rules:**
- If the job description is primarily in German → score 0, set german_language to true
- If the job explicitly requires "German fluent/native/C1/C2" or "Deutsch" as a requirement → score 0, set german_language to true
- If the job requires 10+ years experience → cap score at 30
- If the job is clearly for a different field (nursing, accounting, mechanical engineering, etc.) → score 0
- Senior roles (8+ years) are OK, cap at 70 unless it's a perfect match
- Director/VP/Head of = cap at 40

**Field meanings:**
- score (0-100): overall fit
- scope: best matching career track (genai | bi_analytics | chief_of_staff | consulting | none)
- german_language: true if German is required, false otherwise
- match_reasons: 2-3 bullet points on why this matches, max 150 chars total
- gaps: 1-2 key gaps or missing requirements, max 100 chars total
- recommendation: apply (>=70, strong match) | maybe (50-69, worth reviewing) | skip (<50, not a fit)"""


# ── Structured output schema ─────────────────────────────────────────────────
class JobScore(BaseModel):
    score: int
    scope: Literal["genai", "bi_analytics", "chief_of_staff", "consulting", "none"]
    german_language: bool
    match_reasons: str
    gaps: str
    recommendation: Literal["apply", "maybe", "skip"]


# ── Helper: build user message for a job ─────────────────────────────────────
def build_user_message(job: dict) -> str:
    """Build the user message for a single job."""
    title = job.get("title", "Unknown")
    company = job.get("company", "Unknown")
    location = job.get("location", "Unknown")
    description = job.get("description", "").strip()
    source = job.get("source", "unknown")

    if not description or len(description) < 50:
        return (
            f"## JOB POSTING (title + company only — no description available)\n\n"
            f"**Title:** {title}\n"
            f"**Company:** {company}\n"
            f"**Location:** {location}\n"
            f"**Source:** {source}\n\n"
            f"Score based on title and company only. "
            f"Be generous since we can't see full requirements — if the title sounds "
            f"relevant, score 50-65 range. If title is clearly irrelevant, score 0-20."
        )

    if len(description) > 3000:
        description = description[:3000] + "\n\n[... description truncated for length ...]"

    return (
        f"## JOB POSTING\n\n"
        f"**Title:** {title}\n"
        f"**Company:** {company}\n"
        f"**Location:** {location}\n"
        f"**Source:** {source}\n\n"
        f"**Description:**\n{description}"
    )


# ── Helper: score one job with a given model ─────────────────────────────────
def score_job(client: anthropic.Anthropic, model: str, user_message: str) -> dict | None:
    """Score one job with the given model. Returns dict on success, None on failure.

    The Anthropic SDK auto-retries 429s and 5xx with exponential backoff,
    so we only handle the terminal failure case here.
    """
    try:
        response = client.messages.parse(
            model=model,
            max_tokens=MAX_OUTPUT_TOKENS,
            system=[{
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }],
            messages=[{"role": "user", "content": user_message}],
            output_format=JobScore,
        )
    except anthropic.APIError:
        return None

    if response.stop_reason in ("refusal", "max_tokens"):
        return None
    if response.parsed_output is None:
        return None

    data = response.parsed_output.model_dump()
    data["score"] = max(0, min(100, int(data["score"])))
    return data


# ── Helper: load already-scored job URLs to skip duplicates ──────────────────
def load_scored_urls() -> set:
    """Load URLs of jobs that have already been scored (from history file)."""
    urls = set()

    for filepath in [OUTPUT_FILE, ALREADY_SCORED_FILE]:
        if filepath.exists():
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        url = row.get("job_url", "").strip()
                        if url:
                            urls.add(url)
            except Exception:
                pass

    return urls


# ── Helper: save one scored job to history file immediately ──────────────────
def save_to_history(scored_job: dict, fieldnames: list):
    """Append a scored job to the history file so progress isn't lost on abort."""
    file_exists = ALREADY_SCORED_FILE.exists() and ALREADY_SCORED_FILE.stat().st_size > 0

    try:
        with open(ALREADY_SCORED_FILE, "a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            if not file_exists:
                writer.writeheader()
            writer.writerow(scored_job)
    except Exception:
        pass  # Silent fail — history is a convenience, not critical


# ── Output column definition ─────────────────────────────────────────────────
FIELDNAMES = [
    "score", "recommendation", "scope_match", "title", "company",
    "location", "job_url", "source", "german_language",
    "match_reasons", "gaps", "description", "date_scored", "scored_by"
]


def _safe(text: str, n: int) -> str:
    return text[:n].encode("utf-8", errors="replace").decode("utf-8")


# ── Main scoring function ───────────────────────────────────────────────────
def score_jobs():
    """Read filtered jobs, two-tier score them, write results."""

    if not INPUT_FILE.exists():
        print(f"ERROR: {INPUT_FILE} not found!")
        print()
        print("Make sure you've run the scraper and filter scripts first:")
        print("  1. py -3.12 jobspy_scraper.py")
        print("  2. py -3.12 job_filter.py")
        sys.exit(1)

    print(f"Reading jobs from {INPUT_FILE}...")
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        jobs = list(csv.DictReader(f))
    print(f"  Found {len(jobs)} filtered jobs.")

    scored_urls = load_scored_urls()
    if scored_urls:
        original = len(jobs)
        jobs = [j for j in jobs if j.get("job_url", "").strip() not in scored_urls]
        skipped = original - len(jobs)
        if skipped > 0:
            print(f"  Skipping {skipped} already-scored jobs.")

    if not jobs:
        print("\nNo new jobs to score! All jobs have already been scored.")
        print(f"Check {OUTPUT_FILE} for results.")
        return

    print(f"  Scoring {len(jobs)} new jobs...\n")

    client = anthropic.Anthropic(api_key=API_KEY, timeout=REQUEST_TIMEOUT)

    # ── Pass 1: Haiku scores every job ───────────────────────────────────────
    print("=" * 60)
    print(f"PASS 1/2 — Haiku 4.5 scoring all {len(jobs)} jobs")
    print("=" * 60)

    pass1_results: list[dict | None] = [None] * len(jobs)
    errors_p1 = 0

    for i, job in enumerate(jobs):
        title = _safe(job.get("title", "Unknown"), 50)
        company = _safe(job.get("company", "Unknown"), 30)
        source = job.get("source", "unknown")
        print(f"  [{i + 1}/{len(jobs)}] {title} @ {company} ({source})", end=" ")

        user_msg = build_user_message(job)
        result = score_job(client, HAIKU_MODEL, user_msg)
        pass1_results[i] = result

        if result is None:
            errors_p1 += 1
            print("-> FAILED")
        elif result["german_language"]:
            print("-> GERMAN (score: 0)")
        else:
            tag = "[Y]" if result["recommendation"] == "apply" else ("~" if result["recommendation"] == "maybe" else "[X]")
            print(f"-> {tag} Haiku: {result['score']} ({result['recommendation']})")

        time.sleep(DELAY_BETWEEN_REQUESTS)

    # ── Pass 2: Sonnet re-scores top candidates ──────────────────────────────
    rescore_indices = [
        i for i, r in enumerate(pass1_results)
        if r is not None and not r["german_language"] and r["score"] >= RESCORE_THRESHOLD
    ]

    print()
    print("=" * 60)
    print(f"PASS 2/2 — Sonnet 4.6 re-scoring {len(rescore_indices)} candidates "
          f"(Haiku score >= {RESCORE_THRESHOLD})")
    print("=" * 60)

    sonnet_results: dict[int, dict] = {}
    errors_p2 = 0

    for n, idx in enumerate(rescore_indices, 1):
        job = jobs[idx]
        haiku = pass1_results[idx]
        title = _safe(job.get("title", "Unknown"), 50)
        company = _safe(job.get("company", "Unknown"), 30)
        print(f"  [{n}/{len(rescore_indices)}] {title} @ {company}", end=" ")

        user_msg = build_user_message(job)
        result = score_job(client, SONNET_MODEL, user_msg)

        if result is None:
            errors_p2 += 1
            print(f"-> FAILED (keeping Haiku score {haiku['score']})")
        else:
            sonnet_results[idx] = result
            delta = result["score"] - haiku["score"]
            sign = "+" if delta >= 0 else ""
            print(f"-> Sonnet: {haiku['score']} -> {result['score']} ({sign}{delta}) "
                  f"({result['recommendation']})")

        time.sleep(DELAY_BETWEEN_REQUESTS)

    # ── Build output rows (Sonnet result wins where present) ─────────────────
    all_scored = []
    english_kept = []
    german_count = 0

    for idx, job in enumerate(jobs):
        scored_job = {**job, "date_scored": time.strftime("%Y-%m-%d")}
        haiku = pass1_results[idx]
        sonnet = sonnet_results.get(idx)

        if haiku is None:
            scored_job.update({
                "score": -1,
                "scope_match": "error",
                "german_language": "",
                "match_reasons": "API error - needs manual review",
                "gaps": "",
                "recommendation": "maybe",
                "scored_by": "error",
            })
            all_scored.append(scored_job)
            english_kept.append(scored_job)  # keep errors for manual review
            save_to_history(scored_job, FIELDNAMES)
            continue

        result = sonnet if sonnet else haiku
        scored_by = "sonnet" if sonnet else "haiku"

        score = result["score"]
        rec = result["recommendation"]
        is_german = result["german_language"]

        if is_german:
            score = 0
            rec = "skip"
            german_count += 1

        scored_job.update({
            "score": score,
            "scope_match": result["scope"],
            "german_language": str(is_german).lower(),
            "match_reasons": result["match_reasons"],
            "gaps": result["gaps"],
            "recommendation": rec,
            "scored_by": scored_by,
        })
        all_scored.append(scored_job)
        if not is_german and score > 0:
            english_kept.append(scored_job)

        save_to_history(scored_job, FIELDNAMES)

    # ── Write results (English jobs only, sorted) ────────────────────────────
    english_kept.sort(key=lambda j: (j["score"] if j["score"] >= 0 else -999), reverse=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        for j in english_kept:
            writer.writerow(j)

    # ── Summary stats ────────────────────────────────────────────────────────
    apply_count = sum(1 for j in english_kept if j.get("recommendation") == "apply")
    maybe_count = sum(1 for j in english_kept if j.get("recommendation") == "maybe")
    skip_count = sum(1 for j in english_kept if j.get("recommendation") == "skip")
    sonnet_used = sum(1 for j in english_kept if j.get("scored_by") == "sonnet")
    haiku_used = sum(1 for j in english_kept if j.get("scored_by") == "haiku")
    zero_score_removed = len(all_scored) - german_count - len(english_kept) - errors_p1

    print(f"\n{'=' * 60}")
    print("SCORING COMPLETE")
    print("=" * 60)
    print(f"  Total scored:        {len(all_scored)}")
    print(f"  German removed:      {german_count}")
    print(f"  Zero-score removed:  {zero_score_removed}")
    print(f"  Results kept:        {len(english_kept)}")
    print(f"    Sonnet re-scored:  {sonnet_used}")
    print(f"    Haiku only:        {haiku_used}")
    print(f"    APPLY (70+):       {apply_count}")
    print(f"    MAYBE (50-69):     {maybe_count}")
    print(f"    SKIP (<50):        {skip_count}")
    print(f"  Pass 1 errors:       {errors_p1}")
    print(f"  Pass 2 errors:       {errors_p2}")
    print()
    print(f"  Results saved to:    {OUTPUT_FILE}")
    print("  (German + zero-score jobs saved to history only — won't be re-scored)")

    top_jobs = [j for j in english_kept if j["score"] >= 50][:10]
    if top_jobs:
        print(f"\n{'=' * 60}")
        print("TOP MATCHES")
        print("=" * 60)
        for j in top_jobs:
            score = j["score"]
            title = j["title"][:45]
            company = j["company"][:25]
            scope = j["scope_match"]
            tier = j.get("scored_by", "?")
            print(f"  {score:>3}  {title:<45}  {company:<25}  [{scope}]  ({tier})")

    print(f"\nDone! Open {OUTPUT_FILE.name} to review your scored jobs.")


# ── Run ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("JOB SCORER — Claude Haiku 4.5 + Sonnet 4.6 (two-tier)")
    print("=" * 60)
    print()

    job_count = 0
    if INPUT_FILE.exists():
        with open(INPUT_FILE, "r", encoding="utf-8") as f:
            job_count = sum(1 for _ in csv.DictReader(f))

        scored_urls = load_scored_urls()
        new_jobs = max(0, job_count - len([u for u in scored_urls if u]))

        if new_jobs > 0:
            # Rough cost estimate (with prompt caching on system prompt):
            #   Haiku 4.5: ~$0.0015 per call
            #   Sonnet 4.6: ~$0.005 per call (assumes ~30% of jobs cross threshold)
            est_haiku = new_jobs * 0.0015
            est_sonnet = (new_jobs * 0.30) * 0.005
            est_cost = est_haiku + est_sonnet
            est_seconds = new_jobs * (DELAY_BETWEEN_REQUESTS + 1.5)  # ~1.5s per call
            print(f"  {job_count} total filtered jobs, ~{new_jobs} new to score")
            print(f"  Estimated cost: ~${est_cost:.2f}  (Haiku ${est_haiku:.2f} + Sonnet ${est_sonnet:.2f})")
            print(f"  Estimated time: ~{est_seconds / 60:.1f} minutes")
        else:
            print(f"  {job_count} filtered jobs found, checking for new ones...")
    else:
        print(f"  {INPUT_FILE} not found — run the scraper & filter first!")
        sys.exit(1)

    print()
    print("Starting scoring automatically via n8n...")
    print()

    score_jobs()
