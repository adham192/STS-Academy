"""
Evaluation for the Speculative RAG pipeline.

Test set format (eval_set.json), only question + ground_truth are needed.
ground_truth can be a string or a list of acceptable strings:

"""
import json
import logging
import re
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

HERE = Path(__file__).resolve()
SESSION_DIR = HERE.parents[2]               # src/session3 (data folders and app.log live here)
SRC_DIR = HERE.parents[3]                   # src/ (holds the shared logging_config.py)
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from logging_config import setup_logging  # noqa: E402

setup_logging(SESSION_DIR)
logger = logging.getLogger("evaluate")

# Import the pipeline pieces used by both the app and evaluation.
from Speculative_RAG import build_index, retrieve, cluster, build_subsets, draft, verify

BASE_DIR = Path(__file__).resolve().parent
EVAL_PATH = SESSION_DIR / "JSON-Test-cases/ground_truth.json"
RESULTS_PATH = SESSION_DIR / "JSON-evaluation-results/eval_results.json"
RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
K = 10
N_SUBSETS = 3
SLEEP_BETWEEN_QUESTIONS = 0  # raise (e.g. 5) if you hit Gemini rate limits


# --------------------------------------------------------------------------- #
# Matching helpers
# --------------------------------------------------------------------------- #
def contains(text, truths):
    """Match an answer phrase or an equivalent numeric value."""
    if isinstance(truths, str):
        truths = [truths]

    def number_values(value):
        values = []
        for match in re.finditer(r"(?<![\w.])[-+]?\d[\d,]*(?:\.\d+)?\s*%?", value):
            token = match.group(0).strip()
            is_percent = token.endswith("%")
            token = token.rstrip("%").replace(",", "").strip()
            try:
                number = float(token)
                values.append(number / 100 if is_percent else number)
            except ValueError:
                continue
        return values

    answer_numbers = number_values(text)
    for truth in truths:
        truth = str(truth)
        numeric_truth = truth.replace(",", "").strip()
        # Numeric ground truths must match a whole numeric value, allowing
        # equivalent percentage notation (0.932 == 93.2%).
        if re.fullmatch(r"[-+]?\d+(?:\.\d+)?%?", numeric_truth):
            truth_numbers = number_values(truth)
            if any(abs(candidate - expected) <= max(1e-9, abs(expected) * 1e-9)
                   for expected in truth_numbers for candidate in answer_numbers):
                return True
            continue

        needle = re.sub(r"[^a-z0-9]+", " ", truth.lower()).strip()
        haystack = re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()
        if needle and re.search(rf"(?<![a-z0-9]){re.escape(needle)}(?![a-z0-9])", haystack):
            return True
    return False


# --------------------------------------------------------------------------- #
# Pipelines (return internals so each stage can be measured)
# --------------------------------------------------------------------------- #
def run_speculative(question, k=K, n_subsets=N_SUBSETS):
    logger.debug("Retrieving document chunks (k=%d)", k)
    print("   Retrieving document chunks...", flush=True)
    docs, _metas, embs = retrieve(question, k=k)
    if not docs:
        logger.warning("No chunks retrieved for: %s", question)
        return {
            "answer": "", "docs": [], "subsets": [], "drafts": [],
            "scores": [], "evidence": "", "best": None,
        }

    labels = cluster(embs, n_clusters=n_subsets)
    subsets = build_subsets(docs, labels, n_subsets=n_subsets)

    logger.debug("Generating %d Ollama drafts", n_subsets)
    print(f"   Generating {n_subsets} Ollama drafts...", flush=True)
    with ThreadPoolExecutor(max_workers=n_subsets) as pool:
        drafts = list(pool.map(lambda s: draft(question, s), subsets))
    logger.debug("Comparing drafts with Gemini")
    print("   Comparing drafts with Gemini...", flush=True)
    scores, best, evidence, final_answer = verify(question, drafts, subsets)
    logger.debug("Gemini scores: %s, best draft: %s", scores, best)
    return {
        "answer": final_answer,
        "docs": docs,
        "subsets": subsets,
        "drafts": drafts,
        "scores": scores,
        "evidence": evidence,
        "best": best,
    }


def run_baseline(question, k=K):
    """Plain RAG: all k chunks in ONE prompt, ONE model call, no clustering/verifying."""
    logger.debug("Baseline retrieval (k=%d)", k)
    docs, _metas, _embs = retrieve(question, k=k)
    if not docs:
        logger.warning("Baseline: no chunks retrieved")
        return {"answer": "", "docs": []}
    return {"answer": draft(question, docs)["answer"], "docs": docs}


# --------------------------------------------------------------------------- #
# Per-question scoring
# --------------------------------------------------------------------------- #
def first_hit_rank(docs, truths):
    """1-based rank of the first retrieved chunk containing the ground truth (0 if none)."""
    for rank, d in enumerate(docs, start=1):
        if contains(d, truths):
            return rank
    return 0


def diagnose(r):
    """Name the first stage that lost the answer (only meaningful when final is wrong)."""
    if not r["retrieval_hit"]:
        return "retrieval"
    if not r["subset_hit"]:
        return "subsets"
    if not r["oracle"]:
        return "drafter"
    return "verifier"


def score_question(item):
    q, truth = item["question"], item["ground_truth"]

    t = time.time()
    out = run_speculative(q)
    spec_latency = time.time() - t

    t = time.time()
    base = run_baseline(q)
    base_latency = time.time() - t

    rank = first_hit_rank(out["docs"], truth)
    r = {
        "question": q,
        "ground_truth": truth,
        "spec_answer": out["answer"],
        "baseline_answer": base["answer"],
        "draft_answers": [d["answer"] for d in out["drafts"]],
        "scores": out["scores"],
        "verifier_evidence": out["evidence"],
        # retrieval
        "retrieval_hit": rank > 0,
        "reciprocal_rank": 1 / rank if rank else 0.0,
        # did at least one subset keep a chunk containing the answer?
        "subset_hit": any(contains(d, truth) for s in out["subsets"] for d in s),
        # drafting
        "oracle": any(contains(d["answer"], truth) for d in out["drafts"]),
        # end-to-end
        "spec_correct": contains(out["answer"], truth),
        "baseline_correct": contains(base["answer"], truth),
        "spec_latency": spec_latency,
        "baseline_latency": base_latency,
    }
    r["failure_stage"] = None if r["spec_correct"] else diagnose(r)
    logger.info(
        "Result: spec_correct=%s baseline_correct=%s failure_stage=%s spec=%.1fs baseline=%.1fs",
        r["spec_correct"], r["baseline_correct"], r["failure_stage"],
        r["spec_latency"], r["baseline_latency"],
    )
    return r


# --------------------------------------------------------------------------- #
# Aggregate metrics
# --------------------------------------------------------------------------- #
def summarize(results):
    failed = [r for r in results if r.get("status") == "error"]
    total = len(results)
    results = [r for r in results if r.get("status") == "ok"]
    n = len(results)
    print(f"\nEnd-to-end if failed questions count as wrong: "
          f"{sum(r['spec_correct'] for r in results)}/{total}")
    print(f"\nQuestions: {n} completed, {len(failed)} failed")
    if not n:
        logger.warning("No questions completed successfully; no metrics to summarize")
        print("No questions completed successfully; no metrics to summarize.")
        return
    pct = lambda key: sum(r[key] for r in results) / n
    logger.info(
        "Evaluation summary: %d completed, %d failed, Speculative %.1f%%, Baseline %.1f%%",
        n, len(failed), pct("spec_correct") * 100, pct("baseline_correct") * 100,
    )

    oracle_n = sum(r["oracle"] for r in results)
    final_correct_when_possible = sum(r["oracle"] and r["spec_correct"] for r in results)

    print("\n" + "=" * 60)
    print(f"EVALUATION RESULTS ({n} questions)")
    print("=" * 60)

    print("\n[Retrieval]")
    print(f"  Hit rate@{K}            : {pct('retrieval_hit'):.1%}  (answer text in a retrieved chunk)")
    print(f"  MRR                    : {sum(r['reciprocal_rank'] for r in results) / n:.3f}  (higher = answer ranked earlier)")
    print(f"  Subset coverage        : {pct('subset_hit'):.1%}  (answer text survived into some subset)")

    print("\n[Drafting]")
    print(f"  Oracle draft accuracy  : {pct('oracle'):.1%}  (at least one of the drafts is correct)")

    print("\n[Verification]")
    conditional_accuracy = final_correct_when_possible / oracle_n if oracle_n else 0.0
    print(
        f"  Final answer accuracy when a draft was correct: {conditional_accuracy:.1%}"
    )

    print("\n[End-to-end]")
    print(f"  Speculative RAG acc.   : {pct('spec_correct'):.1%}")
    print(f"  Baseline RAG acc.      : {pct('baseline_correct'):.1%}")

    print("\n[Latency, average per question]")
    print(f"  Speculative RAG        : {sum(r['spec_latency'] for r in results) / n:.1f}s")
    print(f"  Baseline RAG           : {sum(r['baseline_latency'] for r in results) / n:.1f}s")
    print("  Note: Gemini candidate comparison is rate-limited to about one call every 13 seconds.")

    failures = [r for r in results if not r["spec_correct"]]
    print(f"\n[Where the {len(failures)} wrong answers were lost]")
    for stage in ("retrieval", "subsets", "drafter", "verifier"):
        count = sum(r["failure_stage"] == stage for r in failures)
        print(f"  {stage:<10}: {count}")


def evaluate():
    logger.info("Evaluation started (K=%d, N_SUBSETS=%d)", K, N_SUBSETS)
    with open(EVAL_PATH, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or any(
        not isinstance(item, dict) or not item.get("question") or "ground_truth" not in item
        for item in data
    ):
        raise ValueError("ground_truth.json must be a list of objects with question and ground_truth fields")

    logger.info("Loaded %d test cases from %s", len(data), EVAL_PATH)
    build_index()

    verifier_calls = len(data)
    minimum_spacing_minutes = max(0, verifier_calls - 1) * 13 / 60
    print(
        f"Gemini rate-limit spacing alone may take about "
        f"{minimum_spacing_minutes:.1f} minutes ({verifier_calls} comparison calls); "
        "model response time and retries add to this."
    )

    results = []
    for i, item in enumerate(data, start=1):
        logger.info("[%d/%d] %s", i, len(data), item["question"])
        print(f"[{i}/{len(data)}] {item['question']}", flush=True)
        try:
            result = score_question(item)
            result["status"] = "ok"
            results.append(result)
        except Exception as e:
            logger.exception("Question %d failed", i)
            print(f"   skipped, error: {e}")
            results.append({
                "question": item.get("question", ""),
                "ground_truth": item.get("ground_truth"),
                "status": "error",
                "evaluation_error": str(e),
            })
            error_text = str(e)
            if any(marker in error_text for marker in (
                "Gemini quota is exhausted",
                "Gemini model ",
                "Gemini verification failed after",
            )):
                logger.error("Stopping evaluation because Gemini verification is unavailable")
                print("Stopping evaluation because Gemini verification is unavailable.")
                break
        if SLEEP_BETWEEN_QUESTIONS:
            time.sleep(SLEEP_BETWEEN_QUESTIONS)

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info("Per-question details saved to %s", RESULTS_PATH)
    print(f"\nPer-question details saved to {RESULTS_PATH}")
    summarize(results)


if __name__ == "__main__":
    evaluate()