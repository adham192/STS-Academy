"""
Evaluation of the Speculative RAG pipeline against a plain-RAG baseline.

For every question in the ground-truth file the script:
  1. retrieves the top-k chunks and clusters them with K-Means,
  2. builds the evidence subsets (embedding-guided overlapping chains),
  3. writes one draft per subset with the local Ollama model,
  4. lets Gemini score the drafts (a second Gemini call breaks a tie for the top score),
  5. runs a plain-RAG baseline once, using all retrieved chunks in one prompt.

Reported metrics: final accuracy, oracle accuracy (any draft correct), subset hit
(a subset kept an answer-bearing chunk), retrieval MRR and average seconds.
"""
import json
import logging
import re
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

HERE = Path(__file__).resolve()
SESSION_DIR = HERE.parents[2]  # keep the same project layout as the original evaluator
SRC_DIR = HERE.parents[3]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from logging_config import setup_logging  # noqa: E402
setup_logging(SESSION_DIR)
logger = logging.getLogger("evaluate")

from Speculative_RAG import (  # noqa: E402
    build_index,
    list_documents,
    retrieve,
    cluster,
    build_subsets,
    draft,
    verify,
    DEFAULT_RETRIEVAL_K,
    DEFAULT_N_CLUSTERS,
    DEFAULT_N_SUBSETS,
)

EVAL_PATH = SESSION_DIR / "JSON-Test-cases/ground_truth.json"
RESULTS_PATH = SESSION_DIR / "JSON-evaluation-results/eval_results.json"
RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

K = DEFAULT_RETRIEVAL_K
N_SUBSETS = DEFAULT_N_SUBSETS
SLEEP_BETWEEN_QUESTIONS = 0


def contains(text, truths):
    """Return True if an accepted ground-truth phrase appears in text."""
    if isinstance(truths, str):
        truths = [truths]
    text_norm = re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()
    for truth in truths:
        needle = re.sub(r"[^a-z0-9]+", " ", str(truth).lower()).strip()
        if needle and re.search(
            rf"(?<![a-z0-9]){re.escape(needle)}(?![a-z0-9])",
            text_norm,
        ):
            return True
    return False


def matches_item(text, item):
    """
    Require at least one phrase from every required_concepts group.
    This is stricter than a single substring but tolerates common paraphrases.
    Falls back to ground_truth phrase matching for older test-set records.
    """
    groups = item.get("required_concepts")
    if not groups:
        return contains(text, item["ground_truth"])
    normalized = re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()
    for alternatives in groups:
        if not any(
            re.search(
                rf"(?<![a-z0-9]){re.escape(re.sub(r'[^a-z0-9]+', ' ', phrase.lower()).strip())}(?![a-z0-9])",
                normalized,
            )
            for phrase in alternatives
        ):
            return False
    return True


def run_drafts_and_verification(question, subsets):
    """Write one draft per subset with Ollama, then verify them with Gemini."""
    with ThreadPoolExecutor(max_workers=len(subsets)) as pool:
        drafts = list(pool.map(lambda subset: draft(question, subset), subsets))
    scores, best, tie_reason, final_answer = verify(question, drafts, subsets)
    return {
        "answer": final_answer,
        "drafts": drafts,
        "scores": scores,
        "best_draft_index": best,
        "verifier_decision": tie_reason,
    }


def score_one_question(item):
    question = item["question"]
    truth = item["ground_truth"]
    source = item.get("source") or None

    # Retrieval and clustering are performed once per question.
    started = time.perf_counter()
    docs, metas, embeddings, distances = retrieve(
        question, k=K, return_distances=True, source=source
    )
    retrieval_seconds = time.perf_counter() - started

    if not docs:
        return {
            "question": question,
            "level": item.get("level"),
            "ground_truth": truth,
            "source": source,
            "status": "no_retrieval",
        }

    labels = cluster(embeddings, n_clusters=DEFAULT_N_CLUSTERS)
    retrieval_rank = next(
        (rank for rank, doc in enumerate(docs, start=1) if matches_item(doc, item)),
        0,
    )

    started = time.perf_counter()
    subsets = build_subsets(
        docs,
        labels,
        n_subsets=N_SUBSETS,
        distances=distances,
        embeddings=embeddings,
    )
    construction_seconds = time.perf_counter() - started

    started = time.perf_counter()
    generated = run_drafts_and_verification(question, subsets)
    generation_and_verification_seconds = time.perf_counter() - started

    subset_hit = any(matches_item(doc, item) for subset in subsets for doc in subset)
    oracle = any(matches_item(draft_item["answer"], item)
                 for draft_item in generated["drafts"])
    final_correct = matches_item(generated["answer"], item)

    speculative = {
        "answer": generated["answer"],
        "draft_answers": [d["answer"] for d in generated["drafts"]],
        "draft_rationales": [d["rationale"] for d in generated["drafts"]],
        "verifier_scores": generated["scores"],
        "best_draft_index": generated["best_draft_index"],
        "verifier_decision": generated["verifier_decision"],
        "subset_chunk_positions": [
            [docs.index(doc) + 1 for doc in subset] for subset in subsets
        ],
        "subset_hit": subset_hit,
        "oracle_draft_correct": oracle,
        "final_correct": final_correct,
        "subset_construction_seconds": construction_seconds,
        "draft_and_verification_seconds": generation_and_verification_seconds,
    }

    # Plain RAG baseline uses the same retrieved top-k chunks and runs only once.
    started = time.perf_counter()
    baseline_answer = draft(question, docs)["answer"]
    baseline_seconds = time.perf_counter() - started

    return {
        "question": question,
        "level": item.get("level"),
        "topic": item.get("topic"),
        "ground_truth": truth,
        "source": source,
        "status": "ok",
        "retrieval": {
            "retrieved_chunks": len(docs),
            "answer_bearing_chunk_found": retrieval_rank > 0,
            "first_answer_bearing_rank": retrieval_rank or None,
            "reciprocal_rank": 1.0 / retrieval_rank if retrieval_rank else 0.0,
            "seconds": retrieval_seconds,
            "cluster_sizes": {
                str(label): int(sum(int(x) == int(label) for x in labels))
                for label in sorted(set(labels))
            },
        },
        "speculative_rag": speculative,
        "baseline": {
            "answer": baseline_answer,
            "correct": matches_item(baseline_answer, item),
            "seconds": baseline_seconds,
        },
    }


def summarize(results):
    completed = [r for r in results if r.get("status") == "ok"]
    failed = [r for r in results if r.get("status") != "ok"]
    print("\n" + "=" * 76)
    print("SPECULATIVE RAG EVALUATION")
    print("=" * 76)
    print(f"Completed: {len(completed)} | Failed/no retrieval: {len(failed)}")
    if not completed:
        print("No completed questions; metrics cannot be calculated.")
        return

    n = len(completed)
    spec = [r["speculative_rag"] for r in completed]
    accuracy = sum(s["final_correct"] for s in spec) / n
    oracle = sum(s["oracle_draft_correct"] for s in spec) / n
    subset_hit = sum(s["subset_hit"] for s in spec) / n
    mrr = sum(r["retrieval"]["reciprocal_rank"] for r in completed) / n
    avg_sec = sum(
        s["subset_construction_seconds"] + s["draft_and_verification_seconds"]
        for s in spec
    ) / n
    baseline_accuracy = sum(r["baseline"]["correct"] for r in completed) / n
    baseline_sec = sum(r["baseline"]["seconds"] for r in completed) / n

    header = f"{'System':<18} {'Accuracy':>10} {'Avg sec':>10}"
    print("\n" + header)
    print("-" * len(header))
    print(f"{'Speculative RAG':<18} {accuracy:>9.1%} {avg_sec:>10.1f}")
    print(f"{'Plain RAG':<18} {baseline_accuracy:>9.1%} {baseline_sec:>10.1f}")

    print(f"\nOracle draft accuracy : {oracle:.1%}")
    print(f"Subset hit rate       : {subset_hit:.1%}")
    print(f"Retrieval MRR         : {mrr:.3f}")

    print("\nAccuracy by difficulty")
    for level in ("easy", "medium", "hard"):
        rows = [r for r in completed if r.get("level") == level]
        if not rows:
            continue
        spec_acc = sum(r["speculative_rag"]["final_correct"] for r in rows) / len(rows)
        base_acc = sum(r["baseline"]["correct"] for r in rows) / len(rows)
        print(f"  {level.title()} ({len(rows)} questions)")
        print(f"    {'speculative':<12}: {spec_acc:.1%}")
        print(f"    {'baseline':<12}: {base_acc:.1%}")

    print("\nInterpretation notes:")
    print("- Oracle = at least one generated draft matched the ground-truth phrase.")
    print("- Subset hit = at least one subset retained a chunk containing that phrase.")
    print("- MRR measures retrieval only (where the first answer-bearing chunk ranks).")
    print("- Phrase matching is an automatic proxy; inspect saved answers for semantic equivalents.")


def evaluate():
    with open(EVAL_PATH, encoding="utf-8") as file:
        test_items = json.load(file)
    if not isinstance(test_items, list) or not test_items:
        raise ValueError("The ground-truth file must contain a non-empty JSON list.")
    for index, item in enumerate(test_items, start=1):
        if not isinstance(item, dict) or not item.get("question") or "ground_truth" not in item:
            raise ValueError(f"Invalid test item at position {index}: question and ground_truth are required.")

    print(f"Loaded {len(test_items)} questions from {EVAL_PATH}")
    print("Subset strategy: embedding-guided overlapping chains")
    print(f"Retrieval k={K}; clusters={DEFAULT_N_CLUSTERS}; subsets={N_SUBSETS}")
    print(
        f"Gemini verification calls: {len(test_items)} to {2 * len(test_items)} "
        "(one per question, plus one more whenever the top score is tied). "
        "At a 13-second minimum interval, pacing alone may take "
        f"{(len(test_items) - 1) * 13 / 60:.1f} to {(2 * len(test_items) - 1) * 13 / 60:.1f} minutes."
    )

    build_index()
    indexed = list_documents()
    unknown_sources = sorted({
        item["source"] for item in test_items
        if item.get("source") and item["source"] not in indexed
    })
    if unknown_sources:
        raise ValueError(
            "The following source PDFs are not indexed: "
            f"{unknown_sources}. Indexed documents: {sorted(indexed)}. "
            "Place GPO-CONAN-2022.pdf in the project's pdfs folder and rerun."
        )

    results = []
    for index, item in enumerate(test_items, start=1):
        print(f"\n[{index}/{len(test_items)}] [{item.get('level', 'unrated')}] {item['question']}", flush=True)
        try:
            result = score_one_question(item)
            results.append(result)
        except Exception as error:
            logger.exception("Question %d failed", index)
            results.append({
                "question": item.get("question", ""),
                "level": item.get("level"),
                "ground_truth": item.get("ground_truth"),
                "status": "error",
                "evaluation_error": str(error),
            })
            error_text = str(error).lower()
            if any(token in error_text for token in (
                "gemini quota is exhausted",
                "resource_exhausted",
                "gemini model",
                "gemini verification failed",
            )):
                print("Stopping: Gemini verification is unavailable or quota is exhausted.")
                break
        if SLEEP_BETWEEN_QUESTIONS:
            time.sleep(SLEEP_BETWEEN_QUESTIONS)

    with open(RESULTS_PATH, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)
    print(f"\nDetailed results saved to: {RESULTS_PATH}")
    summarize(results)


if __name__ == "__main__":
    evaluate()