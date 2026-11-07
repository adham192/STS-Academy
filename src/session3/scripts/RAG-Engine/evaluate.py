"""
Controlled evaluation of three Speculative RAG subset-construction strategies.

Strategies:
  ranked      - original relevance-ranked adjacent overlapping pairs
  embedding   - chunk-to-chunk similarity-guided overlapping chains
  dual_anchor - keep the top question-relevant chunk from each cluster as an
                anchor, then vary its embedding-nearest companion

The same retrieval result and K-Means labels are reused across all strategies
for each question. The plain-RAG baseline is run once per question.
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
STRATEGIES = ("ranked", "embedding", "dual_anchor")
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
    """Run the same drafting and verifier stages for one subset strategy."""
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

    # Retrieval and clustering are performed ONCE, then reused by all strategies.
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
            "strategies": {},
        }

    labels = cluster(embeddings, n_clusters=DEFAULT_N_CLUSTERS)
    retrieval_rank = next(
        (rank for rank, doc in enumerate(docs, start=1) if matches_item(doc, item)),
        0,
    )

    strategy_results = {}
    for strategy in STRATEGIES:
        print(f"      Strategy: {strategy}", flush=True)
        started = time.perf_counter()
        subsets = build_subsets(
            docs,
            labels,
            n_subsets=N_SUBSETS,
            distances=distances,
            embeddings=embeddings,
            overlap_strategy=strategy,
        )
        construction_seconds = time.perf_counter() - started

        started = time.perf_counter()
        generated = run_drafts_and_verification(question, subsets)
        generation_and_verification_seconds = time.perf_counter() - started

        subset_hit = any(matches_item(doc, item) for subset in subsets for doc in subset)
        oracle = any(matches_item(draft_item["answer"], item)
                     for draft_item in generated["drafts"])
        final_correct = matches_item(generated["answer"], item)

        strategy_results[strategy] = {
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
        "strategies": strategy_results,
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
    print("THREE-STRATEGY SPECULATIVE RAG EVALUATION")
    print("=" * 76)
    print(f"Completed: {len(completed)} | Failed/no retrieval: {len(failed)}")
    if not completed:
        print("No completed questions; metrics cannot be calculated.")
        return

    baseline_accuracy = sum(r["baseline"]["correct"] for r in completed) / len(completed)
    print(f"\nPlain RAG baseline accuracy: {baseline_accuracy:.1%}")

    header = (
        f"{'Strategy':<14} {'Accuracy':>10} {'Oracle':>10} "
        f"{'Subset hit':>11} {'MRR':>8} {'Avg sec':>10}"
    )
    print("\n" + header)
    print("-" * len(header))

    for strategy in STRATEGIES:
        rows = [r for r in completed if strategy in r["strategies"]]
        if not rows:
            continue
        acc = sum(r["strategies"][strategy]["final_correct"] for r in rows) / len(rows)
        oracle = sum(r["strategies"][strategy]["oracle_draft_correct"] for r in rows) / len(rows)
        subset_hit = sum(r["strategies"][strategy]["subset_hit"] for r in rows) / len(rows)
        mrr = sum(r["retrieval"]["reciprocal_rank"] for r in rows) / len(rows)
        avg_sec = sum(
            r["strategies"][strategy]["subset_construction_seconds"]
            + r["strategies"][strategy]["draft_and_verification_seconds"]
            for r in rows
        ) / len(rows)
        print(
            f"{strategy:<14} {acc:>9.1%} {oracle:>9.1%} "
            f"{subset_hit:>10.1%} {mrr:>8.3f} {avg_sec:>9.1f}"
        )

    print("\nAccuracy by difficulty")
    for level in ("easy", "medium", "hard"):
        rows = [r for r in completed if r.get("level") == level]
        if not rows:
            continue
        print(f"  {level.title()} ({len(rows)} questions)")
        for strategy in STRATEGIES:
            valid = [r for r in rows if strategy in r["strategies"]]
            if valid:
                acc = sum(r["strategies"][strategy]["final_correct"] for r in valid) / len(valid)
                print(f"    {strategy:<12}: {acc:.1%}")
        print(
            f"    {'baseline':<12}: "
            f"{sum(r['baseline']['correct'] for r in rows) / len(rows):.1%}"
        )

    print("\nInterpretation notes:")
    print("- Oracle = at least one generated draft matched the ground-truth phrase.")
    print("- Subset hit = at least one subset retained a chunk containing that phrase.")
    print("- MRR is a shared retrieval metric, so it should be identical across strategies.")
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
    print(f"Strategies: {', '.join(STRATEGIES)}")
    print(f"Retrieval k={K}; clusters={DEFAULT_N_CLUSTERS}; subsets={N_SUBSETS}")
    print(
        f"Gemini verification calls: approximately {len(test_items) * len(STRATEGIES)}. "
        "At a 13-second minimum interval, pacing alone may take "
        f"{(len(test_items) * len(STRATEGIES) - 1) * 13 / 60:.1f} minutes."
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
                "strategies": {},
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