"""Evaluate rag_engine.py against two PDFs. Run: uv run evaluate.py.

Answer matching is deterministic and requires every configured answer group.
It is stricter than simple any-phrase matching, but manual review is still needed.
"""

import json
import re
import time
from pathlib import Path

from rag_engine import (
    get_chroma_collection,
    get_embeddings,
    query_rag,
    ingest_document,
    DISTANCE_THRESHOLD,
)

# Keep these PDFs in the same folder as this script, or replace with full paths.
PDF_PATHS = [
    "sample_pdfs/devops_ci_cd_architecture.pdf",
    "sample_pdfs/pediatric_bone_age_clinical.pdf",
]
RESULTS_PATH = "naive_rag_eval_results.json"
SYSTEM_K = 4
EVAL_K = 10
REFUSAL = "I could not find this in the PDF."

# Each answerable case uses required_groups: every group must be satisfied,
# while phrases inside one group are alternatives. This avoids passing a
# multi-part question because only one item appeared in the answer.
# Empty required_groups means the question should be refused.
TEST_CASES = [
    {"question": "Which container runtimes are mentioned as OCI-compliant?",
     "answers": ["Docker", "containerd"],
     "required_groups": [["Docker"], ["containerd"]], "source": "devops_ci_cd_architecture.pdf"},
    {"question": "Which Linux mechanisms provide container isolation and resource control?",
     "answers": ["Linux namespaces", "cgroups v2"],
     "required_groups": [["Linux namespaces"], ["cgroups v2"]], "source": "devops_ci_cd_architecture.pdf"},
    {"question": "Which service is mentioned for internal DNS and service discovery?",
     "answers": ["CoreDNS"], "source": "devops_ci_cd_architecture.pdf"},
    {"question": "Which tools are mentioned for GitOps reconciliation?",
     "answers": ["ArgoCD", "Flux"],
     "required_groups": [["ArgoCD", "Argo CD"], ["Flux"]], "source": "devops_ci_cd_architecture.pdf"},
    {"question": "Which tool is mentioned for container vulnerability scanning?",
     "answers": ["Trivy"], "source": "devops_ci_cd_architecture.pdf"},
    {"question": "What does the Horizontal Pod Autoscaler monitor?",
     "answers": ["CPU utilization", "request latencies"],
     "required_groups": [["CPU utilization", "CPU usage", "CPU metrics"], ["request latencies", "request latency"]],
     "source": "devops_ci_cd_architecture.pdf"},
    {"question": "Which tool dynamically provisions cloud instances?",
     "answers": ["Karpenter"], "source": "devops_ci_cd_architecture.pdf"},
    {"question": "What are the three pillars of distributed observability?",
     "answers": ["metrics", "traces", "logs"],
     "required_groups": [["metrics"], ["traces"], ["logs"]], "source": "devops_ci_cd_architecture.pdf"},
    {"question": "What do RTO and RPO stand for?",
     "answers": ["Recovery Time Objective", "Recovery Point Objective"],
     "required_groups": [["Recovery Time Objective", "RTO"], ["Recovery Point Objective", "RPO"]],
     "source": "devops_ci_cd_architecture.pdf"},
    {"question": "Which hormones regulate skeletal growth according to the document?",
     "answers": ["Growth Hormone", "IGF-1", "thyroid hormones", "sex steroids"],
     "required_groups": [["Growth Hormone", "GH"], ["IGF-1", "IGF1"], ["thyroid hormones", "thyroid hormone"], ["sex steroids", "sex steroid"]],
     "source": "pediatric_bone_age_clinical.pdf"},
    {"question": "Which hand and radiographic view are used for standard skeletal maturity evaluation?",
     "answers": ["left hand", "posterior-anterior", "PA radiograph"],
     "required_groups": [["left hand"], ["posterior-anterior", "PA view", "PA radiograph"]],
     "source": "pediatric_bone_age_clinical.pdf"},
    {"question": "How many carpal bones are mentioned?",
     "answers": ["Eight", "8"],
     "required_groups": [["Eight", "8"]], "source": "pediatric_bone_age_clinical.pdf"},
    {"question": "What are the two primary clinical bone-age assessment methodologies?",
     "answers": ["Greulich-Pyle", "Tanner-Whitehouse"],
     "required_groups": [["Greulich-Pyle", "Greulich Pyle"], ["Tanner-Whitehouse", "Tanner Whitehouse"]],
     "source": "pediatric_bone_age_clinical.pdf"},
    {"question": "How many anatomical regions of interest does Tanner-Whitehouse analyze?",
     "answers": ["twenty", "20"],
     "required_groups": [["twenty", "20"]], "source": "pediatric_bone_age_clinical.pdf"},
    {"question": "Which explainability method generates visual heatmaps?",
     "answers": ["Grad-CAM", "Gradient-weighted Class Activation Mapping", "Gradient weighted Class Activation Mapping"],
     "required_groups": [["Grad-CAM", "Gradient-weighted Class Activation Mapping", "Gradient weighted Class Activation Mapping"]],
     "source": "pediatric_bone_age_clinical.pdf"},
    {"question": "Which systems are mentioned for hospital radiology integration?",
     "answers": ["Hospital Information Systems", "HIS", "PACS"],
     "required_groups": [["Hospital Information Systems", "HIS"], ["PACS"]],
     "source": "pediatric_bone_age_clinical.pdf"},
    {"question": "Which tool is mentioned for metrics monitoring?",
     "answers": ["Prometheus"], "source": "All"},
    {"question": "Which explainability method is mentioned in the supplied documents?",
     "answers": ["Grad-CAM", "Gradient-weighted Class Activation Mapping"],
     "required_groups": [["Grad-CAM", "Gradient-weighted Class Activation Mapping", "Gradient weighted Class Activation Mapping"]], "source": "All"},
    {"question": "Who won the football World Cup in 2010?",
     "answers": [], "source": "All"},
    {"question": "What is the current price of an Amazon EC2 instance?",
     "answers": [], "source": "All"},
    {"question": "What hospital treated the pediatric patient?",
     "answers": [], "source": "All"},
]


def normalize(text):
    """Normalize punctuation to spaces, preserving word boundaries (e.g. 8 != 18)."""
    text = re.sub(r"[^a-z0-9]+", " ", str(text).lower())
    return " ".join(text.split())


def required_groups(case):
    """Return required groups; phrases within each group are interchangeable."""
    if "required_groups" in case:
        return case["required_groups"]
    answers = case.get("answers", [])
    return [answers] if answers else []


def group_matches(text, alternatives):
    """Return True if any alternative phrase in the group is present in the text."""
    normalized_text = f" {normalize(text)} "
    for phrase in alternatives:
        normalized_phrase = normalize(phrase)
        if normalized_phrase and f" {normalized_phrase} " in normalized_text:
            return True
    return False


def matches_expected(text, case):
    """All required groups must be present; variants inside a group are OR."""
    groups = required_groups(case)
    return bool(groups) and all(group_matches(text, group) for group in groups)


def groups_present_in_chunks(chunks, case):
    """Check all required groups across the combined text of retrieved chunks."""
    combined = " ".join(str(chunk) for chunk in chunks)
    return matches_expected(combined, case)


def is_refusal(answer):
    """Recognize common explicit refusals without requiring one exact sentence."""
    text = normalize(answer)
    refusal_patterns = [
        "i could not find this in the pdf",
        "i could not find this information in the pdf",
        "not provided in the pdf",
        "not mentioned in the pdf",
        "does not contain this information",
        "cannot answer from the provided documents",
        "unable to answer from the provided documents",
    ]
    return any(pattern in text for pattern in refusal_patterns)

def ensure_index(collection):
    """Index each PDF only if its filename is not already present in Chroma."""
    for pdf_path in PDF_PATHS:
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(
                f"PDF not found: {path}. Put it beside evaluate.py or update PDF_PATHS."
            )

        filename = path.name
        existing = collection.get(where={"source": filename}, include=["metadatas"])
        if existing and existing.get("ids"):
            print(f"[Index] Already indexed: {filename}")
        else:
            with path.open("rb") as file:
                count = ingest_document(file.read(), filename)
            print(f"[Index] Ingested {filename}: {count} chunks")


def load_corpus(collection):
    """Returns all indexed chunks grouped by PDF."""
    data = collection.get(include=["documents", "metadatas"])
    corpus = {}
    for document, metadata in zip(data.get("documents") or [],
                                  data.get("metadatas") or []):
        corpus.setdefault(metadata.get("source", "Unknown"), []).append(document)
    return corpus


def retrieve_top(collection, question, source, k):
    """Embeds the question and retrieves the top k chunks with their distances"""
    vector = get_embeddings([question])[0]
    where = None if source == "All" else {"source": source}
    result = collection.query(
        query_embeddings=[vector],
        n_results=k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    docs = (result.get("documents") or [[]])[0] or []
    metas = (result.get("metadatas") or [[]])[0] or []
    distances = (result.get("distances") or [[]])[0] or []
    return [
        {"text": doc, "source": meta.get("source", "Unknown"), "distance": dist}
        for doc, meta, dist in zip(docs, metas, distances)
    ]


def evaluate_case(case, collection, corpus):
    question = case["question"]
    answers = case["answers"]
    source = case.get("source", "All")
    answerable = bool(required_groups(case))

    retrieved = retrieve_top(collection, question, source, EVAL_K)
    top_system = retrieved[:SYSTEM_K]
    filtered = [chunk for chunk in top_system
                if chunk["distance"] <= DISTANCE_THRESHOLD]

    scoped_corpus = (corpus.get(source, []) if source != "All"
                     else [chunk for chunks in corpus.values() for chunk in chunks])
    in_index = (groups_present_in_chunks(scoped_corpus, case)
                if answerable else None)

    rank, answer_distance = 0, None
    if answerable:
        for position in range(1, len(retrieved) + 1):
            if groups_present_in_chunks([chunk["text"] for chunk in retrieved[:position]], case):
                rank = position
                answer_distance = retrieved[position - 1]["distance"]
                break

    start = time.perf_counter()
    answer, sources = query_rag(question, source)
    latency = time.perf_counter() - start
    refused = is_refusal(answer)
    correct = matches_expected(answer, case) if answerable else refused
    in_filtered_context = (
        groups_present_in_chunks([chunk["text"] for chunk in filtered], case)
        if answerable else None
    )

    return {
        "question": question,
        "reference_answers": answers,
        "required_answer_groups": required_groups(case),
        "source_filter": source,
        "answerable": answerable,
        "generated_answer": answer,
        "sources_returned": sources,
        "correct": correct,
        "refused": refused,
        "latency_seconds": round(latency, 3),
        "answer_string_in_index": in_index,
        "answer_rank_at_eval_k": rank,
        "hit_at_system_k": bool(answerable and 0 < rank <= SYSTEM_K),
        "hit_at_eval_k": bool(answerable and rank > 0),
        "reciprocal_rank": 1 / rank if rank else 0.0,
        "answer_string_in_filtered_context": in_filtered_context,
        "chunks_retrieved_at_eval_k": len(retrieved),
        "chunks_passed_distance_filter": len(filtered),
        "top1_distance": retrieved[0]["distance"] if retrieved else None,
        "answer_chunk_distance": answer_distance,
    }


def pct(numerator, denominator):
    return f"{numerator / denominator:.1%}" if denominator else "n/a"


def summarize(results):
    answerable = [r for r in results if r["answerable"]]
    unanswerable = [r for r in results if not r["answerable"]]
    total = len(results)

    print(f"\n{'=' * 58}\nRESULTS: {total} questions "
          f"({len(answerable)} answerable, {len(unanswerable)} unanswerable)\n{'=' * 58}")
    print("\n[Retrieval: answerable questions]")
    print("All required answer elements present in index:", pct(sum(r["answer_string_in_index"] for r in answerable),
                                         len(answerable)))
    print(f"Recall@{SYSTEM_K}:", pct(sum(r["hit_at_system_k"] for r in answerable),
                                   len(answerable)))
    print(f"Recall@{EVAL_K}:", pct(sum(r["hit_at_eval_k"] for r in answerable),
                                   len(answerable)))
    mrr = (sum(r["reciprocal_rank"] for r in answerable) / len(answerable)
           if answerable else None)
    print("MRR:", f"{mrr:.3f}" if mrr is not None else "n/a")
    print("All required answer elements survive distance filter:",
          pct(sum(r["answer_string_in_filtered_context"] for r in answerable),
              len(answerable)))

    print("\n[Generation]")
    print("Answerable-question accuracy:",
          pct(sum(r["correct"] for r in answerable), len(answerable)))
    print("False-refusal rate on answerable questions:",
          pct(sum(r["refused"] for r in answerable), len(answerable)))

    if unanswerable:
        print("\n[Abstention: unanswerable questions]")
        print("Correct refusal rate:",
              pct(sum(r["correct"] for r in unanswerable), len(unanswerable)))
        print("Responses without refusal phrase:",
              pct(sum(not r["refused"] for r in unanswerable), len(unanswerable)))

    print("\n[Overall]")
    print("Overall accuracy:", pct(sum(r["correct"] for r in results), total))
    print("Average latency:",
          f"{sum(r['latency_seconds'] for r in results) / total:.2f}s" if total else "n/a")

    failures = [r for r in results if not r["correct"]]
    print(f"\n[Failed questions: {len(failures)}]")
    for item in failures:
        print(f"- {item['question']}")
        print(f"  Expected answer groups: {item.get('required_answer_groups') or 'refusal'}")
        print(f"  Answer: {item['generated_answer']}")
        print(f"  All required answer elements present in index: {item['answer_string_in_index']}")
        print(f"  Rank in top {EVAL_K}: {item['answer_rank_at_eval_k']}")
        print(f"  All required elements in filtered context: {item['answer_string_in_filtered_context']}")


def evaluate():
    collection = get_chroma_collection()
    ensure_index(collection)
    corpus = load_corpus(collection)

    print("\n[Index]")
    for source, chunks in corpus.items():
        print(f"  {source}: {len(chunks)} chunks")
    print(f"\nSystem k={SYSTEM_K}; diagnostic k={EVAL_K}; "
          f"distance threshold={DISTANCE_THRESHOLD}")

    results = []
    for number, case in enumerate(TEST_CASES, start=1):
        print(f"\n[{number}/{len(TEST_CASES)}] {case['question']}")
        try:
            result = evaluate_case(case, collection, corpus)
            results.append(result)
            print(f"  Correct: {result['correct']} | Latency: {result['latency_seconds']}s")
        except Exception as error:
            print(f"  ERROR: {error}")

    if not results:
        print("No evaluation results were produced.")
        return

    with open(RESULTS_PATH, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)
    print(f"\nDetailed results saved to {RESULTS_PATH}")
    summarize(results)


if __name__ == "__main__":
    evaluate()
