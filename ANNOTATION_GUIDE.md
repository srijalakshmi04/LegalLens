# Legal Lens — Human Relevance Benchmark Annotation Guidelines (ANNOTATION_GUIDE.md)

## 1. Overview & Objective

This guideline provides clear instructions for human legal domain reviewers annotating candidate Supreme Court judgments retrieved by Legal Lens for **50 benchmark legal queries**.

The objective of annotation is to establish a **ground-truth relevance benchmark** for evaluating Information Retrieval (IR) algorithms (Precision@K, Recall@K, MRR, NDCG@10) and training second-stage supervised relevance reranking models.

---

## 2. Ordinal Relevance Scale (0 – 3)

Annotators must evaluate each `(Query, Retrieved Judgment)` candidate pair and select exactly one ordinal relevance level:

| Rating | Category | Technical Definition | Domain Annotation Criteria |
| :---: | :--- | :--- | :--- |
| **3** | **Highly Relevant** | Exact factual and legal match | The candidate judgment directly addresses the core legal issue, statutory provisions, and factual context described in the query. Serving as direct precedent. |
| **2** | **Relevant** | Substantial legal overlap | The candidate judgment shares the same primary area of law, statutory section, or legal doctrine, but may differ slightly in specific factual background. |
| **1** | **Marginally Relevant** | Weak conceptual similarity | The candidate judgment touches upon a related legal field or secondary statutory provision mentioned in the query, but is not primarily about the query's core issue. |
| **0** | **Irrelevant** | Unrelated area of law | The candidate judgment belongs to an entirely different area of law or shares only incidental keywords without legal relationship. |

---

## 3. Explicit Annotation Rules

### WHAT TO EVALUATE:
1. **Factual Similarity**: Do the case facts reflect the legal situation described in the query?
2. **Legal Issue Overlap**: Does the judgment resolve the legal question raised by the query?
3. **Statutory/Constitutional Overlap**: Are the same Articles (e.g. Article 21) or Sections (e.g. Section 438 CrPC, Section 138 NI Act) central to the decision?
4. **Judicial Reasoning**: Is the court's reasoning applicable to the query topic?

### WHAT NOT TO EVALUATE:
1. **Do NOT judge win probability**: Annotators must NOT evaluate whether a litigant would win or lose.
2. **Do NOT judge judicial correctness**: Annotators must NOT evaluate whether the Supreme Court judges made the correct decision.
3. **Do NOT judge good-law / overruled status**: Precedent status (overruled/distinguished) is not evaluated during retrieval relevance scoring.

---

## 4. How to Use the Annotation Tool

1. Access **`http://127.0.0.1:8000/annotate/`** in your browser.
2. Review the **Search Query**, **Legal Category**, and **Candidate Judgment Details** (Title, Year, Citation, Statutory References, and Evidence Passage).
3. Select the appropriate rating ($0, 1, 2, 3$).
4. Click **`Save Annotation`**. The progress indicator (`Annotated: X / 50`) will update automatically in real-time.
5. Use **`Next`**, **`Prev`**, or **`Skip`** to navigate between items.
