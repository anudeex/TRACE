Code for our paper - _"Who Wrote the Book? Detecting and Attributing LLM Ghostwriters"_



The GhostWriteBench dataset (as a zip, with three splits as per the paper) can be found in [/ghostwritebench](./ghostwritebench) folder.

TRACE basically involves:
1. Generating token-level scores using an evaluator language model ([generate-token-scores-TRACE.py](./src/generate-token-scores-TRACE.py)).</li>
2. Then, using these scores to compute fingerprints during inference.</li>
   * Rank-based - [test-rank-based-TRACE.py](./src/test-rank-based-TRACE.py)</li>
   * Entropy-based - [test-entropy-based-TRACE.py](./src/test-entropy-based-TRACE.py)</li>
 
