Phase 0 — Audit the existing project
15–25 min

- Inspect the repository structure, Git status, model artifacts, and training/evaluation scripts.
- Verify the adapter directory, configuration, checkpoint selection, and referenced report files.
- Identify the actual 50-task benchmark, category breakdown, overlap audit, and evaluation implementation.
- Check whether the adapter is present locally, committed, or needs to be recovered from an existing artifact location.
- Confirm Hugging Face authentication and Space availability without exposing credentials.
Gate: The agent can identify the exact adapter to publish and the exact benchmark artifacts supporting the reported result.

Phase 1 — Reproducibility and artifact validation
30–45 min

- Verify adapter files, base model identifier, tokenizer, LoRA target modules, and loading configuration.
- Run a small local smoke test using the existing environment, without rerunning training.
- Re-run the existing 50-task evaluation if practical; otherwise verify the saved evaluation records and report.
- Check that reported percentages agree with the raw results and category counts.
- Verify that benchmark tasks are not accidentally used as training examples, to the extent supported by available artifacts.
Gate: The model loads, generates output, and the benchmark evidence is internally consistent—or unresolved discrepancies are explicitly documented.

Phase 2 — Prepare the public model artifacts
20–40 min

- Prepare a model card describing Phi-2, QLoRA, the adapter, intended use, limitations, training configuration, and evaluation.
- Include a clear base-versus-adapted comparison and link to the GitHub training repository.
- Upload only the selected adapter and necessary configuration files to the Hugging Face model repository.
- Check base-model licensing and applicable dataset redistribution restrictions before publishing.
Gate: The public model repository contains the intended artifacts and its documented claims match the evidence.

Phase 3 — Build the Gradio playground
30–60 min

- Create a small Gradio interface with prompt input, output comparison, generation settings, and error handling.
- Load the base model and adapter according to the verified inference configuration.
- Avoid loading two full model copies simultaneously unless measured memory requirements justify it.
- Provide example tasks covering different benchmark categories.
- Add a benchmark-results panel based on verified results, not hardcoded unverified claims.
Gate: The interface builds and at least one base-versus-adapted comparison succeeds locally.

Phase 4 — Deploy the Space
30–60 min

- Create or connect the Gradio Space to the intended GitHub/Hugging Face artifacts.
- Choose hardware based on measured inference needs and available budget.
- Configure model identifiers and secrets using Space settings.
- Inspect build logs, model downloads, memory use, and startup behavior.
- Verify the public URL and actual inference behavior.
Gate: A real public Space starts successfully and serves the expected functionality.

Phase 5 — Final acceptance and documentation
20–30 min

- Test base and adapter loading, representative prompts, generation limits, and graceful failures.
- Verify the published model card, model artifact links, and demo URL.
- Update the GitHub README with deployment instructions, benchmark methodology, hardware constraints, and the live demo link.
- Generate DEPLOYMENT_REPORT.md with actual test outcomes and unresolved issues.
- Commit and push only verified changes.
Gate: The demo, model artifacts, and repository documentation agree with each other.