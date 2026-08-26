# Vendored IFBench evaluator

The evaluator sources in this directory are vendored from the official
[`allenai/IFBench`](https://github.com/allenai/IFBench) repository at commit
`db69a6f05689830b0068b8f1529ebcfd2f3b164c` and retain their Apache-2.0
copyright headers.

Local changes are intentionally limited to package-relative imports, removal
of import-time NLTK downloads and filesystem writes, and a fixed langdetect seed
for repeatable scoring. Runtime evaluation requires
`punkt`, `punkt_tab`, `stopwords`, and `averaged_perceptron_tagger_eng` to be
staged ahead of time and discoverable through `NLTK_DATA`.
