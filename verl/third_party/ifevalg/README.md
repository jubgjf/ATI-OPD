# Vendored IFEvalG verifier

The verifier sources in this directory are vendored from
[`allenai/open-instruct`](https://github.com/allenai/open-instruct), directory
`open_instruct/IFEvalG`, at commit
`e6c3044aa6201c8bfa4573c1b03142af599e730d`. The upstream files retain their
Apache-2.0 copyright headers.

The evaluator source changes are package-relative imports. The wrapper fixes the
langdetect seed for repeatable scoring and requires NLTK `punkt` and `punkt_tab`
data to be staged before the process starts; it never downloads resources or
writes to a user directory while scoring.
