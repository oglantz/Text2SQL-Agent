"""Error taxonomy + plots (Phase 3).

Categorize failures into:
    - schema-linking errors   — wrong table/column chosen
    - value-grounding errors  — wrong literal (what sample_values should prevent)
    - logic errors            — wrong join, aggregation, or filter structure
    - looping/no-submit       — ran out of steps

Then show how the distribution shifts baseline -> off-policy -> on-policy.
Optionally compute BIRD's Valid Efficiency Score (VES).

Placeholder — implement in Phase 3.
"""


def classify_failure(record):
    raise NotImplementedError


def main():
    raise NotImplementedError


if __name__ == "__main__":
    main()
