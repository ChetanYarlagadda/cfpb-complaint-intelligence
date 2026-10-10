import numpy as np
import pytest

from complaintiq.dedup import estimated_jaccard, minhash_signatures, near_duplicate_groups, tokens

TEMPLATE = (
    "I am filing a complaint against the payment app due to inadequate customer service and unfair practices "
    "which violate consumer protection law. The company failed to take timely and effective measures to prevent "
    "fraud on their platform and their dispute process did not follow the error resolution requirements."
)


def test_tokens_drop_redaction_marks():
    assert tokens("On XX/XX/XXXX I sent {$200.00} to XXXX") == ["on", "i", "sent", "200", "00", "to"]


def test_template_variants_group_and_unrelated_texts_do_not():
    texts = [
        TEMPLATE,
        "Date of incident was XX/XX/2024. " + TEMPLATE,  # personalised prefix
        TEMPLATE.replace("payment app", "Cash App"),  # one-word edit
        "My crypto exchange froze my account for three weeks and support never answered my emails.",
        "The international wire never arrived and the remittance company refuses to refund the fee.",
    ]
    groups = near_duplicate_groups(minhash_signatures(texts), bands=32, threshold=0.5)
    assert groups[0] == groups[1] == groups[2]
    assert len({groups[0], groups[3], groups[4]}) == 3
    assert groups[0] == 0  # ids follow first appearance


def test_minhash_estimates_jaccard():
    rng = np.random.default_rng(0)
    vocab = [f"w{i}" for i in range(80)]
    a = list(rng.choice(vocab, 300))
    b = a[120:] + list(rng.choice(vocab, 120))

    def shingle_set(words: list[str]) -> set[tuple[str, ...]]:
        return {tuple(words[i : i + 5]) for i in range(len(words) - 4)}

    exact = len(shingle_set(a) & shingle_set(b)) / len(shingle_set(a) | shingle_set(b))
    sig = minhash_signatures([" ".join(a), " ".join(b)], num_perm=256)
    assert estimated_jaccard(sig[0], sig[1]) == pytest.approx(exact, abs=0.08)


def test_bands_must_divide_signature():
    with pytest.raises(ValueError):
        near_duplicate_groups(np.zeros((3, 10), dtype=np.uint64), bands=3)
