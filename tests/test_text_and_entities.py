import pandas as pd

from complaintiq.data import clean_text
from complaintiq.entities import brace_amounts, mentions_alias, spot_check


def test_clean_text_removes_redactions_and_braces():
    raw = "On XX/XX/XXXX I sent {$1,200.00} to XXXX XXXX via Zelle."
    assert clean_text(raw) == "On I sent $1,200.00 to via Zelle."


def test_brace_amounts_parse_cfpb_format():
    assert brace_amounts("charged {$1,250.50} and then {$20.00} again") == [1250.5, 20.0]


def test_alias_matching_is_whole_word():
    assert mentions_alias("I used Cash App to pay", ["cash app"])
    assert mentions_alias("chase bank closed it", ["chase"])
    assert not mentions_alias("they kept chasing me", ["chase"])


def test_spot_check_scores_money_and_org_against_references():
    df = pd.DataFrame(
        {
            "complaint_id": ["1"],
            "company": ["Coinbase, Inc."],
            "narrative": ["Coinbase took {$500.00} from my account on XX/XX/2024."],
        }
    )
    ents = [[("ORG", "Coinbase"), ("MONEY", "500.00"), ("MONEY", "$9"), ("DATE", "XX/XX/2024")]]
    per_doc, summary = spot_check(df, ents)
    row = per_doc.iloc[0]
    assert row.money_amounts_found == 1 and row.money_spurious == 1
    assert row.company_in_orgs and row.n_date_redacted == 1
    assert summary.set_index("check").loc["ORG: share of those where an ORG entity names the company", "value"] == 1.0
