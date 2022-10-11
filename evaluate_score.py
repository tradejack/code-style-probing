import json
from pathlib import Path

import typer
import pandas as pd

from utils.eval_utils import exclude_same_io, evaluate_pred_df, evaluate_codebleu

def main(
    pred_dir: Path,
    output_dir: Path,
    target_feat: str,
    is_nl_tokens_added: bool = False,
    clean_diff: bool=True
):
    pred_df = pd.read_csv(pred_dir)
    pred_df = exclude_same_io(pred_df)    
    target_feats=[target_feat]

    codebleu = evaluate_codebleu("", replaced_df=pred_df, weights='0.25,0.25,0.25,0.25')
    report = evaluate_pred_df(pred_df, target_feats, is_nl=is_nl_tokens_added, parse_test=True, clean_diff=clean_diff)

    scores = {}
    print("CodeBLEU: ", codebleu)
    for key, val in report.items():
        if type(val) != list:
            print(key, ":", val)
            scores[key] = val
    scores["codebleu"] = codebleu

    with open(output_dir, 'w') as f:
        json.dump(scores, f, indent=4)


if __name__ == "__main__":
    typer.run(main)