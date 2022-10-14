# eval on predictions
import ast
import re
import difflib
from termcolor import colored

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from evaluator.CodeBLEU.calc_code_bleu import get_codebleu
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from utils.regex_parse import comment


def exclude_same_io(df):
    # excluding those input exactly same as the output
    exact_match_bool = df["inputs"] == df["labels"]
    df = df.drop(df[exact_match_bool].index)
    return df


# parsable eval
def is_parsable(input_code):
    try:
        ast.parse(input_code)
    except SyntaxError:
        return False
    except Exception as e:
        print(input_code)
        print(e)
        return False
    return True


def evaluate_codebleu(
    pred_filename,
    weights="0.25,0.25,0.25,0.25",
    replaced_df=None,
    dropna=False,
    is_exclude_same_io=False,
):
    pred_df = None
    if replaced_df is not None:
        pred_df = replaced_df
    else:
        pred_df = pd.read_csv(pred_filename)
    if dropna:
        pred_df = pred_df.dropna()
    if is_exclude_same_io:
        pred_df = exclude_same_io(pred_df)
    # a list of gold codes (which is just some variants of the same code, we can use every code of different styles)
    refs = [pred_df["labels"]]
    # the prediction code
    hyp = pred_df["preds"]
    score = get_codebleu(refs, hyp, "python", weights)
    return score


def get_docstring(text):
    regex_docstr = "^\s*'{3}([\s\S]*?)'{3}|^\s*\"{3}([\s\S]*?)\"{3}"
    docstr_matches = re.findall(regex_docstr, text, re.M | re.S)
    docstrs = []
    for match in docstr_matches:
        docstr_a, docstr_b = match
        if docstr_a:
            docstrs += [docstr_a]
        else:
            docstrs += [docstr_b]
    return docstrs


def print_split_line(s):
    print(f"\n====================={s.upper()}=====================\n")


def tokenize(s):
    return re.split("\s+", s)


def get_diff_list(str_1, str_2):
    s1 = tokenize(str_1)
    s2 = tokenize(str_2)

    matcher = difflib.SequenceMatcher(a=s1, b=s2)

    diff_blocks_a = []
    diff_blocks_b = []

    prev_match = None
    for idx, match in enumerate(matcher.get_matching_blocks()):

        if idx == 0:
            prev_match = match
            if match.a != 0:
                start_idx_a = 0
                end_idx_a = match.a
                diff_blocks_a += s1[start_idx_a:end_idx_a]
            if match.b != 0:
                start_idx_b = 0
                end_idx_b = match.b
                diff_blocks_b += s2[start_idx_b:end_idx_b]
            continue

        start_idx_a = prev_match.a + prev_match.size
        end_idx_a = match.a

        start_idx_b = prev_match.b + prev_match.size
        end_idx_b = match.b

        diff_list_a = s1[start_idx_a:end_idx_a]
        diff_list_b = s2[start_idx_b:end_idx_b]
        if len(diff_list_a):
            diff_blocks_a += diff_list_a
        if len(diff_list_b):
            diff_blocks_b += diff_list_b

        prev_match = match
    return diff_blocks_a, diff_blocks_b


def get_diff_str(input_str, output_str):
    return " ".join(get_diff_list(input_str, output_str)[1])


def remove_nl_prompt(script):
    return re.sub("<nl>.*<\/nl>", "", script)


def post_process_parsed_script(script):
    
    script = script.replace("\\n", "\n")
    script = script.replace("\\t", "\t")
    script = script.replace("\\s", "\s")
    script = script.replace("\\'", "'")
    
    # script = script.replace('"""', "")
    # script = script.replace("\\n", "")
    # script = re.sub(r"\"\"\"([.\s\S]*)\n\"\"\"", "\'\\1\'", script)
    # script = re.sub(r"\"\"\"([.\s\S]*)\"\"\"", "\'\\1\'", script)
    # script = re.sub(r"if\s\((.*)\):", "if \\1:", script)
    # script = re.sub(r"assert\s\((.*)\)", "assert \\1", script)
    # script = re.sub(r"return\s\((.*)\)", "return \\1", script)
    # script = re.sub(r"\(-\s([0-9]*)\)", "-\\1", script)
    

    script = " ".join(script.split())
    script = script.replace("'", "")
    script = script.replace('"', "")
    script = script.replace('(', "")
    script = script.replace(')', "")
    script = re.sub(r"-\s([0-9]*)", "-\\1", script)
    
    return script

def is_valid_transfer(input_code, label):
    diff = get_diff_str(post_process_parsed_script(input_code), post_process_parsed_script(label))
    return len(diff.strip()) > 0

def get_valid_pred_df(df):
    return df[df.apply(lambda row: is_valid_transfer(row["inputs"], row["labels"]), axis=1)]

def evaluate_pred_df(pred_df, target_feats, clean_diff=False, is_nl=False, parse_test=True):
    smoothing_function = SmoothingFunction()

    inputs = pred_df["inputs"].to_numpy()
    labels = pred_df["labels"].to_numpy()
    preds = pred_df["preds"].to_numpy()

    if is_nl:
        inputs = [remove_nl_prompt(input_script) for input_script in inputs]

    code_scores = []
    diff_bleu_scores = []

    # if comment, need to extract comment
    gold_comments = []
    pred_comments = []
    comment_text_scores = []

    # if docstring, need to extract docstring
    gold_docstrings = []
    pred_docstrings = []
    docstr_text_scores = []

    # if parse test
    is_parsables = []

    pred_diffs = []
    gold_diffs = []

    total_len = preds.shape[0]

    for idx in tqdm(range(total_len)):
        input_code = inputs[idx]
        gold = labels[idx]
        pred = preds[idx]

        refs = [[gold]]
        hyp = [pred]

        # get code bleu score
        code_score = get_codebleu(refs, hyp, "python", "0.25,0.25,0.25,0.25")

        if "docstring" in target_feats:
            gold_docstr = get_docstring(gold)
            pred_docstr = get_docstring(pred)
            gold_docstr_text = "\n".join(gold_docstr)
            pred_docstr_text = "\n".join(pred_docstr)
            docstr_text_score = 0
            if len(pred_docstr_text.split()) > 0:
                docstr_text_score = sentence_bleu(
                    [gold_docstr_text.split()],
                    pred_docstr_text.split(),
                    auto_reweigh=True,
                )

            gold_docstrings += [gold_docstr]
            pred_docstrings += [pred_docstr]
            docstr_text_scores += [docstr_text_score]

        if "comment" in target_feats:
            gold_comment = comment(gold)
            pred_comment = comment(pred)
            gold_comment_text = "\n".join(gold_comment)
            pred_comment_text = "\n".join(pred_comment)
            comment_text_score = 0
            if len(pred_comment_text.split()) > 0:
                comment_text_score = sentence_bleu(
                    [gold_comment_text.split()],
                    pred_comment_text.split(),
                    auto_reweigh=True,
                )

            gold_comments += [gold_comment]
            pred_comments += [pred_comment]
            comment_text_scores += [comment_text_score]



        input_code_for_diff = input_code
        gold_for_diff = gold
        pred_for_diff = pred

        # get the diff bleu score
        if clean_diff:
            input_code_for_diff = post_process_parsed_script(input_code)
            gold_for_diff = post_process_parsed_script(gold)
            pred_for_diff = post_process_parsed_script(pred)
        gold_diff_str = get_diff_str(input_code_for_diff, gold_for_diff)
        pred_diff_str = get_diff_str(input_code_for_diff, pred_for_diff)

        pred_diffs += [pred_diff_str]
        gold_diffs += [gold_diff_str]

        diff_bleu_score = 0
        if len(pred_diff_str.split()) > 0:
            diff_bleu_score = sentence_bleu(
                [gold_diff_str.split()], pred_diff_str.split(), smoothing_function=smoothing_function.method1, auto_reweigh=True
            )

        code_scores += [code_score]
        diff_bleu_scores += [diff_bleu_score]
        if parse_test:
            is_parsables += [is_parsable(pred)]

    code_bleus = np.array([s["code_bleu"] for s in code_scores])

    report = {
        "inputs": inputs.tolist(),
        "labels": labels.tolist(),
        "preds": preds.tolist(),
        "pred_diffs": pred_diffs,
        "gold_diffs": gold_diffs,
        "codebleu": code_scores,
        "codebleu_perfect": sum(code_bleus == 1).item() / total_len,
        "codebleu_above_90": sum(code_bleus >= 0.9).item() / total_len,
        "diff_bleu": diff_bleu_scores,
        "diff_bleu_avg": np.mean(diff_bleu_scores).item(),
        "diff_bleu_perfect": sum(np.array(diff_bleu_scores) == 1).item() / total_len,
        "diff_bleu_above_90": sum(np.array(diff_bleu_scores) >= 0.9).item() / total_len,
    }

    if "docstring" in target_feats:
        report["gold_docstrings"] = gold_docstrings
        report["pred_docstrings"] = pred_docstrings
        report["docstr_text_scores"] = docstr_text_scores
        report["docstr_text_scores_avg"] = np.array(docstr_text_scores).mean().item()
        report["docstr_text_scores_perfect"] = (
            sum(np.array(docstr_text_scores) == 1).item() / total_len
        )
        report["docstr_text_scores_above_90"] = (
            sum(np.array(docstr_text_scores) >= 0.9).item() / total_len
        )

    if "comment" in target_feats:
        report["gold_comments"] = gold_comments
        report["pred_comments"] = pred_comments
        report["comment_text_scores"] = comment_text_scores
        report["comment_text_scores_avg"] = np.array(comment_text_scores).mean().item()
        report["comment_text_scores_perfect"] = (
            sum(np.array(comment_text_scores) == 1).item() / total_len
        )
        report["comment_text_scores_above_90"] = (
            sum(np.array(comment_text_scores) >= 0.9).item() / total_len
        )

    if parse_test:
        report["parse_test_accuracy"] = sum(np.array(is_parsables)).item() / total_len

    return report.copy()


def print_colored_diff(str_1, str_2):
    text_1 = ""
    text_2 = ""
    idx_1 = 0
    idx_2 = 0
    matcher = difflib.SequenceMatcher(a=str_1, b=str_2)
    for match in matcher.get_matching_blocks():
        diff_text_1 = ""
        if idx_1 < match.a:
            diff_text_1 += colored(str_1[idx_1 : match.a], "red")

        diff_text_2 = ""
        if idx_2 < match.b:
            diff_text_2 += colored(str_2[idx_2 : match.b], "red")

        match_text_1 = str_1[match.a : match.a + match.size]
        match_text_2 = str_2[match.b : match.b + match.size]

        idx_1 = match.a + match.size
        idx_2 = match.b + match.size

        text_1 += diff_text_1 + match_text_1
        text_2 += diff_text_2 + match_text_2

    if idx_1 < len(str_1):
        text_1 += colored(str_1[idx_1:], "red")

    if idx_2 < len(str_2):
        text_2 += colored(str_2[idx_2:], "red")
    return text_1, text_2


def lookup_examples(
    report,
    score_upper_bound,
    score_lower_bound,
    metric="diff_bleu",
    start_idx=0,
    count=10,
):
    total = len(report["inputs"])
    if count == "all":
        count = total
    current_count = 0
    for idx in range(total):
        if current_count == count:
            break
        if idx < start_idx:
            continue

        # checking upper bound
        if report[metric][idx] > score_upper_bound:
            continue
        # checking lower bound
        if report[metric][idx] < score_lower_bound:
            continue

        input_code = report["inputs"][idx]
        pred_code = report["preds"][idx]
        gold_code = report["labels"][idx]

        c_input, c_gold = print_colored_diff(input_code, gold_code)
        _, c_pred = print_colored_diff(input_code, pred_code)

        print_split_line(f"{idx}-input")
        print(c_input)
        print_split_line(f"{idx}-prediction")
        print(c_pred)
        print_split_line(f"{idx}-gold labels")
        print(c_gold)
        print_split_line(f"{idx}-{metric}")
        print(report[metric][idx])

        current_count += 1
        # break
