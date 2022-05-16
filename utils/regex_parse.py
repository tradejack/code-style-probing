import re

CASING_REGEX_MAPPING = {
    "snake_case": r"[a-z0-9]*_*(_[a-z0-9]*)+_*",
    "lower_camel_case": r"^[a-z]+([A-Z][a-z0-9]+)+$",
    "upper_camel_case": r"^[A-Z][a-z]+([A-Z][a-z0-9]+)*$",
    "lower_case": r"^[a-z][a-z0-9]*$",
    "upper_case": r"^[A-Z][A-Z0-9]*$",
}
COMMENT_REGEX = r"#.*"

# TODO: need to add none-of-these
# TODO: need to make regex consistent
def casing(token):
    """
    returns the casing of the input text, as well as any subword splits
    Input: string
    Output: string, denoting
        snake_case, lower_camel_case, upper_camel_case, lower_case, upper_case, else other_case
    """
    for case_key, case_regex in CASING_REGEX_MAPPING.items():
        if re.match(case_regex, token):
            return case_key
    return "other_case"


def comment(code):
    return re.findall(COMMENT_REGEX, code)
