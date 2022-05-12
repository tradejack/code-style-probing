def function_name(par_1, parTwo, camelCase):
    """
    docstring time 1
    """
    """
	docstring time 2
	"""
    # single line comment
    var_1 = 42  # cool and awesome comment
    print("hello world!")  # comment too
    return


def function_name2(par_1, parTwo, camelCase):
    """
    docstring time 1
    """
    """
	docstring time 2
	"""
    # single line comment
    var_1 = 42  # cool and awesome comment
    print("hello world!")  # comment too
    return


def input_code(nums):
    """_summary_"""
    results = []
    for num in nums:
        if num > 0:
            results.append(num + 1)
    return results


def output_code(nums):
    """_summary_

    Args:
        nums (_type_): _description_

    Returns:
        _type_: _description_
    """
    return [num + 1 for num in nums if num > 0]
