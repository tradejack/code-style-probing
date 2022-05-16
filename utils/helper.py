def read_file_to_string(filename):
    f = open(filename, "rb")
    s = ""
    try:
        s = f.read()
    except:
        print(filename)
    f.close()
    return s.decode(errors="replace")


def calculate_ratio(numerator, denominator, round_val=6):
    return round(numerator / denominator, round_val) if denominator > 0 else 0
