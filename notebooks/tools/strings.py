def strip_prefix(prefix, string):
    if string.startswith(prefix):
        return string[len(prefix):]
    return string