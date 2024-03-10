"""
Common utility functions in this file
"""


def make_bool(data: str):
    return data is not None and (data.lower() == "true" or data == "")
