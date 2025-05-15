import pytest
from pathlib import Path
from identify_faces import get_suffix_num


@pytest.mark.parametrize("file_path, dup_files, expected", [
    (Path("a.jpg"), {Path("x.jpg"): 0}, ""),
    (Path("a.jpg"), {Path("a.jpg"): 0}, ""),
    (Path("a.jpg"), {Path("a.jpg"): 1}, "(1)"),
    (Path("a.jpg"), {Path("a.jpg"): 2}, "(2)"),
])
def test_get_suffix_num(file_path, dup_files, expected):
    assert get_suffix_num(file_path, dup_files) == expected
