import pytest
from pathlib import Path
from identify_faces import find_duplicates


@pytest.mark.parametrize("filelist, expected", [
    ([Path("a.jpg")], {}),
    ([Path("a.jpg"), Path("b.jpg")], {}),
    ([Path("x/a.jpg"), Path("y/a.jpg")], {"a.jpg": 0}),
])
def test_find_duplicates(filelist, expected):
    assert find_duplicates(filelist) == expected
