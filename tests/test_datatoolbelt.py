from importlib.metadata import version

import datatoolbelt


def test_version():
    assert datatoolbelt.__version__ == version("datatoolbelt")
