"""Compatibility shim for setuptools-based builds.

Project metadata lives in ``pyproject.toml``. This file exists only so older
tooling that still imports ``setup.py`` continues to work.
"""

from setuptools import setup

setup()
