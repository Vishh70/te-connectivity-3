from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parent
PYTEST_TMP_ROOT = PROJECT_ROOT / ".pytest_tmp_local"


@pytest.fixture
def tmp_path():
    PYTEST_TMP_ROOT.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix="tmp-", dir=PYTEST_TMP_ROOT))
    try:
        yield temp_dir
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
