"""Download the CAMeL Parser models into resources/parser_models (or $CAMEL_PARSER_MODELS)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import PARSER_MODEL_DIR  # noqa: E402
from camel_parser.src.utils.model_downloader import download_default_models  # noqa: E402

if __name__ == '__main__':
    PARSER_MODEL_DIR.parent.mkdir(parents=True, exist_ok=True)
    download_default_models(PARSER_MODEL_DIR)
