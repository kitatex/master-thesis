import logging
from datetime import datetime
from pathlib import Path

from data_processing.config.constants import VERBOSE

# Create a timestamped log file name
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)  # ensure the logs directory exists
log_file = log_dir / f"search_log_{timestamp}.txt"


logging.basicConfig(
    level=logging.INFO if not VERBOSE else logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(log_file, mode="w", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger(__name__)
