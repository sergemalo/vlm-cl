from pathlib import Path
import logging

def init_logging(log_level: str, output_dir: Path):
    """
    Initialize logging with both console and file handlers.
     - Console logs are filtered by the specified log level.
     - File logs capture everything at DEBUG level for detailed analysis.
     - Log file is saved in the output directory with a timestamped name.
    """
    log_file    = output_dir / "run.log"

    # Tricky: using root logger to ensure all logs (including from imported modules) are captured and directed to our handlers.
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # Global minimum level

    # --- Console handler ---
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level.upper())

    # --- File handler ---
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # Attach handlers
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)    
