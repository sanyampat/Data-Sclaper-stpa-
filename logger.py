"""
logger.py — Centralised logging for the hype pipeline.

Every module imports get_logger(__name__) rather than calling
logging.basicConfig() independently. This means:
  • All modules share the same handlers (console + rotating file).
  • Log level is set once, here.
  • The rotating file handler caps log files at 5 MB × 3 backups.
"""

import logging
import sys
import threading
from logging.handlers import RotatingFileHandler
from pathlib import Path

from config import LOG_DIR

_LOG_FILE    = LOG_DIR / "pipeline.log"
_LOG_LEVEL   = logging.INFO
_FMT         = "%(asctime)s [%(levelname)-8s] %(name)s — %(message)s"
_DATE_FMT    = "%Y-%m-%d %H:%M:%S"
_initialised = False
_init_lock   = threading.Lock()


def _init():
    global _initialised
    with _init_lock:
        if _initialised:
            return
        _initialised = True
    root = logging.getLogger()
    root.setLevel(_LOG_LEVEL)

    fmt = logging.Formatter(_FMT, datefmt=_DATE_FMT)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    root.addHandler(ch)

    # Rotating file handler — keeps last 3 × 5 MB
    fh = RotatingFileHandler(
        _LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    fh.setFormatter(fmt)
    root.addHandler(fh)

    # Silence noisy third-party loggers
    for noisy in ("urllib3", "requests", "pytrends", "charset_normalizer"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    _init()
    return logging.getLogger(name)
