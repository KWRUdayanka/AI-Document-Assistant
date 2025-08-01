import logging
import colorlog

formatter = colorlog.ColoredFormatter(
    "%(log_color)s[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    log_colors={
        'DEBUG': 'cyan',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'bold_red',
    }
)

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# ✅ Only add handler if no handlers are attached
if not logger.handlers:
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
