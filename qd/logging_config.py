# qd/logging_config.py
"""Configures structlog + stdlib logging for the QD pipeline.

Console format : hh:mm:ss [LEVEL] message | key=value ...
File format    : hh:mm:ss [LEVEL] [filename:lineno] message | key=value ...

Console verbosity : configurable (default INFO)
File verbosity    : configurable (default DEBUG)
Log files         : written under ``logs/`` by default.
"""

import datetime
import logging
import os

import structlog
from structlog.processors import CallsiteParameter
from structlog.stdlib import ProcessorFormatter


_RESET = "\x1b[0m"
_LEVEL_COLORS = {
    "DEBUG": "\x1b[36m",     # cyan
    "INFO": "\x1b[32m",      # green
    "WARNING": "\x1b[33m",   # yellow
    "ERROR": "\x1b[31m",     # red
    "CRITICAL": "\x1b[35;1m",  # bold magenta
}


def _use_color() -> bool:
    """Enable ANSI colors unless explicitly disabled by NO_COLOR."""
    return os.getenv("NO_COLOR") is None


def _colorize_level(level: str) -> str:
    if not _use_color():
        return f"[{level}]"
    color = _LEVEL_COLORS.get(level, "")
    if not color:
        return f"[{level}]"
    return f"{color}[{level}]{_RESET}"


def _extra(event_dict: dict) -> str:
    return (
        " | " + " ".join(f"{k}={v}" for k, v in sorted(event_dict.items()))
        if event_dict
        else ""
    )


def _console_renderer(_, __, event_dict: dict) -> str:
    """``hh:mm:ss [LEVEL] message`` — no callsite info."""
    ts    = event_dict.pop("timestamp", "")
    level = event_dict.pop("level", "???").upper()
    # drop callsite keys so they don't end up in the extra block
    event_dict.pop("filename", None)
    event_dict.pop("lineno", None)
    msg   = event_dict.pop("event", "")
    return f"{ts} {_colorize_level(level)} {msg}{_extra(event_dict)}"


def _file_renderer(_, __, event_dict: dict) -> str:
    """``hh:mm:ss [LEVEL] [filename:lineno] message`` — full callsite info."""
    ts       = event_dict.pop("timestamp", "")
    level    = event_dict.pop("level", "???").upper()
    filename = event_dict.pop("filename", "?")
    lineno   = event_dict.pop("lineno", "?")
    msg      = event_dict.pop("event", "")
    return f"{ts} [{level}] [{filename}:{lineno}] {msg}{_extra(event_dict)}"


# Processors that run before the per-handler renderer
_PRE_CHAIN = [
    structlog.stdlib.add_log_level,
    structlog.processors.TimeStamper(fmt="%H:%M:%S"),
    structlog.processors.CallsiteParameterAdder(
        parameters=[CallsiteParameter.FILENAME, CallsiteParameter.LINENO],
    ),
    structlog.processors.StackInfoRenderer(),
]


def setup_logging(
    log_dir: str = "logs",
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
    log_filename: str | None = None,
) -> str:
    """Set up structlog + stdlib logging for the QD pipeline.

    Parameters
    ----------
    log_dir       : Directory under which log files are written.
    console_level : Minimum log level shown on the console (e.g. ``logging.INFO``).
    file_level    : Minimum log level written to the log file (e.g. ``logging.DEBUG``).
    log_filename  : Explicit path for the log file.  If *None*, defaults to
                    ``{log_dir}/run_YYYYMMDD_HHMMSS.log``.

    Returns
    -------
    str
        Absolute path to the log file that was opened.
    """
    os.makedirs(log_dir, exist_ok=True)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if log_filename is None:
        log_filename = os.path.join(log_dir, f"run_{ts}.log")
    else:
        log_filename = os.path.join(log_dir, f"{log_filename}_{ts}.log")

    # ── stdlib root logger ──────────────────────────────────────────────────
    # Attach handlers to a project-specific logger instead of the root logger
    # so that DEBUG/INFO noise from third-party libraries (numba, dask, etc.)
    # is never written to our log file.
    project_log = logging.getLogger("qd")
    project_log.setLevel(logging.DEBUG)
    project_log.propagate = False  # don't forward to root

    # Clear handlers added by previous calls (safe for notebook re-runs)
    project_log.handlers.clear()

    console_h = logging.StreamHandler()
    console_h.setLevel(console_level)
    console_h.setFormatter(ProcessorFormatter(
        foreign_pre_chain=_PRE_CHAIN,
        processors=[ProcessorFormatter.remove_processors_meta, _console_renderer],
    ))
    project_log.addHandler(console_h)

    file_h = logging.FileHandler(log_filename, encoding="utf-8")
    file_h.setLevel(file_level)
    file_h.setFormatter(ProcessorFormatter(
        foreign_pre_chain=_PRE_CHAIN,
        processors=[ProcessorFormatter.remove_processors_meta, _file_renderer],
    ))
    project_log.addHandler(file_h)

    # ── structlog ───────────────────────────────────────────────────────────
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            *_PRE_CHAIN,
            ProcessorFormatter.wrap_for_formatter,
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    log = get_logger(__name__)
    log.info(
        "Logging initialised",
        log_file=log_filename,
        console_level=logging.getLevelName(console_level),
        file_level=logging.getLevelName(file_level),
    )
    return log_filename


def get_logger(name: str = "") -> structlog.stdlib.BoundLogger:
    """Return a structlog logger guaranteed to be under the ``qd`` namespace.

    All modules in the pipeline should use this instead of calling
    ``structlog.get_logger`` directly, so every log record is routed through
    the single project handler regardless of how the module was imported.

    Parameters
    ----------
    name : str
        Short module name (e.g. ``"emitter"``, ``"qd_runner"``).
        If already fully qualified (starts with ``"qd"``), it is used
        as-is.  Pass no argument (or ``""``) to get the root project logger.
    """
    if not name:
        qualified = "qd"
    elif name.startswith("qd"):
        qualified = name
    else:
        # Strip any leading dots/slashes that __name__ or __file__ might produce
        qualified = f"qd.{name.lstrip('.')}"
    return structlog.get_logger(qualified)
