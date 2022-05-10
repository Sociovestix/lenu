import logging
from logging import Handler, LogRecord

from click import echo


class TyperEchoHandler(Handler):
    def emit(self, record: LogRecord) -> None:
        if record.levelno >= self.level:
            msg = (
                self.formatter.format(record) if self.formatter else record.getMessage()
            )
            echo(msg, err=record.levelno >= logging.WARNING)


def typer_log_config(verbose: bool = False, loglevel: str = "INFO"):
    if verbose:
        logger = logging.getLogger()
        logger.setLevel(loglevel.upper())

        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s", "%m-%d-%Y %H:%M:%S"
        )

        echo_handler = TyperEchoHandler()
        echo_handler.setFormatter(formatter)
        logger.addHandler(echo_handler)
