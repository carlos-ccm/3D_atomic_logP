#!/usr/bin/env python
# ----------------------------------------------------------
# CBDD - Torribera, Universitat de Barcelona
#
# File: logger.py
# ----------------------------------------------------------
# Coloured console messages.
# ----------------------------------------------------------

import sys
from os.path import exists

# ANSI escape colors
COLORS = {
    "NC": "\033[0m",
    "YELLOW": "\033[38;5;226m",
    "YELLOW_B": "\033[1;33m",
    "YELLOW_B2": "\033[38;5;178m",
    "RED": "\033[1;31m",
    "GREEN_B": "\033[1;32m",
    "GREEN": "\033[0;92m",
    "CYAN": "\033[1;36m",
    "UNDERLINE": "\033[4m"
}

def banner(message, level="INFO"):
    """
    Generic banner printer.
    level: INFO | WARNING | ERROR
    """

    Y = COLORS["YELLOW"]
    YB = COLORS["YELLOW_B"]
    YB2 = COLORS["YELLOW_B2"]
    R = COLORS["RED"]
    G = COLORS["GREEN"]
    GB = COLORS["GREEN_B"]
    C = COLORS["CYAN"]
    NC = COLORS["NC"]
    UNDERLINE = COLORS["UNDERLINE"]

    # Select color and prefix based on level
    if level == "INFO":
        prefix = f"                         {GB}{UNDERLINE}INFO{NC}"
        line = f"{GB}{'=' * 60}{NC}"
    elif level == "WARNING":
        prefix = f"{YB}WARNING{NC}"
        line = f"{YB}{'=' * 60}{NC}"
    elif level == "WARNING_DEV":
        prefix = f"{YB2}WARNING FOR DEVELOPERS{NC}"
        line = f"{YB2}{'=' * 60}{NC}"
    elif level == "ERROR":
        prefix = f"{R}ERROR{NC}"
        line = f"{R}{'=' * 60}{NC}"
    elif level == "TIME":
        prefix = f"{C}TIME{NC}"
        line = f"{C}{'=' * 60}{NC}"
    elif level == "INFO_YELLOW":
        print(f"{Y}{message}{NC}")
        return
    elif level == "INFO_YELLOW_BOLD":
        print(f"{YB}{message}{NC}")
        return
    elif level == "INFO_GREEN":
        print(f"{G}{message}{NC}")
        return
    elif level == "INFO_GREEN_BOLD":
        print(f"{GB}{message}{NC}")
        return
    elif level == "INFO_BASIC":
        print(f"{message}")
        return
    else:
        prefix = f"{C}{level}{NC}"  # fallback for custom types
        line = f"{C}{'=' * 55}{NC}"

    print(f"{line}")
    print(f"{prefix}{NC}")
    print(f"{message}{NC}")
    print(f"{line}")

def msg(text):
    banner(text, "INFO")

def msg_y(text):
    banner(text, "INFO_YELLOW")

def msg_yb(text):
    banner(text, "INFO_YELLOW_BOLD")

def msg_g(text):
    banner(text, "INFO_GREEN")

def msg_gb(text):
    banner(text, "INFO_GREEN_BOLD")

def msg_st(text):
    banner(text, "INFO_BASIC")

def msg_warning(text):
    banner(text, "WARNING")

def msg_warning_dev(text):
    banner(text, "WARNING_DEV")

def msg_time(text):
    banner(text, "TIME")

def msg_error(text):
    banner(text, "ERROR")

def msg_error_quit(text):
    banner(text, "ERROR")
    sys.exit(1)

def check_exist(file):
    if not exists(file):
            msg_error_quit(f"The input file does not exist: {file}")