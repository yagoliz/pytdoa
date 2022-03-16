#   Copyright (C) IMDEA Networks Institute 2022
#   This program is free software: you can redistribute it and/or modify
#
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with this program.  If not, see http://www.gnu.org/licenses/.
#
#   Authors: Yago Lizarribar <yago.lizarribar [at] imdea [dot] org>
#

import argparse
import json
import logging
import yaml

from pytdoa.pytdoa import pytdoa

################################################################################
# Logging library configuration
logger = logging.getLogger("MAIN")


# Load user defined configuration
def configure_logger(filename=None, level="INFO"):

    if filename == None:
        logging.basicConfig(
            level=level,
            format="%(name)s \t- %(levelname)s \t- %(message)s",
        )

        logger.warning("No logging configuration provided. Using default")

    else:
        with open(filename, "r") as f:
            config = yaml.safe_load(f.read())
            logging.config.dictConfig(config)


################################################################################
# MAIN definition
if __name__ == "__main__":

    # Argument parsing
    parser = argparse.ArgumentParser(
        description="Multilateration with RTL-SDR receivers"
    )

    # TDOA configuration
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="Configuration file for the optimizer in JSON format",
    )

    # Logging
    parser.add_argument(
        "--logging-config",
        type=str,
        required=False,
        help="Logging configuration file",
        default=None,
    )

    parser.add_argument(
        "--logging-level",
        type=str,
        choices=["DEBUG", "INFO", "WARN", "ERROR", "CRITICAL"],
        default="INFO",
        required=False,
        help="Debug level [Default: INFO]",
    )
    args = parser.parse_args()

    # Load configuration
    config_filename = args.config
    with open(config_filename) as f:
        config = json.load(f)

    # Parse logger configuration
    configure_logger(filename=args.logging_config, level=args.logging_level)

    # Position estimation
    position = pytdoa(config)
    logger.info(f"Result: {position.tolist()}")
