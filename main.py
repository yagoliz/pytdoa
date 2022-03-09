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
        default=None
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