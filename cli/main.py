"""
Main CLI entry point.
Handles command parsing and execution.
"""
import argparse
import asyncio
import logging
import sys
from typing import List, Optional

from rich.logging import RichHandler

from cli.commands import COMMANDS

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger("cli")


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser"""
    parser = argparse.ArgumentParser(
        description="Neuroimaging Analysis Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Add global arguments
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set the logging level"
    )
    
    # Create subparsers for commands
    subparsers = parser.add_subparsers(
        title="commands",
        dest="command",
        help="Available commands"
    )
    
    # Add command subparsers
    for name, command in COMMANDS.items():
        subparser = subparsers.add_parser(
            name,
            help=command.help,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        command.add_arguments(subparser)
    
    return parser


async def main(args: Optional[List[str]] = None) -> int:
    """Main entry point"""
    try:
        # Parse arguments
        parser = create_parser()
        parsed_args = parser.parse_args(args)
        
        # Update log level if specified
        if parsed_args.log_level:
            logger.setLevel(parsed_args.log_level)
        
        # Get command
        if not parsed_args.command:
            parser.print_help()
            return 1
        
        command = COMMANDS[parsed_args.command]
        
        # Execute command
        await command.execute(parsed_args)
        return 0
        
    except KeyboardInterrupt:
        logger.warning("\nOperation cancelled by user")
        return 130
    except Exception as e:
        logger.error(f"Operation failed: {e}")
        if parsed_args.log_level == "DEBUG":
            logger.exception(e)
            logger.error(f"{e.traceback()}")
        return 1


def run() -> None:
    """Run CLI"""
    try:
        sys.exit(asyncio.run(main()))
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    run()