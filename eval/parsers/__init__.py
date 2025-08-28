try:
    from parsers.parser import Parser, ParserType
    from parsers.search_ai import SearchAIParser
    from eval.parsers.deepscholar_base import DeepScholarBaseParser
    from parsers.storm import StormParser
    from parsers.openscholar import OpenScholarParser
    from parsers.deepresearcher import DeepResearcherParser
    from parsers.groundtruth import GroundTruthParser
except ImportError:
    from .parser import Parser, ParserType
    from .search_ai import SearchAIParser
    from .deepscholar_base import DeepScholarBaseParser
    from .storm import StormParser
    from .openscholar import OpenScholarParser
    from .deepresearcher import DeepResearcherParser
    from .groundtruth import GroundTruthParser
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

__all__ = [
    "SearchAIParser",
    "DeepScholarBaseParser",
    "StormParser",
    "OpenScholarParser",
    "DeepResearcherParser",
    "GroundTruthParser",
    "Parser",
    "ParserType",
]


def get_parser(config: dict, folder_path: str):
    logger.info(f"Getting parser for {config['mode']}")
    if not isinstance(config["mode"], ParserType):
        parser_type = ParserType(config["mode"])
    else:
        parser_type = config["mode"]

    if parser_type == ParserType.SEARCH_AI:
        return SearchAIParser(folder_path, config)
    elif parser_type == ParserType.DEEPSCHOLAR_BASE:
        return DeepScholarBaseParser(folder_path, config)
    elif parser_type == ParserType.STORM:
        return StormParser(folder_path, config)
    elif parser_type == ParserType.OPENSCHOLAR:
        return OpenScholarParser(folder_path, config)
    elif parser_type == ParserType.DEEPRESEARCHER:
        return DeepResearcherParser(folder_path, config)
    elif parser_type == ParserType.GROUNDTRUTH:
        return GroundTruthParser(folder_path, config)
