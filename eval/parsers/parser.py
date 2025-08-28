from abc import ABC, abstractmethod
import pandas as pd

try:
    from parsers.parser_type import ParserType
except ImportError:
    from ..parsers.parser_type import ParserType


class Parser(ABC):
    parser_type: ParserType

    def __init__(self, folder_path: str, config: dict):
        """
        Args:
            folder_path: the path to the folder containing the file. E.g.: "baselines/results_66/lotus/0"
        """
        self.config = config
        self.folder_path = folder_path
        self.file_id = (
            self.config["file_id"]
            if "file_id" in self.config
            else self.folder_path.split("/")[-1]
        )
        self.use_local_reference_map = self.config.get("use_local_reference_map", True)

        self.raw_generated_text: str | None = None
        self.clean_text: str | None = None
        self.docs: list[dict[str, str]] | None = None
        self.citations_for_cite_quality: list[tuple[str, str]] | None = None

        self._load_dataset()
        self._load_file()

    def _load_dataset(self):
        if "s_map_groundtruth" in self.config:
            self.s_map_groundtruth = self.config["s_map_groundtruth"]
        else:
            if "dataset" in self.config:
                dataset = self.config["dataset"]
            else:
                dataset = pd.read_csv(self.config["dataset_path"])

            row = dataset.iloc[int(self.file_id)]
            self.s_map_groundtruth = {
                "title": row["title"],
                "abstract": row["abstract"],
                "arxiv_link": row["arxiv_link"],
                "related_works_section": row.get("clean_latex_related_works", ""),
                "arxiv_id": row["arxiv_id"],
            }

    @abstractmethod
    def _load_file(self):
        pass

    def get_folder_info(
        self, include_related_works_section: bool = True
    ) -> dict[str, str]:
        return {
            "folder_path": self.folder_path,
            "paper_title": self.s_map_groundtruth["title"],
            "paper_abstract": self.s_map_groundtruth["abstract"],
            **(
                {"generated_related_works_section": self.clean_text or ""}
                if include_related_works_section
                else {}
            ),
            **(
                {
                    "related_works_section": self.s_map_groundtruth[
                        "related_works_section"
                    ]
                }
                if include_related_works_section
                else {}
            ),
        }
