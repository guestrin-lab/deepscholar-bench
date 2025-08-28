"""
LaTeX content extractor for downloading ArXiv source files and extracting related works sections.
"""

import asyncio
import logging
import re
import tarfile
import tempfile
import os
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime
import requests  # type: ignore
import pandas as pd

try:
    import PyPDF2

    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    logging.warning("PyPDF2 not available. Install it with: pip install PyPDF2")

try:
    from config import PipelineConfig
    from arxiv_scraper import ArxivPaper
except ImportError:
    from .config import PipelineConfig
    from .arxiv_scraper import ArxivPaper


logger = logging.getLogger(__name__)


@dataclass
class PaperData:
    """Data structure for extracted paper information."""

    arxiv_link: str
    publication_date: datetime
    paper_title: str
    abstract: str
    related_works_section: str


@dataclass
class CitationData:
    """Data structure for citation information."""

    parent_paper_title: str
    parent_arxiv_link: str
    citation_shorthand: str
    raw_citation_text: str

    # Citation details (populated during lookup)
    cited_paper_title: Optional[str] = None
    cited_paper_arxiv_link: Optional[str] = None
    cited_paper_abstract: Optional[str] = None

    # Bibliography information
    bib_paper_authors: Optional[str] = None
    bib_paper_year: Optional[str] = None
    bib_paper_month: Optional[str] = None
    bib_paper_url: Optional[str] = None
    bib_paper_doi: Optional[str] = None
    bib_paper_journal: Optional[str] = None


class LatexExtractor:
    """Extract content from ArXiv LaTeX source files."""

    def __init__(self, config: PipelineConfig):
        self.config = config

    async def extract_papers_content(self, papers: List[ArxivPaper]) -> List[PaperData]:
        """
        Extract related works sections from a list of papers.

        Args:
            papers: List of ArxivPaper objects

        Returns:
            List of PaperData objects with extracted content
        """
        if not PDF_AVAILABLE:
            logger.error(
                "PyPDF2 is required for PDF processing. Install it with: pip install PyPDF2"
            )
            return []

        paper_data_list = []

        for paper in papers:
            try:
                logger.info(f"Processing paper: {paper.title}")

                # Download LaTeX source first (for citation extraction)
                latex_content = await self._download_latex_source(paper)
                if not latex_content:
                    logger.warning(
                        f"Could not download LaTeX source for {paper.arxiv_id}"
                    )
                    continue

                # Store project files temporarily for multi-file support
                if hasattr(paper, "_project_files"):
                    self._current_project_files = paper._project_files
                else:
                    self._current_project_files = None

                # Extract related works section from LaTeX (for citation parsing)
                latex_related_works = self._extract_related_works_section(latex_content)
                if not latex_related_works:
                    logger.warning(
                        f"No related works section found in LaTeX for {paper.arxiv_id}"
                    )
                    continue

                # Download PDF and extract related works section from it (for clean final output)
                pdf_related_works = await self._download_and_extract_pdf_related_works(
                    paper
                )
                clean_latex_content = self._clean_latex_content(latex_related_works)
                pdf_failed = False
                if not pdf_related_works:
                    logger.warning(
                        f"No related works section found in PDF for {paper.arxiv_id}, using LaTeX version"
                    )
                    pdf_related_works = clean_latex_content
                    pdf_failed = True
                # Create PaperData object with PDF-extracted related works for final output
                paper_data = PaperData(
                    arxiv_link=paper.abs_url,
                    publication_date=paper.published_date,
                    paper_title=paper.title,
                    abstract=paper.abstract,
                    related_works_section=pdf_related_works,  # Clean version for CSV output
                )

                # Store LaTeX-based related works for citation extraction
                paper_data._latex_related_works = (  # type: ignore
                    latex_related_works  # Raw LaTeX for citation parsing
                )
                paper_data._full_latex_content = latex_content  # type: ignore
                paper_data._paper_object = (  # type: ignore
                    paper  # Store reference to access project files
                )

                paper_data_list.append(paper_data)

                # Save paper data to temporary CSV file for debugging
                temp_dir = os.path.join(self.config.output_dir, "related_works")
                os.makedirs(temp_dir, exist_ok=True)
                temp_file = os.path.join(temp_dir, f"{paper.arxiv_id}.csv")

                # Create DataFrame with paper data
                temp_data = {
                    "arxiv_id": [paper.arxiv_id],
                    "arxiv_link": [paper.abs_url],
                    "publication_date": [paper.published_date],
                    "title": [paper.title],
                    "abstract": [paper.abstract],
                    "raw_latex_related_works": [latex_related_works],
                    "clean_latex_related_works": [clean_latex_content],
                    "pdf_related_works": [pdf_related_works]
                    if not pdf_failed
                    else "N/A",
                }
                temp_df = pd.DataFrame(temp_data)

                # Save to CSV
                temp_df.to_csv(temp_file, index=False)
                logger.info(f"Saved temporary paper data to {temp_file}")

                # Clean up temporary storage
                self._current_project_files = None

                # Add delay to be respectful
                await asyncio.sleep(self.config.request_delay)

            except Exception as e:
                logger.error(f"Error processing paper {paper.arxiv_id}: {e}")
                continue

        logger.info(
            f"Successfully extracted content from {len(paper_data_list)} papers"
        )
        return paper_data_list

    async def extract_citations_from_papers(
        self, paper_data_list: List[PaperData]
    ) -> List[CitationData]:
        """
        Extract citations from related works sections.

        Args:
            paper_data_list: List of PaperData objects

        Returns:
            List of CitationData objects
        """
        all_citations = []

        for paper_data in paper_data_list:
            try:
                logger.info(f"Extracting citations from: {paper_data.paper_title}")

                # First get the full LaTeX content to extract bibliography
                latex_content = getattr(paper_data, "_full_latex_content", None)
                bibliography = None

                if latex_content:
                    # Get project files for bibliography lookup
                    project_files = getattr(paper_data, "_paper_object", None)
                    if hasattr(paper_data, "_paper_object") and hasattr(
                        paper_data._paper_object, "_project_files"
                    ):
                        project_files = paper_data._paper_object._project_files
                    else:
                        project_files = None

                    bibliography = self._extract_bibliography(
                        latex_content, project_files
                    )
                    if bibliography:
                        logger.info(
                            f"Found bibliography with {len(bibliography)} entries"
                        )

                # Parse citations from related works section
                citations = await self._extract_citations_from_text(
                    getattr(
                        paper_data,
                        "_latex_related_works",
                        paper_data.related_works_section,
                    ),  # Use LaTeX version for citation parsing
                    paper_data.paper_title,
                    paper_data.arxiv_link,
                    bibliography,
                )
                # save temporary citations for paper
                temp_df = pd.DataFrame(citations)
                temp_dir = os.path.join(self.config.output_dir, "citations")
                os.makedirs(temp_dir, exist_ok=True)
                arxiv_id = paper_data.arxiv_link.split("/")[-1]
                temp_file = os.path.join(temp_dir, f"{arxiv_id}.csv")
                temp_df.to_csv(temp_file, index=False)
                logger.info(f"Saved temporary citations to {temp_file}")

                all_citations.extend(citations)

                # Add delay between citation lookups
                await asyncio.sleep(self.config.request_delay)

            except Exception as e:
                logger.error(
                    f"Error extracting citations from {paper_data.paper_title}: {e}"
                )
                continue

        logger.info(f"Extracted {len(all_citations)} citations total")
        return all_citations

    async def _download_latex_source(self, paper: ArxivPaper) -> Optional[str]:
        """Download and extract LaTeX source from ArXiv."""
        try:
            # Construct source URL
            arxiv_id = paper.arxiv_id
            source_url = f"https://arxiv.org/src/{arxiv_id}"

            temp_dir = os.path.join(self.config.output_dir, "latex_source")
            os.makedirs(temp_dir, exist_ok=True)
            archive_path = os.path.join(temp_dir, f"{arxiv_id}.tar.gz")
            temp_dir = temp_dir + "/" + arxiv_id
            latex_content: str | None
            if os.path.exists(archive_path):
                logger.info(
                    f"Using cached LaTeX source for {arxiv_id} from {archive_path}"
                )
                with open(archive_path, "rb") as f:
                    content = f.read()
            else:
                # Download the source archive
                response = requests.get(source_url, timeout=30)
                response.raise_for_status()
                content = response.content

                with open(archive_path, "wb") as f:
                    f.write(content)

            # Extract the archive
            try:
                with tarfile.open(archive_path, "r:gz") as tar:
                    tar.extractall(temp_dir)
            except tarfile.ReadError as e:
                # Try as a regular file (sometimes ArXiv returns single .tex files)
                latex_content = content.decode("utf-8", errors="ignore")
                logger.warning(
                    f"Failed to extract tar.gz archive for {paper.arxiv_id}, treating as single .tex file"
                )
                logger.error(f"Error reading tar.gz: {e}")
                return latex_content

            # Find the main .tex file
            latex_content = self._find_main_tex_file(temp_dir)

            # Read all project files for bibliography lookup (before temp_dir is deleted)
            if latex_content:
                paper._project_files = self._read_all_project_files(temp_dir)

            return latex_content

        except Exception as e:
            logger.error(f"Error downloading LaTeX source for {paper.arxiv_id}: {e}")
            return None

    def _read_all_project_files(self, directory: str) -> Dict[str, str]:
        """Read all text files in the project directory for bibliography lookup."""
        project_files = {}

        # Read .tex files
        for tex_file in Path(directory).glob("**/*.tex"):
            try:
                content = tex_file.read_text(encoding="utf-8", errors="ignore")
                project_files[tex_file.name] = content
            except Exception as e:
                logger.debug(f"Could not read {tex_file}: {e}")

        # Read .bib files
        for bib_file in Path(directory).glob("**/*.bib"):
            try:
                content = bib_file.read_text(encoding="utf-8", errors="ignore")
                project_files[bib_file.name] = content
            except Exception as e:
                logger.debug(f"Could not read {bib_file}: {e}")

        # Read .bbx, .bst, and other bibliography-related files
        for ext in ["*.bbx", "*.bst", "*.cls", "*.sty", "*.bbl"]:
            for file_path in Path(directory).glob(f"**/{ext}"):
                try:
                    content = file_path.read_text(encoding="utf-8", errors="ignore")
                    project_files[file_path.name] = content
                except Exception as e:
                    logger.debug(f"Could not read {file_path}: {e}")

        logger.debug(
            f"Read {len(project_files)} project files: {list(project_files.keys())}"
        )
        return project_files

    def _find_main_tex_file(self, directory: str) -> Optional[str]:
        """Find and read the main LaTeX file from extracted archive."""
        tex_files = list(Path(directory).glob("**/*.tex"))

        if not tex_files:
            return None

        # Heuristics to find the main file
        main_candidates = []

        for tex_file in tex_files:
            # import pdb; pdb.set_trace()
            content = tex_file.read_text(encoding="utf-8", errors="ignore")

            # Look for document class (main files typically have this)
            # if r'\documentclass' in content:
            match_names = [
                f"\\section{{{name}}}"
                for name in self.config.related_works_section_names
            ]
            match_names = [re.escape(name) for name in match_names]
            if any(
                re.search(pattern, content, re.IGNORECASE) for pattern in match_names
            ):
                # if r'\section{Related work}' in content:
                main_candidates.append((tex_file, content))

            # match_names = [f"\\subsection{{{name}}}" for name in self.config.related_works_section_names]
            # match_names = [re.escape(name) for name in match_names]
            # if any(re.search(pattern, content, re.IGNORECASE) for pattern in match_names):
            # # if r'\section{Related work}' in content:
            #     main_candidates.append((tex_file, content))

        if main_candidates:
            # If multiple candidates, prefer the one with more content
            main_file, content = max(main_candidates, key=lambda x: len(x[1]))
            return content

        # # Fallback: return the largest .tex file
        # if tex_files:
        #     largest_file = max(tex_files, key=lambda f: f.stat().st_size)
        #     return largest_file.read_text(encoding='utf-8', errors='ignore')

        return None

    def _extract_related_works_section(self, latex_content: str) -> Optional[str]:
        """Extract related works section from LaTeX content, supporting both inline and multi-file projects."""

        # First, try the current logic - look for sections directly in the main file
        section_patterns = []
        for section_name in self.config.related_works_section_names:
            # Escape special regex characters and create flexible pattern
            escaped_name = re.escape(section_name)
            # Allow for some flexibility in spacing and formatting
            pattern = escaped_name.replace(r"\ ", r"\s+")
            section_patterns.append(pattern)

        # Create comprehensive regex pattern for inline sections
        pattern = (
            r"\\section\*?\{("
            + "|".join(section_patterns)
            + r")\}(.*?)(?=\\section|\Z)"
        )

        matches = re.findall(pattern, latex_content, re.DOTALL | re.IGNORECASE)

        if matches:
            # Return the RAW content of the first matching section (don't clean yet!)
            section_title, section_content = matches[0]

            if len(section_content.strip()) >= 100:  # Minimum length check
                logger.info(
                    f"Found related works section directly in main file: '{section_title}'"
                )
                return section_content.strip()

        # If no direct section found, look for multi-file structure
        logger.info(
            "No related works section found in main file, checking for multi-file structure..."
        )

        # Look for \input{} commands that might include related works
        input_pattern = r"\\input\{([^}]+)\}"
        input_matches = re.findall(input_pattern, latex_content, re.IGNORECASE)

        if input_matches:
            logger.info(f"Found \\input commands: {input_matches}")

            # Check if we have project files available
            if hasattr(self, "_current_project_files") and self._current_project_files:
                # Look for files that might contain related works
                for input_file in input_matches:
                    # Check if this input file might be related works
                    if self._is_likely_related_works_file(input_file):
                        logger.info(
                            f"Checking input file '{input_file}' for related works content..."
                        )

                        # Try different file name variations
                        possible_names = [
                            f"{input_file}.tex",
                            f"{input_file}",
                            input_file.split("/")[-1]
                            + ".tex",  # Just the filename part
                            input_file.split("/")[-1],
                        ]

                        for name in possible_names:
                            if name in self._current_project_files:
                                file_content = self._current_project_files[name]
                                logger.info(
                                    f"Found file '{name}' with {len(file_content)} characters"
                                )

                                # Check if this file contains substantial content
                                if len(file_content.strip()) >= 100:
                                    logger.info(
                                        f"Using content from '{name}' as related works section"
                                    )
                                    return file_content.strip()

                # If no obvious related works files found, search all project files
                logger.info(
                    "No obvious related works input files found, searching all project files..."
                )
                return self._search_all_files_for_related_works()
            else:
                logger.warning(
                    "Multi-file structure detected but project files not available"
                )

        return None

    def _is_likely_related_works_file(self, filename: str) -> bool:
        """Check if a filename is likely to contain related works content."""
        # Remove path and extension for checking
        base_name = filename.split("/")[-1].replace(".tex", "").lower()

        related_keywords = [
            "relatedwork",
            "related_work",
            "related-work",
            "background",
            "literature",
            "survey",
            "prior",
            "previous",
            "review",
        ]

        for keyword in related_keywords:
            if keyword in base_name:
                return True

        return False

    def _search_all_files_for_related_works(self) -> Optional[str]:
        """Search all project files for related works content."""
        if (
            not hasattr(self, "_current_project_files")
            or not self._current_project_files
        ):
            return None

        best_content = None
        best_score = 0

        for filename, content in self._current_project_files.items():
            if not filename.endswith(".tex"):
                continue

            # Skip the main file (already checked)
            if "documentclass" in content:
                continue

            # Look for section headers in this file
            section_pattern = r"\\section\*?\{([^}]+)\}"
            sections = re.findall(section_pattern, content, re.IGNORECASE)

            for section_title in sections:
                # Check if this section title matches our related works patterns
                for target_name in self.config.related_works_section_names:
                    if target_name.lower() in section_title.lower():
                        # Extract the content of this section
                        section_content_pattern = (
                            r"\\section\*?\{"
                            + re.escape(section_title)
                            + r"\}(.*?)(?=\\section|\Z)"
                        )
                        section_matches = re.findall(
                            section_content_pattern, content, re.DOTALL | re.IGNORECASE
                        )

                        if section_matches and len(section_matches[0].strip()) >= 100:
                            logger.info(
                                f"Found related works section '{section_title}' in file '{filename}'"
                            )
                            return section_matches[0].strip()

            # If no section headers, but filename suggests related works, use entire content
            if (
                self._is_likely_related_works_file(filename)
                and len(content.strip()) >= 100
            ):
                # Score based on content length and filename relevance
                score = len(content.strip())
                if "related" in filename.lower():
                    score *= 2

                if score > best_score:
                    best_content = content.strip()
                    best_score = score
                    logger.info(
                        f"Candidate related works file: '{filename}' (score: {score})"
                    )

        if best_content:
            logger.info(
                f"Using best candidate file for related works (score: {best_score})"
            )
            return best_content

        logger.info("No related works content found in any project files")
        return None

    def _clean_latex_content(self, content: str) -> str:
        """Clean LaTeX content by removing common commands and formatting."""
        # Remove comments
        content = re.sub(r"%.*$", "", content, flags=re.MULTILINE)

        # Remove figures and their content
        content = re.sub(
            r"\\begin\{figure\}.*?\\end\{figure\}", "", content, flags=re.DOTALL
        )
        content = re.sub(
            r"\\begin\{figure\*\}.*?\\end\{figure\*\}", "", content, flags=re.DOTALL
        )
        content = re.sub(
            r"\\begin\{subfigure\}.*?\\end\{subfigure\}", "", content, flags=re.DOTALL
        )
        content = re.sub(
            r"\\begin\{subfigure\*\}.*?\\end\{subfigure\*\}",
            "",
            content,
            flags=re.DOTALL,
        )

        # Remove common LaTeX commands but keep the text, preserving citation commands
        # content = re.sub(r'\\[a-zA-Z]+\*?\{([^}]*)\}(?!\s*\\cite\{|\s*\\citep\{)', r'\1', content)  # Don't remove \cite{} or \citep{}
        # content = re.sub(r'\\[a-zA-Z]+\*?\[[^\]]*\]\{([^}]*)\}(?!\s*\\cite\{|\s*\\citep\{)', r'\1', content)  # Don't remove \cite{} or \citep{}
        # content = re.sub(r'(?!\\cite|\\citep)\\\([a-zA-Z]+\*?\{([^}]*)\}', r'\1', content)
        # content = re.sub(r'(?!\\cite|\\citep)\\\([a-zA-Z]+\*?\[[^\]]*\]\{([^}]*)\}', r'\1', content)

        # # Remove standalone commands
        # content = re.sub(r'\\[a-zA-Z]+\*?', '', content)

        # Remove labels and possible newline
        content = re.sub(r"\\label\{[^}]*\}\n?", "", content)

        # Clean up whitespace
        # content = re.sub(r'\s+', ' ', content)
        # Remove leading and trailing whitespace from the content string
        content = content.strip()

        return content

    def _parse_bib_file(self, bib_content: str) -> Dict[str, Dict[str, str | None]]:
        """Parse a .bbl file and extract entries."""
        bibliography = {}

        # Pattern to match BibTeX entries
        # @article{key, field1={value1}, field2={value2}, ...}
        bibtex_pattern = r"@\w+\s*\{\s*([^,\s]+)\s*,(.*?)(?=@\w+\s*\{|\Z)"

        entries = re.findall(bibtex_pattern, bib_content, re.DOTALL | re.IGNORECASE)

        for key, fields in entries:
            # Clean the key
            clean_key = key.strip()
            # Extract all available fields from BibTeX entry
            title = self._extract_bibtex_field(fields, "title")
            author = self._extract_bibtex_field(fields, "author")
            year = self._extract_bibtex_field(fields, "year")
            month = self._extract_bibtex_field(fields, "month")
            journal = self._extract_bibtex_field(fields, "journal")
            url = self._extract_bibtex_field(fields, "url")
            url_2 = self._extract_bibtex_field(fields, "URL")
            doi = self._extract_bibtex_field(fields, "doi")

            # Create a formatted bibliography entry
            bib_entry = dict()
            bib_entry["author"] = author if author else None
            bib_entry["title"] = title if title else None
            bib_entry["journal"] = journal if journal else None
            bib_entry["year"] = year if year else None
            bib_entry["month"] = month if month else None
            bib_entry["doi"] = doi if doi else None
            bib_entry["url"] = url_2 if url_2 else url if url else None

            if bib_entry:
                bibliography[clean_key] = bib_entry
                logger.debug(f"BibTeX entry: {clean_key} -> {bibliography[clean_key]}")

        return bibliography

    # write a function _extract_bibliography that finds the .bib file in the project files and parses it
    def _extract_bibliography(
        self, latex_content: str, project_files: Optional[Dict[str, str]] = None
    ) -> Optional[Dict[str, Dict[str, str | None]]]:
        """Extract bibliography entries from project files."""

        bibliography = {}

        # Look for .bib files in project files
        if not project_files:
            return None

        bib_files = [file for file in project_files if file.endswith(".bib")]
        if not bib_files:
            logger.info("No .bib files found in project files")
            return None

        # Parse the first .bib file found
        # If multiple .bib files found, concatenate their contents
        if len(bib_files) > 1:
            # just concatenate all the files
            bib_content = ""
            for bib_file in bib_files:
                bib_content += project_files[bib_file] + "\n"
            logger.info(f"Concatenated {len(bib_files)} .bib files")
        else:
            bib_file = bib_files[0]
            bib_content = project_files[bib_file]

        # Parse the .bib file
        bibliography = self._parse_bib_file(bib_content)

        return bibliography

    def _extract_bibtex_field(self, fields: str, field_name: str) -> Optional[str]:
        """Extract a specific field from BibTeX entry fields."""
        # Pattern to match field = {value} or field = "value"
        # Use word boundaries to ensure exact field name match
        # This prevents matching 'paper_title' when looking for 'title'
        # First try to match with outer braces
        match = re.search(
            rf"\b{field_name}\s*=\s*{{((?:[^{{}}]|{{(?:[^{{}}]|{{[^{{}}]*}})*}})*)}}",
            fields,
            re.IGNORECASE | re.DOTALL,
        )
        if not match:
            # Try matching with double braces
            match = re.search(
                rf'\b{field_name}\s*=\s*{{{{([^{{}}]*(?:{{[^{{}}]*}}[^{{}}"]*)*)}}}}',
                fields,
                re.IGNORECASE | re.DOTALL,
            )
        if not match:
            # If no match with braces, try with quotes
            match = re.search(
                rf'\b{field_name}\s*=\s*"([^"]*)"', fields, re.IGNORECASE | re.DOTALL
            )

        if match:
            value = match.group(1).strip().strip('"')
            # Clean up LaTeX commands in the value
            value = self._clean_latex_content(value)
            return value

        return None

    async def _extract_citations_from_text(
        self,
        text: str,
        parent_title: str,
        parent_arxiv_link: str,
        bibliography: Optional[Dict[str, Dict[str, str | None]]] = None,
    ) -> List[CitationData]:
        """Extract citations from a text and look up their details."""
        citations = []
        seen_citations = set()  # Track seen citations to avoid duplicates

        # First, remove LaTeX comments to avoid extracting citations from commented lines
        cleaned_text = self._remove_latex_comments(text)

        # Pattern to match citations like \cite{key1,key2}, \citep{key}, \citet{key}, etc.
        cite_pattern = r"\\cite[^{}]*\{([^}]+)\}"

        # Also look for inline citations like (Author et al., Year) in cleaned text
        # cleaned_readable_text = self._clean_latex_content(cleaned_text)
        inline_pattern = r"\(([^)]*(?:\d{4})[^)]*)\)"

        # Find all LaTeX citations in comment-free text
        cite_matches = re.findall(cite_pattern, cleaned_text)

        # Find inline citations in cleaned text
        inline_matches = re.findall(inline_pattern, cleaned_text)

        logger.info(
            f"Found {len(cite_matches)} LaTeX citations and {len(inline_matches)} inline citations (after comment removal)"
        )

        # Process \cite{} references
        for cite_match in cite_matches:
            # Split multiple citations
            cite_keys = [key.strip() for key in cite_match.split(",")]

            for cite_key in cite_keys:
                # Skip if we've already seen this citation key for this paper
                citation_id = f"{parent_title}::{cite_key}"
                if citation_id in seen_citations:
                    continue
                seen_citations.add(citation_id)

                citation_data = CitationData(
                    parent_paper_title=parent_title,
                    parent_arxiv_link=parent_arxiv_link,
                    citation_shorthand=cite_key,
                    raw_citation_text=f"\\cite{{{cite_key}}}",
                )

                # Try to look up the citation details using bibliography first
                await self._lookup_citation_details(citation_data, bibliography)
                citations.append(citation_data)

        # Process inline citations (only if we didn't find LaTeX citations)
        if not cite_matches:
            for inline_match in inline_matches:
                # Simple heuristic: if it contains a year and looks like an author citation
                if re.search(r"\d{4}", inline_match) and (
                    "et al" in inline_match.lower() or "," in inline_match
                ):
                    # Split multiple citations separated by semicolons
                    individual_citations = [
                        cite.strip() for cite in inline_match.split(";")
                    ]

                    for individual_cite in individual_citations:
                        # Ensure each individual citation still looks valid
                        if re.search(r"\d{4}", individual_cite) and (
                            "et al" in individual_cite.lower()
                            or len(individual_cite.split()) >= 2
                        ):
                            # Skip if we've already seen this citation for this paper
                            citation_id = f"{parent_title}::{individual_cite}"
                            if citation_id in seen_citations:
                                continue
                            seen_citations.add(citation_id)

                            citation_data = CitationData(
                                parent_paper_title=parent_title,
                                parent_arxiv_link=parent_arxiv_link,
                                citation_shorthand=individual_cite,
                                raw_citation_text=f"({individual_cite})",
                            )

                            await self._lookup_citation_details(
                                citation_data, bibliography
                            )
                            citations.append(citation_data)

        logger.info(f"Extracted {len(citations)} unique citations after deduplication")
        return citations

    def _remove_latex_comments(self, text: str) -> str:
        """Remove LaTeX comments from text while preserving line structure."""
        lines = text.split("\n")
        cleaned_lines = []

        for line in lines:
            # Find the first unescaped % character
            # We need to handle escaped % characters (\%)
            i = 0
            comment_start = -1

            while i < len(line):
                if line[i] == "%":
                    # Check if this % is escaped by counting preceding backslashes
                    backslash_count = 0
                    j = i - 1
                    while j >= 0 and line[j] == "\\":
                        backslash_count += 1
                        j -= 1

                    # If even number of backslashes (including 0), the % is not escaped
                    if backslash_count % 2 == 0:
                        comment_start = i
                        break
                i += 1

            if comment_start >= 0:
                # Remove everything from the comment character onwards
                cleaned_line = line[:comment_start].rstrip()
            else:
                cleaned_line = line

            cleaned_lines.append(cleaned_line)

        return "\n".join(cleaned_lines)

    async def _lookup_citation_details(
        self,
        citation: CitationData,
        bibliography: Optional[dict[str, dict[str, str | None]]] = None,
    ):
        """Look up details for a citation using bibliography and ArXiv API."""
        # # latex citaitons aren't case sensitive so convert to lower
        # Convert bibliography keys to lowercase
        if bibliography:
            bibliography = {k.lower(): v for k, v in bibliography.items()}

        try:
            # First try to find the citation in the bibliography
            if bibliography and citation.citation_shorthand.lower() in bibliography:
                bib_entry = bibliography[citation.citation_shorthand.lower()]
                logger.debug(
                    f"Found bibliography entry for '{citation.citation_shorthand}': {bib_entry}"
                )

                # Extract title and authors from bibliography entry
                # title, authors = self._parse_bibliography_entry(bib_entry)
                title = bib_entry.get("title")
                authors = bib_entry.get("author")
                # add all bib entry fields to citation
                citation.cited_paper_title = title
                citation.bib_paper_authors = authors
                citation.bib_paper_year = bib_entry.get("year")
                citation.bib_paper_month = bib_entry.get("month")
                citation.bib_paper_url = bib_entry.get("url")
                citation.bib_paper_doi = bib_entry.get("doi")
                citation.bib_paper_journal = bib_entry.get("journal")

                if title:
                    # Try to find this paper on ArXiv using the title and authors
                    paper_info = await self._search_arxiv_by_title_authors(
                        title, authors
                    )

                    if paper_info:
                        citation.cited_paper_title = paper_info.get("title")
                        citation.cited_paper_arxiv_link = paper_info.get("abs_url")
                        citation.cited_paper_abstract = paper_info.get("abstract")
                        logger.debug(
                            f"Found ArXiv match for bibliography entry: {citation.cited_paper_title}"
                        )
                        return
                    else:
                        # Use the bibliography info even if not on ArXiv
                        citation.cited_paper_title = title
                        logger.debug(
                            f"Using bibliography title (not found on ArXiv): {title}"
                        )
                        return

            # Fallback: use the search term approach
            search_terms = self._extract_search_terms(citation.citation_shorthand)

            if not search_terms:
                logger.debug(
                    f"No search terms extracted from citation: {citation.citation_shorthand}"
                )
                return

            # Search ArXiv API
            paper_info = await self._search_arxiv_for_citation(search_terms)

            if paper_info:
                # Only accept results if they seem reasonable
                title = paper_info.get("title", "")
                if title and len(title) > 10:  # Basic sanity check
                    citation.cited_paper_title = title
                    citation.cited_paper_arxiv_link = paper_info.get("abs_url")
                    citation.cited_paper_abstract = paper_info.get("abstract")
                    logger.debug(
                        f"Found metadata for citation '{citation.citation_shorthand}': {title}"
                    )
                else:
                    logger.debug(
                        f"Found low-quality result for citation '{citation.citation_shorthand}'"
                    )
            else:
                logger.debug(
                    f"No ArXiv results found for citation: {citation.citation_shorthand}"
                )

        except Exception as e:
            logger.debug(
                f"Error looking up citation {citation.citation_shorthand}: {e}"
            )

    def _extract_search_terms(self, citation_text: str) -> List[str]:
        """Extract search terms from citation text."""
        # Citation keys often follow patterns like: authorYEAR, author1YEAR, etc.
        terms = []

        # Extract years (4-digit numbers that look like years)
        years = re.findall(r"\b(19|20)\d{2}\b", citation_text)
        if years:
            terms.extend(years)

        # Extract author names - common patterns in citation keys
        # Remove year and common suffixes, then extract remaining text
        cleaned_key = re.sub(r"\b(19|20)\d{2}\b", "", citation_text)  # Remove years
        cleaned_key = re.sub(r"[^a-zA-Z]", " ", cleaned_key)  # Keep only letters

        # Extract words that could be author names (capitalized or all lowercase)
        potential_authors = re.findall(r"\b[a-zA-Z]{2,}\b", cleaned_key)

        # Take first few potential author names
        terms.extend(potential_authors[:2])

        return [term for term in terms if len(term) >= 2]  # Filter out very short terms

    def _parse_bibliography_entry(
        self, bib_entry: str
    ) -> tuple[Optional[str], Optional[List[str]]]:
        """Parse a bibliography entry to extract title and authors."""
        try:
            # Common patterns for bibliography entries:
            # Author, "Title", Journal, Year
            # Author. Title. Journal. Year.
            # etc.

            # Try to find quoted title first
            title_match = re.search(r'["\'\`\{]([^"\'`\}]{10,})["\'\`\}]', bib_entry)
            if title_match:
                title = title_match.group(1).strip()
            else:
                # Try to find title after author and before journal/venue
                # This is a heuristic approach
                parts = re.split(r"[.,;]", bib_entry)
                title = None
                for i, part in enumerate(parts):
                    if len(part.strip()) > 15 and i > 0:  # Likely a title
                        title = part.strip()
                        break

            # Extract authors (usually first part before first comma or period)
            authors = []
            author_part = (
                bib_entry.split(",")[0].strip()
                if "," in bib_entry
                else bib_entry.split(".")[0].strip()
            )

            # Split multiple authors
            if " and " in author_part:
                authors = [auth.strip() for auth in author_part.split(" and ")]
            elif "," in author_part and len(author_part.split(",")) <= 3:
                authors = [auth.strip() for auth in author_part.split(",")]
            else:
                authors = [author_part] if author_part else []

            return title, authors if authors else None

        except Exception as e:
            logger.debug(f"Error parsing bibliography entry: {e}")
            return None, None

    async def _search_arxiv_by_title_authors(
        self, title: str, authors: Optional[List[str] | str]
    ) -> Optional[Dict[str, str]]:
        """Search ArXiv API using title and author information."""
        try:
            # Build search query with title
            title_words = re.findall(
                r"\b\w{3,}\b", title.lower()
            )  # Extract meaningful words
            title_query = " AND ".join(
                [f"ti:{word}" for word in title_words[:5]]
            )  # Use first 5 words

            queries_to_try = [title_query]

            # If we have authors, add author-based queries
            if authors:
                for author in authors[:2]:  # Try first 2 authors
                    # Extract last name (heuristic)
                    author_words = author.split()
                    if author_words:
                        last_name = author_words[-1].strip(".,;")
                        if len(last_name) > 2:
                            # Combine title and author
                            combined_query = f"({title_query}) AND au:{last_name}"
                            queries_to_try.insert(0, combined_query)  # Try this first

            # Try each query
            for query in queries_to_try:
                try:
                    params = {"search_query": query, "start": 0, "max_results": 3}

                    response = requests.get(
                        "http://export.arxiv.org/api/query", params=params, timeout=10
                    )
                    response.raise_for_status()

                    # Parse XML response
                    import xml.etree.ElementTree as ET

                    root = ET.fromstring(response.text)
                    ns = {"atom": "http://www.w3.org/2005/Atom"}

                    entries = root.findall("atom:entry", ns)
                    if entries:
                        # Take the first result
                        entry = entries[0]

                        arxiv_title = entry.find("atom:title", ns)
                        arxiv_title_text = (
                            arxiv_title.text.strip()
                            if arxiv_title is not None and arxiv_title.text is not None
                            else ""
                        )

                        # Check if this is a reasonable match
                        if self._titles_match(title, arxiv_title_text):
                            summary = entry.find("atom:summary", ns)
                            abstract_text = (
                                summary.text.strip()
                                if summary is not None and summary.text is not None
                                else ""
                            )

                            # Get ArXiv URL
                            abs_url = ""
                            for link in entry.findall("atom:link", ns):
                                if link.get("rel") == "alternate":
                                    abs_url = link.get("href", "")
                                    break

                            return {
                                "title": arxiv_title_text,
                                "abstract": abstract_text,
                                "abs_url": abs_url,
                            }

                except Exception as search_error:
                    logger.debug(f"Search query '{query}' failed: {search_error}")
                    continue

            return None

        except Exception as e:
            logger.debug(f"ArXiv title/author search failed: {e}")
            return None

    def _titles_match(
        self, bib_title: str, arxiv_title: str, threshold: float = 0.6
    ) -> bool:
        """Check if two titles are likely the same paper."""
        if not bib_title or not arxiv_title:
            return False

        # Simple word overlap check
        bib_words = set(re.findall(r"\b\w{3,}\b", bib_title.lower()))
        arxiv_words = set(re.findall(r"\b\w{3,}\b", arxiv_title.lower()))

        if not bib_words or not arxiv_words:
            return False

        overlap = len(bib_words.intersection(arxiv_words))
        union = len(bib_words.union(arxiv_words))

        similarity = overlap / union if union > 0 else 0
        return similarity >= threshold

    async def _search_arxiv_for_citation(
        self, search_terms: List[str]
    ) -> Optional[Dict[str, str]]:
        """Search ArXiv API for a paper based on search terms."""
        try:
            if not search_terms:
                return None

            # Build search query - be much more restrictive
            author_terms = [
                term for term in search_terms if term.isalpha() and len(term) > 2
            ]
            year_terms = [
                term for term in search_terms if term.isdigit() and len(term) == 4
            ]

            # Only proceed if we have meaningful search terms
            if not author_terms:
                logger.debug("No meaningful author terms found, skipping search")
                return None

            # Try different search strategies, but be more restrictive
            queries_to_try = []

            # Strategy 1: Author + year search (most restrictive, try first)
            if author_terms and year_terms:
                # Use AND for authors to be more restrictive
                author_query = " AND ".join(
                    [f"au:{term}" for term in author_terms[:2]]
                )  # Max 2 authors
                year_query = (
                    f"submittedDate:[{year_terms[0]}0101 TO {year_terms[0]}1231]"
                )
                queries_to_try.append(f"({author_query}) AND {year_query}")

            # Strategy 2: Just first author + year (if we have year)
            if author_terms and year_terms:
                first_author = author_terms[0]
                year_query = (
                    f"submittedDate:[{year_terms[0]}0101 TO {year_terms[0]}1231]"
                )
                queries_to_try.append(f"au:{first_author} AND {year_query}")

            # Strategy 3: Multiple authors with AND (no year)
            if len(author_terms) >= 2:
                author_query = " AND ".join([f"au:{term}" for term in author_terms[:2]])
                queries_to_try.append(author_query)

            # Strategy 4: Single author only if it's a distinctive name (length > 4)
            if len(author_terms) == 1 and len(author_terms[0]) > 4:
                queries_to_try.append(f"au:{author_terms[0]}")

            # Try each query until we find something reasonable
            for query in queries_to_try:
                try:
                    params = {
                        "search_query": query,
                        "start": 0,
                        "max_results": 5,  # Get a few more results to check
                    }

                    response = requests.get(
                        "http://export.arxiv.org/api/query", params=params, timeout=10
                    )
                    response.raise_for_status()

                    # Parse XML response
                    import xml.etree.ElementTree as ET

                    root = ET.fromstring(response.text)
                    ns = {"atom": "http://www.w3.org/2005/Atom"}

                    entries = root.findall("atom:entry", ns)
                    if entries:
                        # Check each result for relevance
                        for entry in entries:
                            title = entry.find("atom:title", ns)
                            title_text = (
                                title.text.strip()
                                if title is not None and title.text is not None
                                else ""
                            )

                            # Basic relevance check: title should contain some of our search terms
                            if self._is_result_relevant(
                                title_text, author_terms, year_terms
                            ):
                                summary = entry.find("atom:summary", ns)
                                abstract_text = (
                                    summary.text.strip()
                                    if summary is not None and summary.text is not None
                                    else ""
                                )

                                # Get ArXiv URL
                                abs_url = ""
                                for link in entry.findall("atom:link", ns):
                                    if link.get("rel") == "alternate":
                                        abs_url = link.get("href", "")
                                        break

                                logger.debug(f"Found relevant result: {title_text}")
                                return {
                                    "title": title_text,
                                    "abstract": abstract_text,
                                    "abs_url": abs_url,
                                }

                        # If no relevant results found, log and continue to next query
                        logger.debug(f"No relevant results found for query: {query}")

                except Exception as search_error:
                    logger.debug(f"Search query '{query}' failed: {search_error}")
                    continue

            logger.debug(f"No relevant papers found for search terms: {search_terms}")
            return None

        except Exception as e:
            logger.debug(f"ArXiv search failed for terms {search_terms}: {e}")
            return None

    def _is_result_relevant(
        self, title: str, author_terms: List[str], year_terms: List[str]
    ) -> bool:
        """Check if a search result is relevant to our citation."""
        if not title:
            return False

        title_lower = title.lower()

        # Check if title contains any of our author terms (could be in title for some papers)
        author_in_title = any(term.lower() in title_lower for term in author_terms)

        # Check if title contains year
        year_in_title = (
            any(year in title for year in year_terms) if year_terms else False
        )

        # For now, be conservative - only accept if we have some indication of relevance
        # This is a heuristic and could be improved
        if author_in_title or year_in_title:
            return True

        # If the title is very short or generic, it's probably not relevant
        if len(title.split()) < 3:
            return False

        # For now, be very conservative and reject most results
        # This will reduce false positives at the cost of some false negatives
        return False

    async def _download_and_extract_pdf_related_works(
        self, paper: ArxivPaper
    ) -> Optional[str]:
        """Download PDF from ArXiv and extract related works section."""
        try:
            # Construct PDF URL
            arxiv_id = paper.arxiv_id
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

            logger.info(f"Downloading PDF for {arxiv_id}")

            # Download the PDF with timeout
            response = requests.get(
                pdf_url, timeout=30
            )  # Reduced from 60 to 30 seconds
            response.raise_for_status()

            # Check PDF size - skip if too large
            content_length = len(response.content)
            if content_length > 50 * 1024 * 1024:  # Skip PDFs larger than 50MB
                logger.warning(
                    f"PDF too large ({content_length / 1024 / 1024:.1f}MB), skipping PDF extraction"
                )
                return None

            logger.info(f"Downloaded PDF ({content_length / 1024:.1f}KB)")

            # Create temporary file for PDF
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_pdf:
                temp_pdf.write(response.content)
                temp_pdf_path = temp_pdf.name

            try:
                # Extract text from PDF with timeout protection
                logger.info("Extracting text from PDF...")
                pdf_text = self._extract_text_from_pdf(temp_pdf_path)
                if not pdf_text:
                    logger.warning(f"Could not extract text from PDF for {arxiv_id}")
                    return None

                logger.info(f"Extracted {len(pdf_text)} characters from PDF")

                # Extract related works section from PDF text
                related_works = self._extract_related_works_from_pdf_text(pdf_text)
                if related_works:
                    logger.info(
                        f"Successfully extracted related works section from PDF for {arxiv_id} ({len(related_works)} chars)"
                    )
                    return related_works
                else:
                    logger.warning(
                        f"No related works section found in PDF text for {arxiv_id}"
                    )
                    return None

            finally:
                # Clean up temporary file
                try:
                    os.unlink(temp_pdf_path)
                except Exception as e:
                    logger.debug(f"Could not delete temporary PDF file: {e}")

        except requests.Timeout:
            logger.warning(f"PDF download timed out for {paper.arxiv_id}")
            return None
        except requests.RequestException as e:
            logger.warning(f"PDF download failed for {paper.arxiv_id}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error downloading/processing PDF for {paper.arxiv_id}: {e}")
            return None

    def _extract_text_from_pdf(self, pdf_path: str) -> Optional[str]:
        """Extract text from PDF file using PyPDF2."""
        try:
            with open(pdf_path, "rb") as file:
                pdf_reader = PyPDF2.PdfReader(file)

                # Check number of pages - skip if too many
                num_pages = len(pdf_reader.pages)
                if num_pages > 100:  # Skip very long papers
                    logger.warning(
                        f"PDF has too many pages ({num_pages}), skipping extraction"
                    )
                    return None

                logger.info(f"Processing PDF with {num_pages} pages")

                # Extract text from first 20 pages only (most papers have related works early)
                text_parts = []
                pages_to_process = min(num_pages, 20)

                for page_num in range(pages_to_process):
                    try:
                        page = pdf_reader.pages[page_num]
                        text = page.extract_text()
                        if text and len(text.strip()) > 10:
                            text_parts.append(text)
                    except Exception as e:
                        logger.debug(f"Error extracting text from page {page_num}: {e}")
                        continue

                if text_parts:
                    full_text = "\n".join(text_parts)
                    logger.info(
                        f"Extracted text from {len(text_parts)} pages ({len(full_text)} characters)"
                    )
                    return full_text
                else:
                    logger.warning("No text extracted from any pages")
                    return None

        except Exception as e:
            logger.warning(f"Error extracting text from PDF: {e}")
            return None

    def _extract_related_works_from_pdf_text(self, pdf_text: str) -> Optional[str]:
        """Extract related works section from PDF text."""
        try:
            # Build patterns for section detection in PDF text
            section_patterns = []
            for section_name in self.config.related_works_section_names:
                # Create case-insensitive patterns for PDF text
                section_patterns.append(section_name.lower())

            # Split text into lines and find section boundaries
            lines = pdf_text.split("\n")

            # Find the start of related works section
            start_idx = None

            for i, line in enumerate(lines):
                line_clean = line.strip().lower()
                for pattern in section_patterns:
                    if (
                        pattern in line_clean and len(line_clean) < 100
                    ):  # Likely a section header
                        # Check if this is a sentence (letter. Capital letter pattern)
                        sentence_pattern = re.search(
                            r"(?:\d*[a-zA-Z]+\d*)+\.\s+[a-zA-Z]", line_clean
                        )
                        if sentence_pattern:
                            continue

                        start_idx = i
                        break
                if start_idx is not None:
                    break

            if start_idx is None:
                logger.debug("No related works section found in PDF text")
                return None

            # Find the end of the section with improved detection
            end_idx = len(lines)
            section_content_lines = []

            # Process each line after the start
            for i in range(start_idx + 1, len(lines)):
                line_clean = lines[i].strip()

                # Skip empty lines
                if not line_clean:
                    continue

                # Check if this line contains a section header anywhere within it
                # Look for patterns like "text content.3. Section Header" or "text content. 3. Section Header"
                section_header_match = re.search(
                    r"\.?\s*(\d+\.?\s+[A-Z](?:[a-z]|[A-Z]|\s+[A-Z])[^.]*?)(?:\s|$)",
                    line_clean,
                )

                # check for a match like 2.1
                subsection_header_match = re.search(
                    r"\.?\s*(\d+\.\d+\s+[A-Z](?:[a-z]|[A-Z])[^.]*?)(?:\s|$)", line_clean
                )
                if section_header_match and not subsection_header_match:
                    # Found a section header within this line
                    section_header = section_header_match.group(1).strip()
                    header_start = section_header_match.start(1)

                    # Extract the content before the section header
                    content_before_header = line_clean[:header_start].strip()
                    if content_before_header.endswith("."):
                        content_before_header = content_before_header[
                            :-1
                        ].strip()  # Remove trailing period

                    # Add the content before the header to our section
                    if content_before_header:
                        section_content_lines.append(content_before_header)

                    # This is where the section ends
                    end_idx = i
                    logger.debug(
                        f"Found section header within line {i}: '{section_header}' (content before: '{content_before_header[:50]}...')"
                    )
                    break

                # Check for section headers at the beginning of lines (original logic)
                if len(line_clean) > 80:
                    section_content_lines.append(line_clean)
                    continue

                # Check for numbered sections at start of line
                # if re.match(r'^\d+\.?\s+[A-Z][a-z]', line_clean):
                if re.match(r"^\d+(?!\.\d)\s+[A-Z][a-z]\n", line_clean):
                    end_idx = i
                    logger.debug(
                        f"Found numbered section header at line {i}: '{line_clean}'"
                    )
                    break

                # Check for clear section header patterns
                clear_section_patterns = [
                    r"^methodology?$",
                    r"^approach?$",
                    r"^experiments?$",
                    r"^evaluation$",
                    r"^results?$",
                    r"^discussion$",
                    r"^conclusion$",
                    r"^conclusions?$",
                    r"^implementation$",
                    r"^future work$",
                    r"^acknowledgments?$",
                    r"^references?$",
                    r"^bibliography?$",
                    r"^appendix$",
                    r"^limitations?$",
                    r"^author contributions?$",
                ]

                line_lower = line_clean.lower()
                if any(
                    re.match(pattern, line_lower) for pattern in clear_section_patterns
                ):
                    end_idx = i
                    logger.debug(
                        f"Found clear section header at line {i}: '{line_clean}'"
                    )
                    break

                # Check for section headers that start with common words
                section_start_words = [
                    "method",
                    "approach",
                    "experiment",
                    "evaluation",
                    "result",
                    "discussion",
                    "conclusion",
                    "implementation",
                ]
                if (
                    any(line_lower.startswith(word) for word in section_start_words)
                    and len(line_clean.split()) <= 4
                    and line_clean[0].isupper()
                ):
                    # Check if previous line seems complete
                    prev_line_idx = i - 1
                    while (
                        prev_line_idx >= start_idx and not lines[prev_line_idx].strip()
                    ):
                        prev_line_idx -= 1

                    if prev_line_idx >= start_idx and (
                        lines[prev_line_idx].strip().endswith(".")
                        or lines[prev_line_idx].strip().endswith(")")
                        or len(lines[prev_line_idx].strip()) < 20
                    ):
                        end_idx = i
                        logger.debug(
                            f"Found likely section header at line {i}: '{line_clean}'"
                        )
                        break

                # If we haven't found a section header, add this line to content
                section_content_lines.append(line_clean)

            # Combine all the content lines
            section_content = "\n".join(section_content_lines).strip()

            # Basic validation
            if len(section_content) < 100:  # Too short
                logger.debug(
                    f"Related works section too short: {len(section_content)} characters"
                )
                return None

            # Clean up the text (remove excessive whitespace, fix line breaks)
            section_content = re.sub(
                r"\n\s*\n", "\n\n", section_content
            )  # Normalize paragraph breaks
            section_content = re.sub(
                r"[ \t]+", " ", section_content
            )  # Normalize spaces
            section_content = section_content.strip()

            logger.info(
                f"Extracted related works section: {len(section_content)} characters (lines {start_idx + 1} to {end_idx - 1})"
            )
            return section_content

        except Exception as e:
            logger.error(f"Error extracting related works from PDF text: {e}")
            return None
