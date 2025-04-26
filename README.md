# MinerU2json

A document processing tool for cleaning, filtering, and deduplicating parsed academic papers. Supports paragraph reordering, merging, and advanced logging.

---

## Features

- **Flexible Input Parsing**: Processes JSON+Markdown parsed papers.
- **Sensitive Information Filtering**: Removes URLs, emails, phone numbers, server addresses, and excess personal names.
- **Section Filtering**: Skips sections like "References", "Supporting Information", etc.
- **Paragraph Reordering**: Corrects paragraph orders based on punctuation and GPT-2 perplexity.
- **Merging**: Combines text, table, and equation entries into coherent text blocks.
- **Deduplication**:
  - Paragraph-level deduplication (MinHash LSH, fuzzy matching)
  - Exact-match deduplication (SHA-256 hashing)
- **Detailed Logging**: Logs to both console and timestamped files.
- **Command-Line Interface**: Easily specify input and output paths with arguments.

## Installation

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

**Key Libraries:**
- `spacy`
- `nltk`
- `datasketch`
- `beautifulsoup4`
- `transformers`
- `torch`

## Usage

```bash
python Paperfilter.py --input_folder path/to/parsed_papers --output_path path/to/output.json
```

Example:

```bash
python Paperfilter.py --input_folder ./output/Paper --output_path ./output/all_papers_v1.json
```

## Folder Structure

```
input_folder/
  ├── Paper_A/
  │    └── auto/
  │         ├── paperA_content_list.json
  │         └── paperA.md
  ├── Paper_B/
  │    └── auto/
  │         ├── paperB_content_list.json
  │         └── paperB.md
...
```

## Output Format

Each item in the output JSON:

```json
{
  "Folder Name": "Paper_A",
  "Paper Name": "Title of Paper",
  "Text": "Cleaned and merged text content."
}
```

## Advanced Options

- **MinHash Deduplication Threshold**: Default 0.9 similarity.
- **Perplexity Threshold for Reordering**: Default 60.
- **Minimum Merge Length**: Default 250 characters.

---

## License

This project is licensed under the MIT License.

## Contact

For issues, feature requests, or questions, please open an issue on GitHub.

---

> This README was generated based on the Paperfilter.py functionality (version 1).
