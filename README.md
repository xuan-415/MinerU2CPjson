# Paperfilter with Deduplication v1

A document processing tool for academic papers that includes advanced paragraph-level deduplication, exact-match deduplication, filtering, and paragraph reordering.

---

## Features

- **Paragraph-Level Deduplication**  
  Use MinHash LSH to identify and remove highly similar paragraphs.

- **Exact-Match Deduplication**  
  Hash-based detection of exactly identical paragraphs across or within documents.

- **Sensitive Information Filtering**  
  Filters out content containing excessive personal names, URLs, emails, phone numbers, and server addresses.

- **Paragraph Reordering**  
  Reorders misplaced paragraphs based on punctuation endings and perplexity scores computed with GPT-2.

- **Text Merging**  
  Merges text, table contents, and equations into coherent blocks for dataset generation.

- **Logging**  
  Detailed logging to both file and console, with timestamped log files.

## Folder Structure

```
output/
    Paper/
        <Paper Folders>
            auto/
                *_content_list.json
                *.md
    all_paper_v5.json (output file)
log/
    Paper/
        paper_processor_<timestamp>.log
```

## Installation

```bash
pip install -r requirements.txt
```

**Required Libraries:**
- `spacy`
- `nltk`
- `beautifulsoup4`
- `datasketch`
- `transformers`
- `torch`

Download the SpaCy English model if not already installed:

```bash
python -m spacy download en_core_web_sm
```

## Usage

Modify the following lines in `main()` according to your folder paths:

```python
processor = PaperProcessor("../../output/Paper", "../output/all_paper_v5.json")
processor.process()
```

Then simply run:

```bash
python Paperfilter_with_deduplicate_v1.py
```

## Workflow Overview

1. Scan folders containing parsed academic papers.
2. Filter out invalid or noisy sections.
3. Reorder misplaced paragraphs based on perplexity and punctuation heuristics.
4. Merge text, tables, and equations.
5. Apply paragraph-level deduplication.
6. Apply exact-match deduplication.
7. Save the cleaned dataset into a JSON file.

## Output Format

Each dataset entry will look like:

```json
{
  "Folder Name": "Paper123",
  "Paper Name": "Title of Paper",
  "Text": "Merged and cleaned text content."
}
```

## Notes

- The current perplexity model used is GPT-2 (`gpt2`).
- Logging will create a timestamped `.log` file in `./log/Paper/`.
- The deduplication threshold for MinHash is set to 0.9 similarity.

---

## License

This project is licensed under the MIT License.

## Contact

For questions or suggestions, feel free to open an issue or contact the maintainer.
