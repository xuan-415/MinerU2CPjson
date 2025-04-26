import os
import re
import time
import json
import nltk
import torch
import spacy
import logging
import hashlib
import argparse
from nltk.corpus import words
from typing import List, Dict
from bs4 import BeautifulSoup
from datetime import datetime
from datasketch import MinHash, MinHashLSH
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# create log directory (if not exists)
log_dir = './log/Paper'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# generate log file name with timestamp
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_filename = f'paper_processor_{timestamp}.log'
log_path = os.path.join(log_dir, log_filename)

# set log configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_path),  # 將日誌寫入檔案
        logging.StreamHandler()  # 同時輸出到控制台
    ]
)

logger = logging.getLogger(__name__)

class PaperProcessor:
    def __init__(self, folder_path, output_path):
        """
            Initialize the PaperProcessor with input and output paths.

            Args:
                folder_path (str): The path to the folder containing parsed paper files.
                output_path (str): The path where the processed JSON output will be saved.

            Attributes:
                nlp (spacy.Language): The SpaCy NLP model used for entity recognition.
                tokenizer (GPT2Tokenizer): The GPT-2 tokenizer used for perplexity calculation.
                model (GPT2LMHeadModel): The GPT-2 model used for perplexity calculation.
        """
        self.folder_path = folder_path
        self.output_path = output_path
        self.nlp = spacy.load("en_core_web_sm")
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.model = GPT2LMHeadModel.from_pretrained("gpt2")

        logger.info(f"Initialized PaperProcessor with folder_path: {folder_path}, output_path: {output_path}")

    def load_json(self, file_path):
        """
            Load a JSON file and return the data as a list of dictionaries.

            Args:
                file_path (str): The path to the JSON file.

            Returns:
                list: A list of dictionaries containing the data from the JSON file.
        """

        logger.debug(f"Loading JSON file: {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.debug(f"Successfully loaded JSON file with {len(data)} entries")
        return data

    def load_md(self, file_path):
        """
            Load a Markdown (.md) file as a text string.
            
            Args:
                file_path (str): The path to the Markdown file.
        
            Returns:
                str: The content of the Markdown file as a string.
                Returns an empty string if the file is not found.
        """

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                md_content = f.read()
        except FileNotFoundError:
            logger.warning(f"Corresponding .md file not found: {file_path}")
            md_content = ""
        return md_content
    
    def to_json(self, dataset, output_path):
        """
            Save a dataset into a JSON file with indentation and UTF-8 encoding.

            Args:
                dataset (list): The data to be saved, usually a list of dictionaries.
                output_path (str): The path where the JSON file will be saved.

            Raises:
                Exception: If there is an error during file saving.
        """

        logger.info(f"Saving dataset to {output_path}")
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(dataset, f, indent=4, ensure_ascii=False)
            logger.info(f"Successfully saved dataset with {len(dataset)} entries")
        except Exception as e:
            logger.error(f"Error saving JSON file {output_path}: {str(e)}")
            raise

    def filter_document(self, data):
        """
            Check if the data is qualified (whether there is an entry with page_idx > 1).

            Args:
                data (list): The data to be checked, usually a list of dictionaries.
                md_content (str): The content of the Markdown file.

            Returns:
                bool: True if the data is qualified, False otherwise.
        """
        logger.debug("Filtering content based on page_idx")
        for entry in data:
            if entry.get("page_idx", 0) > 1:
                logger.debug("Found valid content with page_idx > 1")
                return True
        logger.debug("No valid content found with page_idx > 1")
        return False

    def detect_first_letter(self, data):
        """
            Detect whether the first letter of the text following an "introduction" section is uppercase.

            This function searches for entries in the data where the text contains the word "introduction"
            (case-insensitive). For each such occurrence, it checks if the next item's text starts with
            an uppercase letter. It counts and returns the number of times the following text does **not**
            start with an uppercase letter, indicating a potential formatting issue.
            
            Args:
                data (list): A list of dictionaries representing parsed document sections.

            Returns:
                int: The number of times a post-introduction section starts with a lowercase letter or has no valid following content.
        """
        detect_cnt = 0
        for i, item in enumerate(data):
            if "text_level" in item:
                # Use regex to check if the text contains "introduction"
                # This will match "introduction" regardless of case and surrounding characters
                if re.search(r'\bintroduction\b', item["text"].lower()):
                    logger.info(f"Found introduction in item: {item['text']}")
                    
                    # Check if there's a next item
                    if i + 1 < len(data):
                        next_item = data[i + 1]
                        if "text" in next_item and next_item["text"]:
                            next_text = next_item["text"]
                            
                            # Check if the first letter of the next item's text is uppercase
                            if next_text and len(next_text) > 0:
                                first_letter = next_text[0]
                                is_uppercase = first_letter.isupper()
                                if not is_uppercase:
                                    detect_cnt += 1        
                                logger.info(f"Next item starts with: '{first_letter}', Is uppercase: {is_uppercase}")
                                
                                # You can store this information or take action as needed
                                item["next_item_starts_with_uppercase"] = is_uppercase
                            else:
                                logger.warning(f"Next item has empty text")
                        else:
                            logger.warning(f"No next item after introduction")
                            detect_cnt += 1
        return detect_cnt
    
    def count_stuck_together_words(self, groups, length_threshold=45):
        """
            Count the number of stuck together words.

            Args:
                groups (list): The list of groups to be counted.
                length_threshold (int): The length threshold of the stuck together words.

            Returns:
                int: 0 if no stuck together words, 1 if there are stuck together words.

        """
        pattern = r"(//|/|\\[a-zA-Z]+|[()\[\]{}]|-|[0-9]|@)"
        remove_groups = []

        for group in groups:
            stuck_cnt = 0
            for item in group:
                parts = item["Text"].split()
                for part in parts:
                    if len(part) > length_threshold and re.search(pattern, part) is None:
                        stuck_cnt += 1
                        if stuck_cnt >= 5:
                            remove_groups.append(group)
                            break
                if stuck_cnt >= 5:
                    logger.info(f"Removing group: {group}")
                    return 1
        return 0
    
    def filters(self, data):
        """
            Filter out unwanted sections, personal information, and unnecessary citations from the parsed data.

            This function performs several filtering steps:
            - Removes sections based on specific keywords (e.g., "Supporting Information", "References").
            - Skips the first page if it matches known unwanted pages (e.g., "Accepted Manuscript").
            - Removes sensitive information like personal names, URLs, emails, phone numbers, and server addresses.
            - Cleans text by removing citation references like [1], [2-5], and redundant punctuation.

            Args:
                data (list): A list of dictionaries representing parsed document sections.

            Returns:
                list: The filtered list of dictionaries with unwanted entries removed and text cleaned.

        """
        sections_to_remove = [
            "Supporting Information", "Acknowledgements", "Acknowledgement", "Keywords", "keyword", "articleinfo", "All rights reserved."
            "Corresponding Author", "NOMENCLATURE", "Article history", "Conflict of Interest", "References", "Received", "Revised", 
            "Publishedcff online", "Funding", "Reference"
        ]

        page_to_remove = ["Accepted Manuscript", "Journal Pre-proofs"]
        
        skip_first_page = False
        for item in data:
            if "text" in item:
                if any(page.lower() in item["text"].lower() for page in page_to_remove):
                    skip_first_page = True
                break
        
        pattern_parts = []
        for section in sections_to_remove:
            section_chars = [re.escape(c.lower()) for c in section.strip()]
            if section_chars:
                last_char = section_chars[-1]
                section_chars[-1] = f"{last_char}[-:\\s]*" 
            spaced_section = r'\s*'.join(section_chars)
            pattern_parts.append(spaced_section)
        
        pattern = '|'.join(pattern_parts)
        citation_pattern = r'([.,!?;]?)(\[\d+(?:[-\s.,]\d+)*\])([.,!?;]?)'
        
        filtered_data = []
        skip_section = False
        skip_next = False

        for entry in data:
            if skip_next:
                skip_next = False
                continue

            # 如果需要跳過第一頁，檢查page_idx
            if skip_first_page and entry.get("page_idx", 1) == 0:
                continue

            if "text_level" in entry:
                current_text = entry["text"].lower().strip()
                if re.search(pattern, current_text):
                    skip_section = True
                    logger.debug(f"Skipping section containing: {current_text}")
                    continue
                else:
                    skip_section = False
                    logger.debug(f"Starting new section: {current_text}")

            if not skip_section:
                if "text" in entry:
                    if self.filter_name_url_email_phone_serveraddress(entry["text"]):
                        cleaned_text = re.sub(citation_pattern, r'\1\3', entry["text"])
                        cleaned_text = re.sub(r'[.,!?;]{2,}', lambda m: m.group(0)[0], cleaned_text)
                        new_entry = entry.copy()
                        new_entry["text"] = cleaned_text.strip()
                        
                        if new_entry["text"]:
                            filtered_data.append(new_entry)
                        else:
                            logger.info(f"Filtered out empty text after cleaning: {entry['text'][:100]}...") 
                    else:
                        logger.info(f"Filtered out text containing sensitive information: {entry['text'][:100]}...")
                else:
                    filtered_data.append(entry)

        logger.info(f"Filtered {len(data) - len(filtered_data)} entries")
        return filtered_data
    
    def filter_name_url_email_phone_serveraddress(self, text):
        """
            Filter out text containing sensitive information.

            Args:
                text (str): The text to be filtered.

            Returns:
                bool: True if the text is filtered out, False otherwise.
        """
        person_name_cnt = 0
        if self._detect_non_math_text(text):
            doc = self.nlp(text)
            for idx, ent in enumerate(doc.ents):
                if ent.label_ == "PERSON" and len(ent.text) >= 5:
                    person_name_cnt += 1    

        # URL pattern (simple version, matches http(s) or www)
        url_pattern = r'(?:https?://|www\.)\S+'

        # Email pattern
        email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'

        # Phone number pattern (simple version, matches common formats like 123-456-7890, (123) 456-7890, +1234567890)
        phone_pattern = r'(?:(?:\+\d{1,3})?[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})'

        # SSH/Server Address pattern (assume matches IP address or domain, e.g. 192.168.1.1, example.com:22)
        server_pattern = r'(?:\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(?::\d+)?|\S+\.(?:com|org|net|edu|gov)(?::\d+)?)'

        # Calculate the number of matches
        url_count = len(re.findall(url_pattern, text))
        email_count = len(re.findall(email_pattern, text))
        phone_count = len(re.findall(phone_pattern, text))
        server_count = len(re.findall(server_pattern, text))

        # Set the condition: if person_name_cnt >= 3 or any other item >= 1, return False
        if person_name_cnt >= 3 or url_count >= 1 or email_count >= 1 or phone_count >= 1 or server_count >= 1:
            return False
        else:
            return True
    
    def _detect_non_math_text(self, text):
        """
            Detect whether there is any text in $...$ format.

            Args:
                text (str): The text to be detected.

            Returns:
                bool: True if there is no text in $...$ format, False otherwise.
        """
        pattern = r'\$[^\$]+\$'
        matches = re.finditer(pattern, text)
        
        # If no text in $...$ format is found, return True
        return not bool(list(matches))

    def reorder_paragraphs(self, data):
        """
            Reorder paragraphs in a document based on punctuation and perplexity rules to improve logical flow.

            This function detects text blocks that start with a lowercase letter (potentially misplaced fragments).
            It then attempts to reposition such fragments after nearby paragraphs that do not end with proper
            punctuation, guided by calculating the language model perplexity between candidate pairs.

            Args:
                data (list): A list of dictionaries, each representing a text block or structural element.

            Returns:
                list: A reordered list of dictionaries with better paragraph logical flow.
        """
        reordered_data = data.copy()
        n = len(data)
        
        locked = [False] * n
        
        i = 0
        while i < n:
            if locked[i] or "text" not in reordered_data[i] or "text_level" in reordered_data[i] or "equation" in reordered_data[i - 1]:
                i += 1
                continue
            
            current_text = reordered_data[i]["text"]
            current_starts_with_lowercase = self._starts_with_lowercase(current_text)
            
            if current_starts_with_lowercase and i > 0:             
                candidates = []
                candidates_index = []
                
                prev_count = 0
                j = i - 1
                while j >= 0 and prev_count < 3:
                    if reordered_data[j]["type"] == "text" and "text_level" not in reordered_data[j] and not locked[j]:
                        candidates.append((j, self._has_punctuation_ending(reordered_data[j]["text"])))
                        candidates_index.append(j)
                        prev_count += 1
                    j -= 1
                
                next_count = 0
                j = i + 1
                while j < n and next_count < 3:
                    if reordered_data[j]["type"] == "text" and "text_level" not in reordered_data[j] and not locked[j]:
                        candidates.append((j, self._has_punctuation_ending(reordered_data[j]["text"])))
                        candidates_index.append(j)
                        next_count += 1
                    j += 1

                position = []
                perplexities = []
                found = False
                
                for idx, has_punctuation in candidates:
                    if not has_punctuation:
                        position.append(idx)
                        candidate_text = reordered_data[idx]["text"]
                        perplexity = self._calculate_perplexity(candidate_text, current_text)
                        perplexities.append(perplexity)
                        logger.info(f"Perplexity for item {i + 1} following item {idx + 1}: {perplexity}")
                        
                if position and min(perplexities) < 60:  
                    best_idx = position[perplexities.index(min(perplexities))]
                    logger.info(f"Moving item {i + 1} to follow item {best_idx + 1} (lowest perplexity: {min(perplexities)}).")
                    
                    item_to_move = reordered_data.pop(i)
                    reordered_data.insert(best_idx, item_to_move)
                    
                    start_lock = max(0, candidates_index[0])
                    end_lock = min(n - 1, candidates_index[-1])
                    
                    for j in range(start_lock, end_lock + 1):
                        locked[j] = True
                    found = True
            i += 1
        
        return reordered_data

    def _has_punctuation_ending(self, paragraph):
        #Check if a paragraph ends with a punctuation mark.
        punctuation = r'[.!?:]$'
        return bool(re.search(punctuation, paragraph.strip()))

    def _starts_with_lowercase(self, paragraph):
        #Check if a paragraph starts with a lowercase letter.
        if not isinstance(paragraph, str) or not paragraph.strip():
            return False
        
        first_char = paragraph.strip()[0]
        
        return first_char.isalpha() and first_char.islower()
    
    def _calculate_perplexity(self, text1, text2):
        #Calculate the perplexity of two texts.
        combined_text = text1 + " " + text2
        # 使用 tokenizer 的 truncation 參數來處理過長的序列
        inputs = self.tokenizer(combined_text, return_tensors="pt", truncation=True, max_length=1024)
        input_ids = inputs["input_ids"]
        
        with torch.no_grad():
            outputs = self.model(input_ids, labels=input_ids)
            loss = outputs.loss
            perplexity = torch.exp(loss).item()
        
        return perplexity
    
    def merge_text(self, data, path_name, min_length=250):
        """
            Merge consecutive text, table, and equation entries into larger text blocks.

            This function consolidates smaller pieces of text, table descriptions, and equations
            into combined text segments. Each combined segment is saved once it reaches the
            specified minimum length threshold. The merging starts from the main article section
            and skips metadata or front matter like "Accepted Manuscript".

            Args:
                data (list): A list of dictionaries representing parsed paper content.
                path_name (str): The folder or filename associated with the paper (for metadata recording).
                min_length (int, optional): The minimum length (in characters) for a merged text block. Default is 250.

            Returns:
                list: A list of dictionaries, each containing merged text and associated metadata.
        """
        
        logger.info("Starting text merging process")
        dataset = []
        buffer_text = ""  # 用於暫存合併的文字
        article = ""
        article_idx = 0
        for idx, item in enumerate(data):
            if "text" in item and item["text"] != "Accepted Manuscript":
                if "text_level" in item and len(item["text"]) >= 15:
                    article = item["text"]
                    article_idx = idx
                    break
            else: 
                if "text" in item:
                    article = item["text"]
                    article_idx = idx
                    break

        for item in data[article_idx:]:
            item_type = item["type"]
            if item_type == "text":
                text = self._process_text(item)
                if text:
                    buffer_text = self._combine_text(buffer_text, text)
            elif item_type == "table":
                table_text = self._process_table(item)
                buffer_text = self._combine_text(buffer_text, table_text)
            elif item_type == "equation":
                equation = item["text"]
                buffer_text = self._combine_text(buffer_text, equation)

            if len(buffer_text) >= min_length:
                dataset.append({
                    "Folder Name": path_name,
                    "Paper Name": article,
                    "Text": buffer_text
                })
                buffer_text = ""  

        logger.info(f"Text merging complete. Generated {len(dataset)} entries")
        return dataset

    def _process_text(self, item):
        #Process text items.
        if "text_level" in item:
            return item["text"]
        qualified, text = self.filter_text(item["text"])
        return text if qualified else ""

    def filter_text(self, text):
        """
            Filter and clean text.

            Args:
                text (str): The text to be filtered and cleaned.

            Returns:
                tuple: A tuple containing a boolean value and the filtered text.
        """
        logger.debug(f"Filtering text of length: {len(text)}")
        if len(text) < 20:
            logger.debug("Text too short, filtering out")
            return False, text
        text = text.replace('\xa0', ' ').replace('\u25a0', ' ').replace('\u2013', ' ').replace('\\', '').replace('\u0142', '').replace('\xb2', '').strip()
        logger.debug("Text successfully filtered")
        return True, text    

    def _process_table(self, item):
        #Process table items.
        if "table_body" in item:
            table_text = self.table2text(item["table_body"])
            if item.get("table_caption"):
                table_text = item["table_caption"][0] + "\n" + table_text
            return table_text
        return ""

    def _combine_text(self, buffer, new_text):
        #Combine text, handle line breaks.
        return buffer + "\n" + new_text if buffer else new_text

    def table2text(self, html_content):
        """
            Convert HTML tables to text.

            Args:
                html_content (str): The HTML content containing the table.

            Returns:
                str: The text representation of the table.
        """

        logger.debug("Converting HTML table to text")
        soup = BeautifulSoup(html_content, 'html.parser')
        table = soup.find('table')
        result = []
        current_category = None

        for row in table.find_all('tr'):
            cells = row.find_all('td')
            if len(cells) == 1 and cells[0].has_attr('colspan'):
                current_category = cells[0].get_text(strip=True)
                result.append(f"Category: {current_category}")
            else:
                row_data = [cell.get_text(strip=True) for cell in cells if cell.get_text(strip=True)]
                if row_data:
                    result.append(", ".join(row_data))

        logger.debug(f"Converted table with {len(result)} rows")
        return "\n".join(result)

    def deduplicate(self, dataset: List[Dict]) -> List[Dict]:
        """
            Run the full deduplication process on a dataset.

            This method performs two deduplication stages in sequence:
            (1) Paragraph-level similarity-based deduplication using MinHash.
            (2) Exact match deduplication based on SHA-256 hash.

            Args:
                dataset (List[Dict]): A list of entries, each with a "Text" field to be deduplicated.

            Returns:
                List[Dict]: A cleaned dataset with duplicate entries removed.
        """
        logger.info("============================================================================================")
        logger.info("Starting deduplication process")

        # Step 1: Paragraph-level similarity-based deduplication using MinHash.
        update_dataset = self.update_deduplicate_dataset(dataset, "paragraph")
        # Step 2: Exact match deduplication based on SHA-256 hash.
        update_dataset = self.update_deduplicate_dataset(update_dataset, "exact_match")
        logger.info("============================================================================================")

        return update_dataset

    def update_deduplicate_dataset(self, dataset: List[Dict], deduplicate_type: str) -> List[Dict]:
        """
            Perform a specific deduplication step and update the dataset with deduplicated results.

            Args:
                dataset (List[Dict]): Original dataset with "Text" field.
                deduplicate_type (str): Type of deduplication: "paragraph" or "exact_match".

            Returns:
                List[Dict]: Updated dataset with deduplicated text and preserved metadata.
        """
        logger.info("Starting deduplication step")
        texts = [item['Text'] for item in dataset]
        
        if deduplicate_type == "exact_match":
            logger.info(f"Starting deduplication step ({deduplicate_type})")
            deduped_results = self.exact_match_deduplication(texts)
        elif deduplicate_type == "paragraph":
            logger.info(f"Starting deduplication step ({deduplicate_type})")
            deduped_results = self.paragraph_deduplicate(texts)
        else:
            logger.warning(f"Unknown deduplication type: {deduplicate_type}")
            return dataset
        
        updated_dataset = []
        
        for result in deduped_results:
            deduped_text = result['text']
            original_index = result['original_index']
            
            if original_index < len(dataset):  
                new_item = dataset[original_index].copy()
                new_item['Text'] = deduped_text
                
                if deduped_text:
                    updated_dataset.append(new_item)
        
        logger.info(f"After {deduplicate_type} deduplication: {len(updated_dataset)} entries")
        return updated_dataset

    def paragraph_deduplicate(self, documents: List[str], threshold=0.9, num_perm=128) -> List[Dict]:
        """
            Perform paragraph-level fuzzy deduplication using MinHash and LSH.

            Args:
                documents (List[str]): List of document paragraphs to deduplicate.
                threshold (float): Jaccard similarity threshold for considering duplicates. Default is 0.9.
                num_perm (int): Number of permutations for MinHash. Default is 128.

            Returns:
                List[Dict]: List of deduplicated paragraphs, each with original index.
        """
        logger.info("Starting paragraph-level deduplication")
        lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        seen_paragraphs = {}  # 用於存儲已見過的段落的MinHash值
        unique_paragraphs = []  # 用於存儲唯一的段落
        original_indices = []  # 保存原始文檔的索引
        duplicate_count = 0
        duplicate_pairs = []  # 儲存重複段落對
        duplicate_hashes = set()  # 用於追蹤重複段落的MinHash值

        for doc_id, doc in enumerate(documents):
            doc = doc.strip()
            if not doc:
                continue
                
            m = MinHash(num_perm=num_perm)
            words = re.findall(r'\w+', doc.lower())
            for word in words:
                m.update(word.encode('utf8'))
        
            if lsh.query(m):
                duplicate_count += 1
                similar_para = seen_paragraphs.get(str(m))
                if similar_para:
                    similar_words = set(re.findall(r'\w+', similar_para['text'].lower()))
                    current_words = set(re.findall(r'\w+', doc.lower()))
                    actual_similarity = len(similar_words & current_words) / len(similar_words | current_words)
                    
                    if actual_similarity >= threshold: 
                        duplicate_pairs.append({
                            'original_doc_id': similar_para['original_index'],
                            'duplicate_doc_id': doc_id,
                            'original': similar_para['text'],
                            'duplicate': doc,
                            'similarity': actual_similarity
                        })
                        logger.info(f"Found duplicate paragraph in document {doc_id}")
                        logger.info(f"Actual similarity: {actual_similarity:.2f}")
                        logger.info("Original paragraph:")
                        logger.info(similar_para['text'])
                        logger.info("Duplicate paragraph:")
                        logger.info(doc)
                        logger.info("-" * 80)
                        duplicate_hashes.add(str(m))
            else:
                lsh.insert(f"doc{doc_id}", m)
                seen_paragraphs[str(m)] = {'text': doc, 'original_index': doc_id}
                unique_paragraphs.append(doc)
                original_indices.append(doc_id)

        logger.info(f"Paragraph deduplication completed. Found {duplicate_count} duplicate paragraphs")
        logger.info(f"Original paragraphs: {len(documents)}, Unique paragraphs: {len(unique_paragraphs)}")
        
        if duplicate_pairs:
            logger.info("\nDuplicate Paragraphs Summary:")
            for pair in duplicate_pairs:
                logger.info(f"Original document: {pair['original_doc_id']}, Duplicate document: {pair['duplicate_doc_id']}")
                logger.info(f"Similarity: {pair['similarity']:.2f}")
                logger.info("-" * 40)
        
        return [{"text": text, "original_index": idx} for text, idx in zip(unique_paragraphs, original_indices)]

    def exact_match_deduplication(self, documents: List[str]) -> List[Dict]:
        """
            Perform exact-match deduplication using SHA-256 hash comparison.

            Args:
                documents (List[str]): List of document strings.

            Returns:
                List[Dict]: List of deduplicated documents, each with original index.
        """
        logger.info("Starting exact match deduplication")
        document_hashes = {}
        unique_documents = []
        original_indices = [] 
        duplicate_count = 0
        duplicate_pairs = []  

        for doc_id, doc in enumerate(documents):
            doc_hash = hashlib.sha256(doc.encode('utf8')).hexdigest()
            
            if doc_hash not in document_hashes:
                document_hashes[doc_hash] = (doc_id, doc)
                unique_documents.append(doc)
                original_indices.append(doc_id)  
            else:
                duplicate_count += 1
                original_doc_id, original_doc = document_hashes[doc_hash]
                
                duplicate_pairs.append({
                    'original_doc_id': original_doc_id,
                    'duplicate_doc_id': doc_id,
                    'original': original_doc,
                    'duplicate': doc
                })
                logger.info(f"Found duplicate document: original {original_doc_id}, duplicate {doc_id}")
                logger.info(f"Document content preview: {doc[:100]}...")
                logger.info("-" * 80)

        logger.info(f"Exact match deduplication completed. Found {duplicate_count} duplicate documents")
        logger.info(f"Original documents: {len(documents)}, Unique documents: {len(unique_documents)}")
        
        if duplicate_pairs:
            logger.info("\nDuplicate Documents Summary:")
            logger.info(f"Total duplicate documents: {duplicate_count}")
            logger.info("-" * 40)
        
        return [{"text": text, "original_index": idx} for text, idx in zip(unique_documents, original_indices)]

    # def remove_figure_references(self, data):
    #     """移除文章中 Fig. 或 Figure 後面跟著數字的內容"""
    #     # 定義匹配模式
    #     # (?:Fig\.|Figure) 匹配 "Fig." 或 "Figure" 而不捕獲
    #     # \s* 匹配零個或多個空白字符
    #     # \d+(?:\.\d+)* 匹配數字（包括小數點格式如 2.1.3）
    #     # (?:-\d+)? 可選的連字符後跟數字（匹配如 2-4）
    #     pattern = r'(?:Fig\.|Figure)\s*\d+(?:\.\d+)*(?:-\d+)?'
        
    #     # 使用 re.sub 將匹配的部分替換為空字串
    #     for item in data:
    #         if item["type"] == "text":
    #             cleaned_text = re.sub(pattern, '', item["text"])
    #             item["text"] = ' '.join(cleaned_text.split())
        
    #     # 移除多餘的空格並返回
    #     return data 
        
    def process(self):
        """
            Process all files in the input folder and generate the final deduplicated dataset.

            This function walks through the specified folder structure, reads parsed paper content
            (content_list.json + .md files), filters and cleans the data,
            reorders paragraphs if necessary, merges smaller text units into larger blocks,
            performs deduplication, and finally saves the cleaned dataset as a JSON file.

            Steps:
                1. Traverse the folder tree to find 'auto' subfolders.
                2. For each paper, load the content list and corresponding markdown file.
                3. Apply filtering rules (page index, introduction checks, stuck-together word detection).
                4. Filter unwanted sections and sensitive information.
                5. Reorder paragraphs based on logical rules.
                6. Merge text, tables, and equations into larger text blocks.
                7. Deduplicate the dataset using paragraph-level and exact match methods.
                8. Save the final output to a JSON file.

        """
        logger.info("Starting paper processing")

        dataset = []
        cnt = 0
        for root, dirs, files in os.walk(self.folder_path):
            if os.path.basename(root) == "auto":
                paper_path = os.path.dirname(root)
                paper_name = os.path.basename(paper_path)

                logger.info(f"Processing paper: {paper_name}")

                content_list_files = [f for f in files if f.endswith("_content_list.json")]
                for content_list_file in content_list_files:
                    md_file = content_list_file.replace("_content_list.json", ".md")
                    content_list_path = os.path.join(root, content_list_file)
                    md_path = os.path.join(root, md_file)
                    
                    logger.info(f"Processing files: {content_list_path} and {md_path}")
                    cnt += 1
                    
                    json_data = self.load_json(content_list_path)
                    md_content = self.load_md(md_path)
                    if self.filter_document(json_data, md_content) and self.detect_first_letter(json_data) == 0 and self.count_stuck_together_words(json_data) == 0:
                        filtered_data = self.filters(json_data)
                        reorder_data = self.reorder_paragraphs(filtered_data)
                        dataset.extend(self.merge_text(reorder_data, path_name=content_list_path.split("/")[-1]))

        dataset = self.deduplicate(dataset)
        self.to_json(dataset, self.output_path)
        logger.info(f"Processing complete. Total files processed: {cnt}, Total entries in dataset: {len(dataset)}")


def main():

    parser = argparse.ArgumentParser(description="Process academic papers for filtering and deduplication.")
    parser.add_argument(
        "--input_folder", 
        type=str, 
        required=True, 
        help="Path to the folder containing parsed paper results (should include 'auto' subfolders)"
    )
    parser.add_argument(
        "--output_path", 
        type=str, 
        required=True, 
        help="Path to save the final deduplicated, filtered and merged JSON output"
    )

    args = parser.parse_args()

    start_time = time.time()
    processor = PaperProcessor(args.input_folder, args.output_path)
    processor.process()

    # calculate execution time
    end_time = time.time()
    execution_time = end_time - start_time

    hours = int(execution_time // 3600)
    minutes = int((execution_time % 3600) // 60)
    seconds = int(execution_time % 60)
    
    logger.info(f"Application completed successfully")
    logger.info(f"Total execution time: {hours} hours, {minutes} minutes, {seconds} seconds")
    logger.info(f"Total execution time in seconds: {execution_time:.2f}")
    


if __name__ == "__main__":
    main()