#------------------------------------------------------------------------------------------------                                                                                          |
#|                        相比v0版本，增加段落級別去重，並且增加log記錄               |                                                       |                                                                                  |
#------------------------------------------------------------------------------------------------

#=======================================================================================================
import os
import re
import time
import json
import nltk
import torch
import spacy
import logging
import hashlib
from nltk.corpus import words
from typing import List, Dict
from bs4 import BeautifulSoup
from datetime import datetime
from datasketch import MinHash, MinHashLSH
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# 創建log目錄（如果不存在）
log_dir = './log/Paper'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# 生成帶時間戳記的日誌檔名
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_filename = f'paper_processor_{timestamp}.log'
log_path = os.path.join(log_dir, log_filename)

# 設置日誌配置
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
        """初始化類別，設定資料夾路徑並載入 SpaCy 模型"""
        self.folder_path = folder_path
        self.output_path = output_path
        self.nlp = spacy.load("en_core_web_sm")
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.model = GPT2LMHeadModel.from_pretrained("gpt2")


        logger.info(f"Initialized PaperProcessor with folder_path: {folder_path}, output_path: {output_path}")

    def load_json(self, file_path):
        """載入 JSON 檔案"""
        logger.debug(f"Loading JSON file: {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.debug(f"Successfully loaded JSON file with {len(data)} entries")
        return data

    def loda_md(self, file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                md_content = f.read()
        except FileNotFoundError:
            logger.warning(f"Corresponding .md file not found: {file_path}")
            md_content = ""
        return md_content

    def table2text(self, html_content):
        """將 HTML 表格轉換為文字"""
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

    def filter_content(self, data, md_content=""):
        """檢查資料是否合格（是否有 page_idx > 1 的條目）"""
        logger.debug("Filtering content based on page_idx")
        for entry in data:
            if entry.get("page_idx", 0) > 1:
                logger.debug("Found valid content with page_idx > 1")
                return True
        logger.debug("No valid content found with page_idx > 1")
        return False

    def filter_text(self, text):
        """過濾和清理文字"""
        logger.debug(f"Filtering text of length: {len(text)}")
        if len(text) < 20:
            logger.debug("Text too short, filtering out")
            return False, text
        text = text.replace('\xa0', ' ').replace('\u25a0', ' ').replace('\u2013', ' ').replace('\\', '').replace('\u0142', '').replace('\xb2', '').strip()
        logger.debug("Text successfully filtered")
        return True, text
    
    def detect_first_letter(self, data):
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
    
    def detect_non_math_text(self, text):
        # 使用正則表達式找出 $...$ 格式的文字
        pattern = r'\$[^\$]+\$'
        matches = re.finditer(pattern, text)
        
        # 如果沒有找到任何 $...$ 格式的文字，回傳 True
        return not bool(list(matches))

    
    def filter_name_url_email_phone_serveraddress(self, text):

        person_name_cnt = 0
        if self.detect_non_math_text(text):
            doc = self.nlp(text)
            for idx, ent in enumerate(doc.ents):
                if ent.label_ == "PERSON" and len(ent.text) >= 5:
                    person_name_cnt += 1    

        # URL模式（簡單版，匹配http(s)或www開頭）
        url_pattern = r'(?:https?://|www\.)\S+'

        # Email模式
        email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'

        # 電話號碼模式（簡單版，匹配常見格式如123-456-7890, (123) 456-7890, +1234567890）
        phone_pattern = r'(?:(?:\+\d{1,3})?[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})'

        # SSH/Server Address模式（假設匹配IP地址或域名，如192.168.1.1, example.com:22）
        server_pattern = r'(?:\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(?::\d+)?|\S+\.(?:com|org|net|edu|gov)(?::\d+)?)'

        # 3. 計算匹配數量
        url_count = len(re.findall(url_pattern, text))
        email_count = len(re.findall(email_pattern, text))
        phone_count = len(re.findall(phone_pattern, text))
        server_count = len(re.findall(server_pattern, text))

        # 4. 設置條件：如果人名>=3 或其他任何一項>=1，則返回False
        if person_name_cnt >= 3 or url_count >= 1 or email_count >= 1 or phone_count >= 1 or server_count >= 1:
            return False
        else:
            return True
    
    def group_by_consecutive_papers(self, data):
        """將資料按連續的 Paper Name 分組"""
        groups = []
        current_group = []
        current_title = None

        for item in data:
            if item["Paper Name"] != current_title:
                if current_group:
                    groups.append(current_group)
                current_group = [item]
                current_title = item["Paper Name"]
            else:
                current_group.append(item)

        if current_group:
            groups.append(current_group)

        return groups


    def count_stuck_together_words(self, groups, length_threshold=45):
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
        """過濾掉不需要的部分"""
        sections_to_remove = [
            "Supporting Information", "Acknowledgements", "Acknowledgement", "Keywords", "keyword", "articleinfo", "All rights reserved."
            "Corresponding Author", "NOMENCLATURE", "Article history",
            "Conflict of Interest", "References", "Received", "Revised", "Publishedcff online", "Funding", "Reference"
        ]

        page_to_remove = ["Accepted Manuscript", "Journal Pre-proofs"]
        
        # 檢查第一個text是否在page_to_remove中
        skip_first_page = False
        for item in data:
            if "text" in item:
                if any(page.lower() in item["text"].lower() for page in page_to_remove):
                    skip_first_page = True
                break
        
        # 改進模式：允許每個字符之間有任意空格，結尾可能有符號
        pattern_parts = []
        for section in sections_to_remove:
            # 將每個字符之間加入\s*（零個或多個空格），結尾允許有符號
            section_chars = [re.escape(c.lower()) for c in section.strip()]
            # 對最後一個字符特殊處理，允許後面跟著符號（包括連字符、冒號等）
            if section_chars:
                last_char = section_chars[-1]
                section_chars[-1] = f"{last_char}[-:\\s]*"  # 加入連字符到允許的結尾符號中
            spaced_section = r'\s*'.join(section_chars)
            pattern_parts.append(spaced_section)
        pattern = '|'.join(pattern_parts)
        
        # 修改模式，使用捕獲組保留前後的標點
        citation_pattern = r'([.,!?;]?)(\[\d+(?:[-\s.,]\d+)*\])([.,!?;]?)'
        
        filtered_data = []
        skip_section = False
        skip_next = False

        for i, entry in enumerate(data):
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
                        # 保留前後標點，只移除中括號部分
                        cleaned_text = re.sub(citation_pattern, r'\1\3', entry["text"])
                        cleaned_text = re.sub(r'[.,!?;]{2,}', lambda m: m.group(0)[0], cleaned_text)
                        new_entry = entry.copy()
                        new_entry["text"] = cleaned_text.strip()
                        
                        if new_entry["text"]:
                            filtered_data.append(new_entry)
                        else:
                            logger.info(f"Filtered out empty text after cleaning: {entry['text'][:100]}...")  # 只記錄前100個字符
                    else:
                        logger.info(f"Filtered out text containing sensitive information: {entry['text'][:100]}...")  # 只記錄前100個字符
                else:
                    filtered_data.append(entry)

        logger.info(f"Filtered {len(data) - len(filtered_data)} entries")
        return filtered_data
    
    def remove_figure_references(self, data):
        """移除文章中 Fig. 或 Figure 後面跟著數字的內容"""
        # 定義匹配模式
        # (?:Fig\.|Figure) 匹配 "Fig." 或 "Figure" 而不捕獲
        # \s* 匹配零個或多個空白字符
        # \d+(?:\.\d+)* 匹配數字（包括小數點格式如 2.1.3）
        # (?:-\d+)? 可選的連字符後跟數字（匹配如 2-4）
        pattern = r'(?:Fig\.|Figure)\s*\d+(?:\.\d+)*(?:-\d+)?'
        
        # 使用 re.sub 將匹配的部分替換為空字串
        for item in data:
            if item["type"] == "text":
                cleaned_text = re.sub(pattern, '', item["text"])
                item["text"] = ' '.join(cleaned_text.split())
        
        # 移除多餘的空格並返回
        return data 
    
    def paragraph_deduplicate(self, documents: List[str], threshold=0.9, num_perm=128) -> List[str]:
        """段落級別的去重"""
        logger.info("Starting paragraph-level deduplication")
        lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        seen_paragraphs = {}  # 用於存儲已見過的段落的MinHash值
        unique_paragraphs = []  # 用於存儲唯一的段落
        duplicate_count = 0
        duplicate_pairs = []  # 儲存重複段落對
        duplicate_hashes = set()  # 用於追蹤重複段落的MinHash值

        for doc_id, doc in enumerate(documents):
            # 直接使用輸入的文檔作為段落，只做基本的清理
            doc = doc.strip()
            if not doc:  # 只忽略空段落
                continue
                
            m = MinHash(num_perm=num_perm)
            # 使用更細緻的詞分割
            words = re.findall(r'\w+', doc.lower())
            for word in words:
                m.update(word.encode('utf8'))
        
            if lsh.query(m):
                duplicate_count += 1
                # 找到相似段落
                similar_para = seen_paragraphs.get(str(m))
                if similar_para:
                    # 計算實際相似度
                    similar_words = set(re.findall(r'\w+', similar_para.lower()))
                    current_words = set(re.findall(r'\w+', doc.lower()))
                    actual_similarity = len(similar_words & current_words) / len(similar_words | current_words)
                    
                    if actual_similarity >= threshold:  # 只在實際相似度達到閾值時才記錄
                        duplicate_pairs.append({
                            'original': similar_para,
                            'duplicate': doc,
                            'doc_id': doc_id,
                            'similarity': actual_similarity
                        })
                        logger.info(f"Found duplicate paragraph in document {doc_id}")
                        logger.info(f"Actual similarity: {actual_similarity:.2f}")
                        logger.info("Original paragraph:")
                        logger.info(similar_para)
                        logger.info("Duplicate paragraph:")
                        logger.info(doc)
                        logger.info("-" * 80)
                        # 將重複段落的MinHash值加入集合
                        duplicate_hashes.add(str(m))
            else:
                lsh.insert(f"doc{doc_id}", m)
                seen_paragraphs[str(m)] = doc
                unique_paragraphs.append(doc)

        # 移除重複的段落
        unique_paragraphs = [para for para in unique_paragraphs 
                           if str(hashlib.sha256(para.encode('utf8')).hexdigest()) not in duplicate_hashes]

        logger.info(f"Paragraph deduplication completed. Found {duplicate_count} duplicate paragraphs")
        logger.info(f"Original paragraphs: {len(documents)}, Unique paragraphs: {len(unique_paragraphs)}")
        
        # 記錄重複段落的統計資訊
        if duplicate_pairs:
            logger.info("\nDuplicate Paragraphs Summary:")
            for pair in duplicate_pairs:
                logger.info(f"Document {pair['doc_id']}")
                logger.info(f"Similarity: {pair['similarity']:.2f}")
                logger.info("-" * 40)
        
        return unique_paragraphs

    def exact_match_deduplication(self, documents: List[str]) -> List[str]:
        """精確匹配去重"""
        logger.info("Starting exact match deduplication")
        paragraph_hashes: Dict[str, tuple] = {}
        deduped_docs: List[List[str]] = [[] for _ in range(len(documents))]
        duplicate_count = 0
        duplicate_pairs = []  # 儲存重複段落對
        MIN_PARAGRAPH_LENGTH = 150  # 最小段落長度

        for doc_id, doc in enumerate(documents):
            paragraphs = self.split_into_paragraphs(doc)
            logger.info(f"Processing document {doc_id} with {len(paragraphs)} paragraphs")
            
            for para_id, para in enumerate(paragraphs):
                # 如果段落太短，直接加入結果中
                if len(para) < MIN_PARAGRAPH_LENGTH:
                    deduped_docs[doc_id].append(para)
                    continue
                    
                # 只有當段落長度足夠時才進行去重比較
                para_hash = hashlib.sha256(para.encode('utf8')).hexdigest()
                
                if para_hash not in paragraph_hashes:
                    paragraph_hashes[para_hash] = (doc_id, para)
                    deduped_docs[doc_id].append(para)
                else:
                    # 如果段落已存在，記錄重複但不加入 deduped_docs
                    duplicate_count += 1
                    original_doc_id, original_para = paragraph_hashes[para_hash]
                    
                    if original_doc_id == doc_id:
                        # 文檔內重複
                        duplicate_pairs.append({
                            'type': 'within_document',
                            'doc_id': doc_id,
                            'para_id': para_id,
                            'original': original_para,
                            'duplicate': para
                        })
                        logger.info(f"Found duplicate paragraph within document {doc_id}, paragraph {para_id}")
                    else:
                        # 跨文檔重複
                        duplicate_pairs.append({
                            'type': 'between_documents',
                            'original_doc_id': original_doc_id,
                            'duplicate_doc_id': doc_id,
                            'para_id': para_id,
                            'original': original_para,
                            'duplicate': para
                        })
                        logger.info(f"Found duplicate paragraph between documents {original_doc_id} and {doc_id}")
                
                    logger.info("Original paragraph:")
                    for line in original_para.split('\n'):
                        logger.info(line)
                    logger.info("Duplicate paragraph:")
                    for line in para.split('\n'):
                        logger.info(line)
                    logger.info("-" * 80)

        result = ['.'.join(paras) for paras in deduped_docs if paras]
        logger.info(f"Exact match deduplication completed. Found {duplicate_count} duplicate paragraphs")
        logger.info(f"Original documents: {len(documents)}, Unique documents: {len(result)}")
        
        # 記錄重複段落的統計資訊
        if duplicate_pairs:
            logger.info("\nDuplicate Paragraphs Summary:")
            within_doc_count = sum(1 for pair in duplicate_pairs if pair['type'] == 'within_document')
            between_doc_count = sum(1 for pair in duplicate_pairs if pair['type'] == 'between_documents')
            logger.info(f"Within-document duplicates: {within_doc_count}")
            logger.info(f"Between-documents duplicates: {between_doc_count}")
            logger.info("-" * 40)
        
        return result
    
    def update_deduplicate_dataset(self, dataset: List[Dict], deduplicate_type: str) -> List[Dict]:
        """執行去重並更新原始 dataset，移除空文檔"""
        logger.info("Starting third deduplication step (exact match)")
        # 提取文本
        texts = [item['Text'] for item in dataset]
        # 執行去重
        if deduplicate_type == "exact_match":
            logger.info(f"Starting deduplication step ({deduplicate_type})")
            deduped_texts = self.exact_match_deduplication(texts)

        elif deduplicate_type == "paragraph":
            logger.info(f"Starting deduplication step ({deduplicate_type})")
            deduped_texts = self.paragraph_deduplicate(texts)
        
        # 更新 dataset 中的 'Text' 字段，並過濾空文檔
        updated_dataset = []
        for i, (item, deduped_text) in enumerate(zip(dataset, deduped_texts)):
            new_item = item.copy()
            new_item['Text'] = deduped_text
            # 只保留非空文檔
            if new_item['Text']:
                updated_dataset.append(new_item)
        
        logger.info(f"After {deduplicate_type} deduplication: {len(updated_dataset)} entries")
        return updated_dataset
    
    def deduplicate(self, dataset: List[Dict]) -> List[Dict]:
        logger.info("============================================================================================")
        logger.info("Starting deduplication process")

        # Step 1: 段落級別去重
        update_dataset = self.update_deduplicate_dataset(dataset, "paragraph")
        logger.info("============================================================================================")
        # Step 2: 精確匹配去重
        update_dataset = self.update_deduplicate_dataset(update_dataset, "exact_match")

        return update_dataset
    

    def split_into_paragraphs(self, document: str) -> List[str]:
        """將文檔分割成段落，以句點作為分隔符，但排除數字中的小數點。短段落會被合併到下一段。"""
        # 使用正向預查(?<!\d)和正向後查(?!\d)來確保句點前後不是數字
        # 同時確保句點前後有空格，避免將縮寫等誤判為句子結束
        raw_paragraphs = [p.strip() for p in re.split(r'(?<!\d)\.(?!\d)', document) if p.strip()]
        # paragraphs = [p for p in raw_paragraphs if len(p) >= 40]
        
        merged_paragraphs = []
        current_paragraph = ""
        
        for i, para in enumerate(raw_paragraphs):
            if len(para) < 40:  # 如果段落太短
                if i < len(raw_paragraphs) - 1:  # 如果不是最後一段
                    # 將當前段落與下一段合併
                    current_paragraph = para + ". " + raw_paragraphs[i + 1]
                    continue  # 跳過下一段，因為已經合併
                else:  # 如果是最後一段
                    # 如果已經有累積的段落，將最後一段加入
                    if current_paragraph:
                        current_paragraph += ". " + para
                    else:
                        current_paragraph = para
            else:  # 如果段落夠長
                if current_paragraph:  # 如果有累積的段落
                    # 將累積的段落與當前段落合併
                    current_paragraph += ". " + para
                else:
                    current_paragraph = para
            
            # 當段落夠長時，加入結果列表
            if len(current_paragraph) >= 150:
                merged_paragraphs.append(current_paragraph)
                current_paragraph = ""
        
        # 處理最後可能殘留的段落
        if current_paragraph:
            merged_paragraphs.append(current_paragraph)
        
        return merged_paragraphs
        # return paragraphs

    
    def merge_text(self, data, path_name, min_length=250):
        """合併文字、表格和方程式，並生成資料集"""
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
        # logger.debug(f"Found article title: {article}")

        # 遍歷每個條目
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

            # 當累積文字達到最小長度時，添加到資料集
            if len(buffer_text) >= min_length:
                dataset.append({
                    "Folder Name": path_name,
                    "Paper Name": article,
                    "Text": buffer_text
                })
                buffer_text = ""  # 重置緩衝區

        logger.info(f"Text merging complete. Generated {len(dataset)} entries")
        return dataset

    def _process_text(self, item):
        """處理文字條目"""
        if "text_level" in item:
            return item["text"]
        qualified, text = self.filter_text(item["text"])
        return text if qualified else ""

    def _process_table(self, item):
        """處理表格條目"""
        if "table_body" in item:
            table_text = self.table2text(item["table_body"])
            if item.get("table_caption"):
                table_text = item["table_caption"][0] + "\n" + table_text
            return table_text
        return ""

    def _combine_text(self, buffer, new_text):
        """合併文字，處理換行符"""
        return buffer + "\n" + new_text if buffer else new_text
    
    def to_json(self, dataset, output_path):
        """將資料集保存為 JSON 檔案"""
        logger.info(f"Saving dataset to {output_path}")
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(dataset, f, indent=4, ensure_ascii=False)
            logger.info(f"Successfully saved dataset with {len(dataset)} entries")
        except Exception as e:
            logger.error(f"Error saving JSON file {output_path}: {str(e)}")
            raise

    def calculate_perplexity(self, text1, text2):
        """
        使用 GPT-2 計算兩個文本拼接後的困惑度。
        如果文本太長，會截斷到模型的最大長度。
        """
        combined_text = text1 + " " + text2
        # 使用 tokenizer 的 truncation 參數來處理過長的序列
        inputs = self.tokenizer(combined_text, return_tensors="pt", truncation=True, max_length=1024)
        input_ids = inputs["input_ids"]
        
        with torch.no_grad():
            outputs = self.model(input_ids, labels=input_ids)
            loss = outputs.loss
            perplexity = torch.exp(loss).item()
        
        return perplexity
    # # 2. 检查段落结尾是否为标点符号
    def has_punctuation_ending(self, paragraph):
        punctuation = r'[.!?:]$'
        return bool(re.search(punctuation, paragraph.strip()))

    # 3. 檢查段落開頭是否為小寫字母
    def starts_with_lowercase(self, paragraph):
        # 檢查輸入是否為字串且不為空
        if not isinstance(paragraph, str) or not paragraph.strip():
            return False
        
        # 提取第一個字符
        first_char = paragraph.strip()[0]
        
        # 檢查第一個字符是否為字母且為小寫
        return first_char.isalpha() and first_char.islower()

    # 4. 重排序逻辑：基于标点符号和首字母大写规则，仅检查前后两个段
    def reorder_paragraphs(self, data):
        reordered_data = data.copy()
        n = len(data)
        
        # 记录哪些段落已被调整（锁定）
        locked = [False] * n
        
        i = 0
        while i < n:
            # 跳过已锁定的段落或非文本项
            if locked[i] or "text" not in reordered_data[i] or "text_level" in reordered_data[i] or "equation" in reordered_data[i - 1]:
                i += 1
                continue
            
            # 检查当前段落开头是否为小写
            current_text = reordered_data[i]["text"]
            current_starts_with_lowercase = self.starts_with_lowercase(current_text)
            
            if current_starts_with_lowercase and i > 0:  # 当前段落以小写开头，且不是第一个段落             
                # 检查前两个段落（i-1 和 i-2）和后两个段落（i+1 和 i+2）
                candidates = []
                candidates_index = []
                
                # 前两个段落：找前面两個是text且沒有text_level的段落
                prev_count = 0
                j = i - 1
                while j >= 0 and prev_count < 3:
                    if reordered_data[j]["type"] == "text" and "text_level" not in reordered_data[j] and not locked[j]:
                        candidates.append((j, self.has_punctuation_ending(reordered_data[j]["text"])))
                        candidates_index.append(j)
                        prev_count += 1
                    j -= 1
                
                # 后两个段落：找後面兩個是text且沒有text_level的段落
                next_count = 0
                j = i + 1
                while j < n and next_count < 3:
                    if reordered_data[j]["type"] == "text" and "text_level" not in reordered_data[j] and not locked[j]:
                        candidates.append((j, self.has_punctuation_ending(reordered_data[j]["text"])))
                        candidates_index.append(j)
                        next_count += 1
                    j += 1

                # 寻找结尾没有标点符号的段落
                position = []
                perplexities = []
                found = False
                
                for idx, has_punctuation in candidates:
                    if not has_punctuation:  # 找到结尾没有标点符号的段落
                        position.append(idx)
                        candidate_text = reordered_data[idx]["text"]
                        perplexity = self.calculate_perplexity(candidate_text, current_text)
                        perplexities.append(perplexity)
                        logger.info(f"Perplexity for item {i + 1} following item {idx + 1}: {perplexity}")
                        
                if position and min(perplexities) < 60:  # 如果找到至少一個結尾無標點的段落
                    # 選擇困惑度最低的段落作為插入位置
                    best_idx = position[perplexities.index(min(perplexities))]
                    logger.info(f"Moving item {i + 1} to follow item {best_idx + 1} (lowest perplexity: {min(perplexities)}).")
                    
                    item_to_move = reordered_data.pop(i)
                    # 調整索引，因為 pop 操作會影響後續段落
                    reordered_data.insert(best_idx, item_to_move)
                    
                    # 鎖定被移動的段落及其周圍的段落
                    # 計算要鎖定的起始和結束索引，確保不超出邊界
                    start_lock = max(0, candidates_index[0])
                    end_lock = min(n - 1, candidates_index[-1])
                    
                    # 鎖定範圍內的所有相關段落
                    for j in range(start_lock, end_lock + 1):
                        locked[j] = True
                    found = True
            i += 1
        
        return reordered_data

    
    def count_stuck_together_words(self, data, length_threshold=45):
        """處理文本列表，計算每個文本的粘連部分數量和列表"""
        # 載入詞典
        stuck_cnt = 0
        
        # 透過 for 迴圈處理每個文本
        for item in data:
            if "text" in item and "text_level" not in item:
                text = item["text"]
                parts = text.split()
                pattern = r"(//|/|\\[a-zA-Z]+|[()\[\]{}]|-|[0-9]|@)"
                for part in parts:
                    if len(part) > length_threshold and re.search(pattern, part) is None:
                        stuck_cnt += 1
                        logger.info(f"Stuck together words: {part}")
        if stuck_cnt >= 5:
            return 1 
        else:
            return 0
        
    def process(self):
        """處理所有檔案並生成資料集"""
        logger.info("Starting paper processing")


        dataset = []
        cnt = 0
        for root, dirs, files in os.walk(self.folder_path):
            if os.path.basename(root) == "auto":
                # 獲取教科書名稱（auto的父目錄）
                paper_path = os.path.dirname(root)
                paper_name = os.path.basename(paper_path)

                logger.info(f"Processing paper: {paper_name}")

                content_list_files = [f for f in files if f.endswith("_content_list.json")]
                for content_list_file in content_list_files:
                    # 獲取對應的 .md 文件名
                    md_file = content_list_file.replace("_content_list.json", ".md")
                    content_list_path = os.path.join(root, content_list_file)
                    md_path = os.path.join(root, md_file)
                    
                    logger.info(f"Processing files: {content_list_path} and {md_path}")
                    cnt += 1
                    
                    # 讀取兩個文件的內容
                    json_data = self.load_json(content_list_path)
                    md_content = self.loda_md(md_path)
                    if self.filter_content(json_data, md_content) and self.detect_first_letter(json_data) == 0 and self.count_stuck_together_words(json_data) == 0:
                        filtered_data = self.filters(json_data)
                        reorder_data = self.reorder_paragraphs(filtered_data)
                        dataset.extend(self.merge_text(reorder_data, path_name=content_list_path.split("/")[-1]))

        dataset = self.deduplicate(dataset)
        self.to_json(dataset, self.output_path)
        logger.info(f"Processing complete. Total files processed: {cnt}, Total entries in dataset: {len(dataset)}")


def main():
    # 記錄開始時間
    start_time = time.time()
    processor = PaperProcessor("../../output/Paper", "../output/all_paper_v5.json")
    processor.process()
    # 計算執行時間
    end_time = time.time()
    execution_time = end_time - start_time

    # 將執行時間轉換為小時、分鐘和秒
    hours = int(execution_time // 3600)
    minutes = int((execution_time % 3600) // 60)
    seconds = int(execution_time % 60)
    
    logger.info(f"Application completed successfully")
    logger.info(f"Total execution time: {hours} hours, {minutes} minutes, {seconds} seconds")
    logger.info(f"Total execution time in seconds: {execution_time:.2f}")
    


if __name__ == "__main__":
    main()