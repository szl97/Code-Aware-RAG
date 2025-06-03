import re
from typing import List

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.util import ngrams

class NLPUtils:
    @staticmethod
    def removeExtraSpaces(text: str) -> str:
        """移除文本中的多余空格"""
        return re.sub(r'\s+', ' ', text).strip()

    @staticmethod
    def stem(text: str) -> str:
        """对文本进行词干提取"""
        stemmer = PorterStemmer()
        words = text.lower().split()
        return ' '.join([stemmer.stem(word) for word in words])

    @staticmethod
    def tokenize(text: str, keep_word_only: bool = True) -> List[dict]:
        """对文本进行分词"""
        tokens = nltk.word_tokenize(text.lower())
        result = []
        for token in tokens:
            token_type = 'word' if token.isalnum() else 'punctuation'
            if not keep_word_only or token_type == 'word':
                result.append({'type': token_type, 'value': token})
        return result

    @staticmethod
    def ngram(text: str, n: int) -> List[str]:
        """生成N元组"""
        tokens = text.lower().split()
        return [' '.join(gram) for gram in ngrams(tokens, n)]

    @staticmethod
    def removeWords(tokens: List[dict], stop_words=None) -> List[dict]:
        """移除停用词"""
        if stop_words is None:
            stop_words = set(stopwords.words('english'))
        return [token for token in tokens if token['value'].lower() not in stop_words]

    @staticmethod
    def setOfWords(tokens: List[dict]) -> List[dict]:
        """去重"""
        seen = set()
        result = []
        for token in tokens:
            if token['value'].lower() not in seen:
                seen.add(token['value'].lower())
                result.append(token)
        return result

