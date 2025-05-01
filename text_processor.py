import re
from collections import defaultdict
from directed_graph import DirectedGraph

class TextProcessor:
    def __init__(self):
        self.graph = DirectedGraph()
        self.word_counts = defaultdict(int)  # 用于存储每个单词的出现次数
        self.total_words = 0  # 文档中的总单词数
    
    def process_file(self, file_path):
        """处理文本文件，构建有向图"""
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # 将文本转换为小写，并替换标点符号为空格
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # 移除非字母字符并分割成单词
        words = re.findall(r'\b[a-z]+\b', text)
        
        # 计算单词频率统计
        self.total_words = len(words)
        for word in words:
            self.word_counts[word] += 1
        
        # 构建有向图
        for i in range(len(words) - 1):
            self.graph.add_edge(words[i], words[i+1])
        
        return self.graph 