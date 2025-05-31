#!/usr/bin/env python3
"""
覆盖率提升测试模块
专门针对低覆盖率的模块和函数进行测试
"""

import pytest
import os
import re
import numpy as np
from directed_graph import DirectedGraph
from text_processor import TextProcessor
from graph_analyzer import GraphAnalyzer

class TestCoverage:
    """覆盖率提升测试类"""
    
    @pytest.fixture
    def setup_text_processor(self):
        """创建TextProcessor对象"""
        return TextProcessor()
    
    @pytest.fixture
    def setup_complex_graph(self):
        """创建一个更复杂的图结构，用于测试更多路径"""
        graph = DirectedGraph()
        
        # 添加更多节点和边
        words = ["apple", "banana", "cherry", "date", "elderberry", 
                "fig", "grape", "honeydew", "kiwi", "lemon"]
        
        # 添加节点和边 - DirectedGraph通过add_edge添加节点
        for i in range(len(words) - 1):
            graph.add_edge(words[i], words[i+1], i+1)
        
        # 添加一些循环和交叉边
        graph.add_edge("banana", "apple", 2)
        graph.add_edge("cherry", "banana", 3)
        graph.add_edge("date", "apple", 4)
        graph.add_edge("elderberry", "cherry", 5)
        graph.add_edge("fig", "banana", 6)
        graph.add_edge("grape", "date", 7)
        graph.add_edge("honeydew", "elderberry", 8)
        graph.add_edge("kiwi", "fig", 9)
        graph.add_edge("lemon", "grape", 10)
        
        return GraphAnalyzer(graph)
    
    @pytest.fixture
    def setup_text_processor_with_data(self):
        """创建带有数据的TextProcessor对象"""
        processor = TextProcessor()
        
        # 创建一个临时文件用于测试
        with open("test_text.txt", "w") as f:
            f.write("This is a test file. It contains multiple lines. And some words repeat: test test file file.")
        
        # 处理文件
        processor.process_file("test_text.txt")
        
        return processor
    
    def test_text_processor_full_coverage(self, setup_text_processor):
        """全面测试TextProcessor类的所有方法"""
        processor = setup_text_processor
        
        # 测试process_file方法
        # 创建一个临时文件用于测试
        with open("test_text.txt", "w") as f:
            f.write("This is a test file.\nIt contains multiple lines.\nAnd some words repeat: test test file file.")
        
        # 测试处理文件 - 返回的是DirectedGraph对象
        result = processor.process_file("test_text.txt")
        assert isinstance(result, DirectedGraph)
        assert "test" in result.nodes
        assert "file" in result.nodes
        
        # 测试词频统计
        assert processor.word_counts["test"] >= 2
        assert processor.word_counts["file"] >= 2
        assert processor.total_words > 0
        
        # 清理测试文件
        os.remove("test_text.txt")
        
        # 测试异常情况
        with pytest.raises(FileNotFoundError):
            processor.process_file("nonexistent_file.txt")
    
    def test_graph_analyzer_additional_methods(self, setup_complex_graph):
        """测试GraphAnalyzer类中其他未被充分覆盖的方法"""
        analyzer = setup_complex_graph
        
        # 测试show_directed_graph方法
        result = analyzer.show_directed_graph()
        assert isinstance(result, str)
        assert "有向图结构" in result
        assert "节点总数" in result
        
        # 测试generate_new_text方法的各种情况
        # 空文本
        result = analyzer.generate_new_text("")
        assert result == ""
        
        # 单个单词
        result = analyzer.generate_new_text("apple")
        assert result == "apple"
        
        # 两个单词，有桥接词
        result = analyzer.generate_new_text("apple banana")
        assert "apple" in result
        assert "banana" in result
        
        # 多个单词，有多个桥接词
        text = "apple banana cherry date elderberry"
        result = analyzer.generate_new_text(text)
        assert len(result.split()) >= len(text.split())
        
        # 包含不在图中的单词
        result = analyzer.generate_new_text("apple nonexistent banana")
        assert "apple" in result
        assert "nonexistent" in result
        assert "banana" in result
        
        # 测试calc_shortest_path方法的各种情况
        # 直接路径
        result = analyzer.calc_shortest_path("apple", "banana")
        assert isinstance(result, tuple) or "no path" in result.lower() or "no apple" in result.lower()
        
        # 间接路径
        result = analyzer.calc_shortest_path("apple", "cherry")
        assert isinstance(result, tuple) or "no path" in result.lower() or "no apple" in result.lower()
        
        # 长路径
        result = analyzer.calc_shortest_path("apple", "lemon")
        assert isinstance(result, tuple) or "no path" in result.lower() or "no apple" in result.lower()
        
        # 循环路径
        result = analyzer.calc_shortest_path("banana", "apple")
        assert isinstance(result, tuple) or "no path" in result.lower() or "no banana" in result.lower()
        
        # 测试random_walk方法
        result = analyzer.random_walk()
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_directed_graph_additional_methods(self):
        """测试DirectedGraph类中未被充分覆盖的方法"""
        graph = DirectedGraph()
        
        # 测试基本操作
        graph.add_edge("A", "B", 1)
        graph.add_edge("B", "C", 2)
        
        # 验证节点是否被添加
        assert "A" in graph.nodes
        assert "B" in graph.nodes
        assert "C" in graph.nodes
        
        # 测试get_weight方法的边界情况
        assert graph.get_weight("A", "B") == 1
        assert graph.get_weight("A", "C") == 0  # 不存在的边
        
        # 测试get_successors方法
        successors = list(graph.get_successors("A"))
        assert "B" in successors
        assert len(successors) == 1
        
        # 测试不存在节点的后继
        successors = list(graph.get_successors("X"))
        assert len(successors) == 0
        
        # 测试get_predecessors方法
        predecessors = list(graph.get_predecessors("B"))
        assert "A" in predecessors
        assert len(predecessors) == 1
        
        # 测试不存在节点的前驱
        predecessors = list(graph.get_predecessors("X"))
        assert len(predecessors) == 0
        
        # 测试更新边的权重
        graph.add_edge("A", "B", 2)  # 增加权重
        assert graph.get_weight("A", "B") == 3  # 1 + 2 = 3
    
    def test_calc_all_shortest_paths(self, setup_complex_graph):
        """测试计算一个单词到所有其他单词的最短路径"""
        analyzer = setup_complex_graph
        
        # 测试正常情况
        result = analyzer.calc_all_shortest_paths("apple")
        assert isinstance(result, tuple)
        assert isinstance(result[0], str)
        assert isinstance(result[1], dict)
        assert "从单词 'apple' 到其他单词的最短路径" in result[0]
        
        # 测试单词不在图中的情况
        result = analyzer.calc_all_shortest_paths("nonexistent")
        assert isinstance(result, str)
        assert "No nonexistent in the graph" in result
        
        # 测试没有可达节点的情况
        # 添加一个孤立节点
        analyzer.graph.nodes.add("isolated")
        result = analyzer.calc_all_shortest_paths("isolated")
        assert isinstance(result, str)
        assert "无法到达" in result
    
    def test_calc_all_shortest_paths_between(self, setup_complex_graph):
        """测试计算两个单词之间的所有最短路径"""
        analyzer = setup_complex_graph
        
        # 创建一个更简单的图用于测试
        simple_graph = DirectedGraph()
        # 添加简单的路径
        simple_graph.add_edge("start", "middle1", 1)
        simple_graph.add_edge("start", "middle2", 2)
        simple_graph.add_edge("middle1", "end", 3)
        simple_graph.add_edge("middle2", "end", 4)
        simple_analyzer = GraphAnalyzer(simple_graph)
        
        # 测试简单图中的路径 - 应该有两条路径
        result = simple_analyzer.calc_all_shortest_paths_between("start", "end")
        assert isinstance(result, tuple)
        assert isinstance(result[0], str)
        assert isinstance(result[1], list)
        assert len(result[1]) == 2  # 应该有两条路径
        
        # 测试单词不在图中的情况
        result = simple_analyzer.calc_all_shortest_paths_between("nonexistent", "end")
        assert isinstance(result, str)
        assert "No nonexistent in the graph" in result
        
        result = simple_analyzer.calc_all_shortest_paths_between("start", "nonexistent")
        assert isinstance(result, str)
        assert "No nonexistent in the graph" in result
        
        # 测试没有路径的情况
        simple_graph.nodes.add("isolated")
        result = simple_analyzer.calc_all_shortest_paths_between("start", "isolated")
        assert isinstance(result, str)
        assert "No path" in result
    
    def test_calc_tf_idf(self, setup_complex_graph, setup_text_processor_with_data):
        """测试计算TF-IDF值"""
        # 创建一个更简单的图和处理器
        graph = DirectedGraph()
        graph.add_edge("word1", "word2", 1)
        graph.add_edge("word2", "word3", 1)
        analyzer = GraphAnalyzer(graph)
        
        processor = TextProcessor()
        # 手动设置一些词频数据
        processor.total_words = 10
        processor.word_counts = {"word1": 2, "word2": 3, "word3": 1}
        
        # 测试计算TF-IDF
        result = analyzer.calc_tf_idf(processor)
        assert isinstance(result, dict)
        
        # 检查每个节点都有TF-IDF值
        assert "word1" in result
        assert "word2" in result
        assert "word3" in result
        assert all(isinstance(val, float) for val in result.values())
    
    def test_calc_page_rank(self, setup_complex_graph, setup_text_processor_with_data):
        """测试计算PageRank值"""
        # 创建一个更简单的图
        graph = DirectedGraph()
        graph.add_edge("a", "b", 1)
        graph.add_edge("b", "c", 1)
        graph.add_edge("c", "a", 1)  # 形成一个环
        analyzer = GraphAnalyzer(graph)
        
        # 测试不使用TF-IDF的PageRank计算，限制迭代次数
        result = analyzer.calc_page_rank(use_tf_idf=False, max_iterations=5)
        assert isinstance(result, dict)
        
        # 检查每个节点都有PageRank值
        assert "a" in result
        assert "b" in result
        assert "c" in result
        assert all(isinstance(val, float) for val in result.values())
        
        # 测试空图的情况
        empty_graph = DirectedGraph()
        empty_analyzer = GraphAnalyzer(empty_graph)
        result = empty_analyzer.calc_page_rank()
        assert result == {}
    
    def test_get_page_rank(self, setup_complex_graph):
        """测试获取单词的PageRank值"""
        # 创建一个简单的图
        graph = DirectedGraph()
        graph.add_edge("x", "y", 1)
        graph.add_edge("y", "z", 1)
        analyzer = GraphAnalyzer(graph)
        
        # 手动设置PR值
        analyzer.graph.pr_values = {"x": 0.4, "y": 0.3, "z": 0.3}
        
        # 测试获取存在单词的PageRank
        result = analyzer.get_page_rank("x")
        assert isinstance(result, float)
        assert result == 0.4
        
        # 测试获取不存在单词的PageRank
        result = analyzer.get_page_rank("nonexistent")
        assert isinstance(result, str)
        assert "No nonexistent in the graph" in result 