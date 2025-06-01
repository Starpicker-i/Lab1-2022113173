#!/usr/bin/env python3
"""
高覆盖率测试模块
针对directed_graph.py, graph_analyzer.py和text_processor.py进行全面测试
目标是达到至少90%的代码覆盖率
"""

import pytest
import os
import tempfile
import random
from unittest import mock
from collections import deque

from directed_graph import DirectedGraph
from graph_analyzer import GraphAnalyzer
from text_processor import TextProcessor

class TestDirectedGraph:
    """测试DirectedGraph类的所有方法"""
    
    def test_init(self):
        """测试初始化"""
        graph = DirectedGraph()
        assert graph.nodes == set()
        assert isinstance(graph.edges, dict)
        assert graph.pr_values == {}
    
    def test_add_edge(self):
        """测试添加边"""
        graph = DirectedGraph()
        
        # 测试添加新边
        graph.add_edge("a", "b", 2)
        assert "a" in graph.nodes
        assert "b" in graph.nodes
        assert graph.edges["a"]["b"] == 2
        
        # 测试更新已有边的权重
        graph.add_edge("a", "b", 3)
        assert graph.edges["a"]["b"] == 5  # 2 + 3
    
    def test_get_weight(self):
        """测试获取边的权重"""
        graph = DirectedGraph()
        
        # 添加边
        graph.add_edge("a", "b", 2)
        
        # 测试获取已有边的权重
        assert graph.get_weight("a", "b") == 2
        
        # 测试获取不存在的边的权重
        assert graph.get_weight("a", "c") == 0
        assert graph.get_weight("c", "b") == 0
    
    def test_get_successors(self):
        """测试获取后继节点"""
        graph = DirectedGraph()
        
        # 添加边
        graph.add_edge("a", "b", 1)
        graph.add_edge("a", "c", 2)
        
        # 测试获取已有节点的后继
        successors = list(graph.get_successors("a"))
        assert len(successors) == 2
        assert "b" in successors
        assert "c" in successors
        
        # 测试获取不存在节点的后继
        assert list(graph.get_successors("d")) == []
    
    def test_get_predecessors(self):
        """测试获取前驱节点"""
        graph = DirectedGraph()
        
        # 添加边
        graph.add_edge("a", "c", 1)
        graph.add_edge("b", "c", 2)
        
        # 测试获取已有节点的前驱
        predecessors = graph.get_predecessors("c")
        assert len(predecessors) == 2
        assert "a" in predecessors
        assert "b" in predecessors
        
        # 测试获取不存在节点的前驱
        assert graph.get_predecessors("d") == []

class TestTextProcessor:
    """测试TextProcessor类的所有方法"""
    
    def test_init(self):
        """测试初始化"""
        processor = TextProcessor()
        assert isinstance(processor.graph, DirectedGraph)
        assert processor.word_counts == {}
        assert processor.total_words == 0
    
    def test_process_file(self):
        """测试处理文本文件"""
        # 创建临时文件
        with tempfile.NamedTemporaryFile(mode='w+', delete=False, encoding='utf-8') as f:
            f.write("Hello world! This is a test. Hello again, world.")
        
        try:
            # 处理文件
            processor = TextProcessor()
            graph = processor.process_file(f.name)
            
            # 验证结果
            assert processor.total_words > 0
            assert processor.word_counts["hello"] == 2
            assert processor.word_counts["world"] == 2
            assert processor.word_counts["test"] == 1
            
            # 验证图结构
            assert "hello" in graph.nodes
            assert "world" in graph.nodes
            assert "test" in graph.nodes
            assert graph.get_weight("hello", "world") > 0
            assert graph.get_weight("this", "is") > 0
            
        finally:
            # 清理临时文件
            os.unlink(f.name)

class TestGraphAnalyzer:
    """测试GraphAnalyzer类的所有方法"""
    
    @pytest.fixture
    def setup_complex_graph(self):
        """创建一个复杂的测试图"""
        graph = DirectedGraph()
        
        # 创建一个有向图，包含多种路径和环
        # a -> b -> c -> d -> e
        # |    |    ^    |    ^
        # v    v    |    v    |
        # f -> g -> h -> i -> j
        
        edges = [
            ("a", "b", 1), ("a", "f", 2),
            ("b", "c", 3), ("b", "g", 1),
            ("c", "d", 2), ("d", "e", 1), ("d", "i", 3),
            ("f", "g", 2), ("g", "h", 1),
            ("h", "c", 3), ("h", "i", 2),
            ("i", "j", 1), ("j", "e", 2)
        ]
        
        for src, dst, weight in edges:
            graph.add_edge(src, dst, weight)
        
        return GraphAnalyzer(graph)
    
    def test_show_directed_graph(self, setup_complex_graph):
        """测试展示有向图"""
        analyzer = setup_complex_graph
        result = analyzer.show_directed_graph()
        
        # 验证结果包含预期的信息
        assert "有向图结构" in result
        assert "节点总数" in result
        assert "边总数" in result
        assert "节点 'a' 的出边" in result
        
        # 测试空图的情况
        empty_analyzer = GraphAnalyzer(DirectedGraph())
        result = empty_analyzer.show_directed_graph()
        assert "图是空的" in result
    
    def test_query_bridge_words(self, setup_complex_graph):
        """测试查询桥接词"""
        analyzer = setup_complex_graph
        
        # 测试存在单个桥接词的情况
        result = analyzer.query_bridge_words("a", "c")
        assert "bridge word" in result.lower()
        assert "b" in result
        
        # 测试存在多个桥接词的情况
        result = analyzer.query_bridge_words("b", "i")
        assert "bridge words" in result.lower() or "bridge word" in result.lower()
        
        # 测试不存在桥接词的情况
        result = analyzer.query_bridge_words("a", "e")
        assert "no bridge words" in result.lower()
        
        # 测试单词不在图中的情况
        result = analyzer.query_bridge_words("x", "y")
        assert "no x and y in the graph" in result.lower()
        
        result = analyzer.query_bridge_words("x", "a")
        assert "no x in the graph" in result.lower()
        
        result = analyzer.query_bridge_words("a", "y")
        assert "no y in the graph" in result.lower()
    
    def test_generate_new_text(self, setup_complex_graph):
        """测试生成新文本"""
        analyzer = setup_complex_graph
        
        # 测试正常情况
        with mock.patch('random.choice', side_effect=lambda x: x[0]):  # 总是选择第一个桥接词
            result = analyzer.generate_new_text("a c d e")
            assert "a" in result
            assert "c" in result
            assert "d" in result
            assert "e" in result
            # 应该插入桥接词b
            assert "b" in result
        
        # 测试输入文本太短的情况
        result = analyzer.generate_new_text("a")
        assert result == "a"
        
        # 测试没有桥接词的情况
        result = analyzer.generate_new_text("a e")
        assert "a" in result
        assert "e" in result
    
    def test_calc_shortest_path(self, setup_complex_graph):
        """测试计算最短路径"""
        analyzer = setup_complex_graph
        
        # 测试存在路径的情况
        result = analyzer.calc_shortest_path("a", "e")
        assert isinstance(result, tuple)
        assert len(result) == 3
        path_str, path, weight = result
        assert "a" in path_str and "e" in path_str
        assert path[0] == "a" and path[-1] == "e"
        assert weight > 0
        
        # 测试不存在路径的情况
        # 添加一个孤立节点
        analyzer.graph.nodes.add("isolated")
        result = analyzer.calc_shortest_path("a", "isolated")
        assert isinstance(result, str)
        assert "no path from a to isolated" in result.lower()
        
        # 测试单词不在图中的情况
        result = analyzer.calc_shortest_path("x", "e")
        assert "no x in the graph" in result.lower()
        
        result = analyzer.calc_shortest_path("a", "y")
        assert "no y in the graph" in result.lower()
    
    def test_calc_all_shortest_paths(self, setup_complex_graph):
        """测试计算一个单词到所有其他单词的最短路径"""
        analyzer = setup_complex_graph
        
        # 测试正常情况
        result = analyzer.calc_all_shortest_paths("a")
        assert isinstance(result, tuple)
        result_str, paths_dict = result
        assert "从单词 'a' 到其他单词的最短路径" in result_str
        assert isinstance(paths_dict, dict)
        assert "b" in paths_dict
        assert "c" in paths_dict
        
        # 测试单词不在图中的情况
        result = analyzer.calc_all_shortest_paths("x")
        assert isinstance(result, str)
        assert "no x in the graph" in result.lower()
        
        # 测试无法到达任何节点的情况
        isolated_graph = DirectedGraph()
        isolated_graph.nodes.add("a")
        isolated_analyzer = GraphAnalyzer(isolated_graph)
        result = isolated_analyzer.calc_all_shortest_paths("a")
        assert isinstance(result, str)
        assert "无法到达任何其他单词" in result
    
    def test_calc_all_shortest_paths_between(self, setup_complex_graph):
        """测试计算两个单词之间的所有最短路径"""
        analyzer = setup_complex_graph
        
        # 测试存在多条最短路径的情况
        # 添加一条与现有路径等长的路径
        analyzer.graph.add_edge("a", "k", 1)
        analyzer.graph.add_edge("k", "c", 3)
        
        result = analyzer.calc_all_shortest_paths_between("a", "c")
        assert isinstance(result, tuple)
        result_str, paths_with_weights = result
        assert "发现" in result_str and "最短路径" in result_str
        assert len(paths_with_weights) >= 1
        
        # 测试不存在路径的情况
        # 添加一个孤立节点
        analyzer.graph.nodes.add("isolated")
        result = analyzer.calc_all_shortest_paths_between("a", "isolated")
        assert isinstance(result, str)
        assert "No path from a to isolated" in result or "No isolated in the graph" in result
        
        # 测试单词不在图中的情况
        result = analyzer.calc_all_shortest_paths_between("x", "c")
        assert "No x in the graph" in result
        
        result = analyzer.calc_all_shortest_paths_between("a", "y")
        assert "No y in the graph" in result
    
    def test_calc_tf_idf(self, setup_complex_graph):
        """测试计算TF-IDF值"""
        analyzer = setup_complex_graph
        
        # 创建一个简单的文本处理器，确保包含图中所有节点的词频
        processor = TextProcessor()
        processor.total_words = 100
        
        # 为图中的每个节点添加词频
        processor.word_counts = {}
        for node in analyzer.graph.nodes:
            processor.word_counts[node] = 5  # 给每个节点一个默认词频
        
        # 添加一些特定的词频
        processor.word_counts["a"] = 10
        processor.word_counts["b"] = 5
        processor.word_counts["c"] = 3
        processor.word_counts["d"] = 2
        
        # 计算TF-IDF值
        tf_idf_values = analyzer.calc_tf_idf(processor)
        
        # 验证结果
        assert isinstance(tf_idf_values, dict)
        assert "a" in tf_idf_values
        assert "b" in tf_idf_values
        assert tf_idf_values["a"] > 0
    
    def test_calc_page_rank(self, setup_complex_graph):
        """测试计算PageRank值"""
        analyzer = setup_complex_graph
        
        # 计算PageRank值
        pr_values = analyzer.calc_page_rank(max_iterations=10)
        
        # 验证结果
        assert isinstance(pr_values, dict)
        assert "a" in pr_values
        assert "b" in pr_values
        assert pr_values["a"] > 0
        
        # 测试使用TF-IDF初始化
        processor = TextProcessor()
        processor.total_words = 100
        
        # 为图中的每个节点添加词频
        processor.word_counts = {}
        for node in analyzer.graph.nodes:
            processor.word_counts[node] = 5  # 给每个节点一个默认词频
        
        # 添加一些特定的词频
        processor.word_counts["a"] = 10
        processor.word_counts["b"] = 5
        processor.word_counts["c"] = 3
        processor.word_counts["d"] = 2
        
        pr_values = analyzer.calc_page_rank(max_iterations=10, text_processor=processor, use_tf_idf=True)
        assert isinstance(pr_values, dict)
        assert "a" in pr_values
        
        # 测试空图
        empty_analyzer = GraphAnalyzer(DirectedGraph())
        pr_values = empty_analyzer.calc_page_rank()
        assert pr_values == {}
    
    def test_get_page_rank(self, setup_complex_graph):
        """测试获取单词的PageRank值"""
        analyzer = setup_complex_graph
        
        # 计算PageRank值
        analyzer.calc_page_rank(max_iterations=10)
        
        # 测试获取已计算的PageRank值
        pr_value = analyzer.get_page_rank("a")
        assert isinstance(pr_value, float)
        assert pr_value > 0
        
        # 测试单词不在图中的情况
        result = analyzer.get_page_rank("x")
        assert "no x in the graph" in result.lower()
    
    @mock.patch('random.choice')
    def test_random_walk(self, mock_choice, setup_complex_graph):
        """测试随机游走"""
        analyzer = setup_complex_graph
        
        # 模拟随机选择
        mock_choice.side_effect = ["a", "b", "c", "d", "e"]
        
        # 测试正常情况
        result = analyzer.random_walk()
        assert isinstance(result, str)
        assert "a" in result
        assert "b" in result
        
        # 重置mock
        mock_choice.reset_mock()
        
        # 测试空图
        empty_analyzer = GraphAnalyzer(DirectedGraph())
        result = empty_analyzer.random_walk()
        assert "图是空的" in result
        
        # 重置mock
        mock_choice.reset_mock()
        
        # 测试遇到重复边的情况
        mock_choice.side_effect = ["a", "b", "a", "b"]
        result = analyzer.random_walk()
        assert "a" in result
        assert "b" in result
        
        # 重置mock
        mock_choice.reset_mock()
        
        # 测试遇到没有出边的节点
        mock_choice.side_effect = ["a", "b", "c", "d", "e"]  # e没有出边
        result = analyzer.random_walk()
        assert "a" in result
        assert "e" in result 