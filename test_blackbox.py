#!/usr/bin/env python3
"""
黑盒测试模块
测试graph_analyzer.py中的主要功能
"""

import os
import pytest
from directed_graph import DirectedGraph
from text_processor import TextProcessor
from graph_analyzer import GraphAnalyzer

class TestBlackBox:
    """黑盒测试类"""
    
    @pytest.fixture
    def setup_graph(self):
        """创建测试用的图结构"""
        # 创建一个简单的有向图用于测试
        graph = DirectedGraph()
        
        # 添加一些边
        graph.add_edge("hello", "world", 1)
        graph.add_edge("world", "hello", 2)
        graph.add_edge("hello", "python", 3)
        graph.add_edge("python", "is", 1)
        graph.add_edge("is", "awesome", 2)
        graph.add_edge("world", "is", 1)
        graph.add_edge("is", "great", 3)
        graph.add_edge("great", "language", 1)
        
        return GraphAnalyzer(graph)
    
    @pytest.fixture
    def setup_empty_graph(self):
        """创建空图用于测试边界情况"""
        return GraphAnalyzer(DirectedGraph())
    
    def test_query_bridge_words(self, setup_graph):
        """测试查询桥接词功能"""
        analyzer = setup_graph
        
        # 测试用例1: 存在单个桥接词
        print("\n测试用例1: 存在单个桥接词")
        result = analyzer.query_bridge_words("hello", "is")
        print(f"输入: hello, is")
        print(f"输出: {result}")
        assert "python" in result
        assert "The bridge word" in result
        
        # 测试用例2: 存在多个桥接词
        print("\n测试用例2: 存在多个桥接词")
        # 在我们的测试图中没有多桥接词的情况，这里模拟一个
        analyzer.graph.add_edge("hello", "test", 1)
        analyzer.graph.add_edge("test", "is", 1)
        result = analyzer.query_bridge_words("hello", "is")
        print(f"输入: hello, is (添加test作为另一个桥接词后)")
        print(f"输出: {result}")
        assert "python" in result
        assert "test" in result
        assert "The bridge words" in result
        
        # 测试用例3: 不存在桥接词
        print("\n测试用例3: 不存在桥接词")
        result = analyzer.query_bridge_words("hello", "awesome")
        print(f"输入: hello, awesome")
        print(f"输出: {result}")
        assert "No bridge words" in result
        
        # 测试用例4: 第一个单词不在图中
        print("\n测试用例4: 第一个单词不在图中")
        result = analyzer.query_bridge_words("nonexistent", "world")
        print(f"输入: nonexistent, world")
        print(f"输出: {result}")
        assert "No nonexistent in the graph" in result
        
        # 测试用例5: 第二个单词不在图中
        print("\n测试用例5: 第二个单词不在图中")
        result = analyzer.query_bridge_words("hello", "nonexistent")
        print(f"输入: hello, nonexistent")
        print(f"输出: {result}")
        assert "No nonexistent in the graph" in result
        
        # 测试用例6: 两个单词都不在图中
        print("\n测试用例6: 两个单词都不在图中")
        result = analyzer.query_bridge_words("nonexistent1", "nonexistent2")
        print(f"输入: nonexistent1, nonexistent2")
        print(f"输出: {result}")
        assert "No nonexistent1 and nonexistent2 in the graph" in result
    
    def test_generate_new_text(self, setup_graph):
        """测试根据桥接词生成新文本功能"""
        analyzer = setup_graph
        
        # 测试用例1: 正常输入，有桥接词可插入
        result = analyzer.generate_new_text("hello is awesome")
        words = result.split()
        assert len(words) >= 3  # 至少应该包含原来的三个单词
        assert words[0] == "hello"
        assert "is" in words
        assert words[-1] == "awesome"
        
        # 测试用例2: 输入文本中单词不在图中
        result = analyzer.generate_new_text("nonexistent words here")
        assert result == "nonexistent words here"  # 应该保持原样
        
        # 测试用例3: 空输入
        result = analyzer.generate_new_text("")
        assert result == ""
        
        # 测试用例4: 只有一个单词
        result = analyzer.generate_new_text("hello")
        assert result == "hello"
    
    def test_calc_shortest_path(self, setup_graph):
        """测试计算最短路径功能"""
        analyzer = setup_graph
        
        # 测试用例1: 存在直接路径
        result = analyzer.calc_shortest_path("hello", "world")
        assert isinstance(result, tuple)
        assert "hello -> world" in result[0]
        
        # 测试用例2: 存在间接路径
        result = analyzer.calc_shortest_path("hello", "awesome")
        assert isinstance(result, tuple)
        assert "hello" in result[0]
        assert "awesome" in result[0]
        
        # 测试用例3: 不存在路径
        # 添加一个孤立的节点
        analyzer.graph.nodes.add("isolated")
        result = analyzer.calc_shortest_path("hello", "isolated")
        assert "No path" in result
        
        # 测试用例4: 第一个单词不在图中
        result = analyzer.calc_shortest_path("nonexistent", "world")
        assert "No nonexistent in the graph" in result
        
        # 测试用例5: 第二个单词不在图中
        result = analyzer.calc_shortest_path("hello", "nonexistent")
        assert "No nonexistent in the graph" in result
    
    def test_random_walk(self, setup_graph, setup_empty_graph):
        """测试随机游走功能"""
        analyzer = setup_graph
        
        # 测试用例1: 正常随机游走
        result = analyzer.random_walk()
        assert isinstance(result, str)
        assert len(result) > 0
        
        # 测试用例2: 空图随机游走
        empty_analyzer = setup_empty_graph
        result = empty_analyzer.random_walk()
        assert "图是空的" in result or "The graph is empty" in result
    
    def test_empty_graph(self, setup_empty_graph):
        """测试空图的情况"""
        analyzer = setup_empty_graph
        
        # 测试查询桥接词
        result = analyzer.query_bridge_words("word1", "word2")
        assert "No word1 and word2 in the graph" in result
        
        # 测试生成新文本
        result = analyzer.generate_new_text("hello world")
        assert result == "hello world"  # 应该保持原样
        
        # 测试计算最短路径
        result = analyzer.calc_shortest_path("word1", "word2")
        assert "No word1 in the graph" in result 