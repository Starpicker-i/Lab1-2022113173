import heapq
import random
import math
import numpy as np
from collections import deque

class GraphAnalyzer:
    def __init__(self, graph):
        self.graph = graph
        self.tf_idf_values = {}  # 存储单词的TF-IDF值
    
    def show_directed_graph(self):
        """展示有向图"""
        if not self.graph.nodes:
            return "图是空的"
        
        result = "有向图结构:\n"
        result += "节点总数: {}\n".format(len(self.graph.nodes))
        result += "边总数: {}\n".format(sum(len(targets) for targets in self.graph.edges.values()))
        
        for source in sorted(self.graph.edges.keys()):
            result += "\n节点 '{}' 的出边:\n".format(source)
            for target, weight in sorted(self.graph.edges[source].items()):
                result += "  -> '{}' (权重: {})\n".format(target, weight)
        
        return result
    
    def query_bridge_words(self, word1, word2):
        """查询桥接词"""
        word1 = word1.lower()
        word2 = word2.lower()
        
        # 检查单词是否在图中
        if word1 not in self.graph.nodes and word2 not in self.graph.nodes:
            return "No {} and {} in the graph!".format(word1, word2)
        elif word1 not in self.graph.nodes:
            return "No {} in the graph!".format(word1)
        elif word2 not in self.graph.nodes:
            return "No {} in the graph!".format(word2)
        
        # 查找桥接词
        bridge_words = []
        for succ in self.graph.get_successors(word1):
            if word2 in self.graph.get_successors(succ):
                bridge_words.append(succ)
        
        # 返回结果
        if not bridge_words:
            return "No bridge words from {} to {}!".format(word1, word2)
        elif len(bridge_words) == 1:
            return "The bridge word from {} to {} is: {}".format(word1, word2, bridge_words[0])
        else:
            bridges = ", ".join(bridge_words[:-1]) + ", and " + bridge_words[-1]
            return "The bridge words from {} to {} are: {}".format(word1, word2, bridges)
    
    def generate_new_text(self, input_text):
        """根据bridge word生成新文本"""
        import re
        
        # 处理输入文本
        input_text = input_text.lower()
        input_text = re.sub(r'[^\w\s]', ' ', input_text)
        words = re.findall(r'\b[a-z]+\b', input_text)
        
        if len(words) < 2:
            return input_text
        
        # 插入桥接词
        result = [words[0]]
        for i in range(len(words) - 1):
            word1, word2 = words[i], words[i+1]
            
            # 检查是否有桥接词
            bridge_words = []
            if word1 in self.graph.nodes and word2 in self.graph.nodes:
                for succ in self.graph.get_successors(word1):
                    if word2 in self.graph.get_successors(succ):
                        bridge_words.append(succ)
            
            # 随机选择一个桥接词并插入
            if bridge_words:
                bridge = random.choice(bridge_words)
                result.append(bridge)
            
            result.append(word2)
        
        return " ".join(result)
    
    def calc_shortest_path(self, word1, word2):
        """计算两个单词之间的最短路径"""
        word1 = word1.lower()
        word2 = word2.lower()
        
        # 检查单词是否在图中
        if word1 not in self.graph.nodes:
            return "No {} in the graph!".format(word1)
        if word2 not in self.graph.nodes:
            return "No {} in the graph!".format(word2)
        
        # 使用Dijkstra算法找最短路径
        distances = {node: float('infinity') for node in self.graph.nodes}
        distances[word1] = 0
        previous = {node: None for node in self.graph.nodes}
        priority_queue = [(0, word1)]
        visited = set()
        
        while priority_queue:
            current_distance, current_node = heapq.heappop(priority_queue)
            
            if current_node == word2:
                break
            
            if current_node in visited:
                continue
            
            visited.add(current_node)
            
            for neighbor in self.graph.get_successors(current_node):
                weight = self.graph.get_weight(current_node, neighbor)
                distance = current_distance + 1/weight  # 权重越大，距离越短
                
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    previous[neighbor] = current_node
                    heapq.heappush(priority_queue, (distance, neighbor))
        
        # 重建路径
        if distances[word2] == float('infinity'):
            return "No path from {} to {}!".format(word1, word2)
        
        path = []
        current = word2
        while current:
            path.append(current)
            current = previous[current]
        
        path.reverse()
        
        # 计算路径总权重
        path_weight = 0
        for i in range(len(path) - 1):
            source, target = path[i], path[i + 1]
            path_weight += self.graph.get_weight(source, target)
        
        path_str = " -> ".join(path)
        return "Shortest path from {} to {}: {} (Total weight: {})".format(word1, word2, path_str, path_weight), path, path_weight
    
    def calc_all_shortest_paths(self, word):
        """计算一个单词到所有其他单词的最短路径"""
        word = word.lower()
        
        # 检查单词是否在图中
        if word not in self.graph.nodes:
            return "No {} in the graph!".format(word)
        
        # 使用Dijkstra算法找到所有最短路径
        distances = {node: float('infinity') for node in self.graph.nodes}
        distances[word] = 0
        previous = {node: None for node in self.graph.nodes}
        priority_queue = [(0, word)]
        visited = set()
        
        while priority_queue:
            current_distance, current_node = heapq.heappop(priority_queue)
            
            if current_node in visited:
                continue
            
            visited.add(current_node)
            
            for neighbor in self.graph.get_successors(current_node):
                weight = self.graph.get_weight(current_node, neighbor)
                distance = current_distance + 1/weight  # 权重越大，距离越短
                
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    previous[neighbor] = current_node
                    heapq.heappush(priority_queue, (distance, neighbor))
        
        # 构建结果
        result = "从单词 '{}' 到其他单词的最短路径:\n".format(word)
        reachable_nodes = sorted([n for n in self.graph.nodes if distances[n] < float('infinity') and n != word])
        
        if not reachable_nodes:
            return "单词 '{}' 无法到达任何其他单词!".format(word)
        
        # 为可能的可视化功能创建路径字典
        paths_dict = {}
        
        for target in reachable_nodes:
            # 重建路径
            path = []
            current = target
            while current:
                path.append(current)
                current = previous[current]
            
            path.reverse()
            
            # 计算路径总权重
            path_weight = 0
            for i in range(len(path) - 1):
                source, target_node = path[i], path[i + 1]
                path_weight += self.graph.get_weight(source, target_node)
            
            path_str = " -> ".join(path)
            result += "  到 '{}': {} (Total weight: {})\n".format(target, path_str, path_weight)
            
            # 存储路径信息
            paths_dict[target] = (path, path_weight)
        
        return result, paths_dict
    
    def calc_all_shortest_paths_between(self, word1, word2):
        """计算两个单词之间的所有最短路径
        
        Args:
            word1: 起始单词
            word2: 目标单词
            
        Returns:
            如果找到路径，返回 (结果字符串, 路径列表, 总权重)
            如果没有找到路径，返回错误信息
        """
        word1 = word1.lower()
        word2 = word2.lower()
        
        # 检查单词是否在图中
        if word1 not in self.graph.nodes:
            return "No {} in the graph!".format(word1)
        if word2 not in self.graph.nodes:
            return "No {} in the graph!".format(word2)
        
        # 使用BFS找到所有最短路径
        queue = deque([(word1, [word1])])  # (当前节点, 当前路径)
        visited = {word1}  # 记录已访问节点
        all_paths = []  # 存储所有最短路径
        min_distance = float('infinity')  # 最短路径的距离
        
        while queue:
            node, path = queue.popleft()
            
            # 如果已经找到了更短的路径，跳过长路径
            if len(path) > min_distance:
                continue
                
            if node == word2:
                # 找到一条路径
                if len(path) < min_distance:
                    # 如果这条路径比之前找到的短，清空之前的路径
                    min_distance = len(path)
                    all_paths = [path]
                elif len(path) == min_distance:
                    # 如果这条路径和之前找到的一样长，添加到结果中
                    all_paths.append(path)
                continue
            
            # 寻找下一个节点
            for neighbor in self.graph.get_successors(node):
                if len(path) + 1 <= min_distance:  # 只考虑不超过当前最短距离的路径
                    new_path = path + [neighbor]
                    queue.append((neighbor, new_path))
        
        if not all_paths:
            return "No path from {} to {}!".format(word1, word2)
        
        # 计算每条路径的总权重
        paths_with_weights = []
        for path in all_paths:
            path_weight = 0
            for i in range(len(path) - 1):
                source, target = path[i], path[i + 1]
                path_weight += self.graph.get_weight(source, target)
            paths_with_weights.append((path, path_weight))
        
        # 按权重排序
        paths_with_weights.sort(key=lambda x: x[1])
        
        # 构建结果字符串
        result = f"发现 {len(all_paths)} 条从 {word1} 到 {word2} 的最短路径:\n"
        for i, (path, weight) in enumerate(paths_with_weights):
            path_str = " -> ".join(path)
            result += f"路径 {i+1}: {path_str} (权重: {weight})\n"
        
        return result, paths_with_weights
    
    def calc_tf_idf(self, text_processor):
        """计算每个单词的TF-IDF值"""
        # 计算每个单词的词频 (TF)
        word_tf = {}
        for word in self.graph.nodes:
            if text_processor.total_words > 0:
                word_tf[word] = text_processor.word_counts[word] / text_processor.total_words
            else:
                word_tf[word] = 0
        
        # 计算每个单词的逆文档频率 (IDF)
        # 在单个文档的情况下，我们使用节点的度数作为替代指标
        word_idf = {}
        for word in self.graph.nodes:
            # 计算入度和出度
            in_degree = len(list(self.graph.get_predecessors(word)))
            out_degree = len(list(self.graph.get_successors(word)))
            # 节点的总度数
            total_degree = in_degree + out_degree
            # 使用度数的对数加1作为IDF值（防止0值）
            word_idf[word] = math.log(1 + total_degree)
        
        # 计算TF-IDF值
        for word in self.graph.nodes:
            self.tf_idf_values[word] = word_tf[word] * word_idf[word]
        
        return self.tf_idf_values
    
    def calc_page_rank(self, d=0.85, max_iterations=100, tolerance=1e-6, text_processor=None, use_tf_idf=True):
        """计算所有节点的PageRank值，可以选择使用TF-IDF作为初始值
        
        Args:
            d: 阻尼系数
            max_iterations: 最大迭代次数
            tolerance: 收敛容差
            text_processor: TextProcessor实例，用于计算TF-IDF
            use_tf_idf: 是否使用TF-IDF值初始化PR
        """
        n = len(self.graph.nodes)
        if n == 0:
            return {}
        
        # 将节点转换为索引
        nodes_list = list(self.graph.nodes)
        node_to_idx = {node: i for i, node in enumerate(nodes_list)}
        
        # 初始化PR值
        if use_tf_idf and text_processor:
            # 计算TF-IDF值
            if not self.tf_idf_values:
                self.calc_tf_idf(text_processor)
            
            # 使用TF-IDF值初始化PR值
            tf_idf_sum = sum(self.tf_idf_values.values())
            if tf_idf_sum > 0:
                pr = np.array([self.tf_idf_values.get(node, 0) / tf_idf_sum for node in nodes_list])
            else:
                pr = np.ones(n) / n
        else:
            # 使用均匀分布初始化
            pr = np.ones(n) / n
        
        # 构建转移矩阵
        M = np.zeros((n, n))
        # 记录出度为0的节点
        dangling_nodes = []
        
        for i, node in enumerate(nodes_list):
            successors = list(self.graph.get_successors(node))
            if successors:
                total_weight = sum(self.graph.get_weight(node, succ) for succ in successors)
                for succ in successors:
                    j = node_to_idx[succ]
                    weight = self.graph.get_weight(node, succ)
                    M[j, i] = weight / total_weight
            else:
                # 记录出度为0的节点
                dangling_nodes.append(i)
        
        # 迭代计算PR值
        for _ in range(max_iterations):
            # 创建出度为0的节点的PR值均分矩阵
            dangling_pr = np.zeros(n)
            for i in dangling_nodes:
                # 将出度为0的节点的PR值均分给所有节点
                dangling_pr += pr[i] / n
            
            # 计算新的PR值，包括出度为0节点的贡献
            pr_next = (1 - d) / n + d * (M.dot(pr) + dangling_pr)
            
            # 检查收敛性
            if np.linalg.norm(pr_next - pr) < tolerance:
                break
            
            pr = pr_next
        
        # 存储结果
        self.graph.pr_values = {node: pr[i] for i, node in enumerate(nodes_list)}
        return self.graph.pr_values
    
    def get_page_rank(self, word):
        """获取单词的PageRank值"""
        word = word.lower()
        if word not in self.graph.nodes:
            return "No {} in the graph!".format(word)
        
        # 如果PR值尚未计算，则计算
        if not self.graph.pr_values:
            self.calc_page_rank()
        
        return self.graph.pr_values.get(word, 0)
    
    def random_walk(self):
        """随机游走"""
        if not self.graph.nodes:
            return "图是空的，无法进行随机游走"
        
        # 随机选择起始节点
        current = random.choice(list(self.graph.nodes))
        path = [current]
        edges_visited = set()
        
        while True:
            # 获取当前节点的所有后继节点
            successors = list(self.graph.get_successors(current))
            if not successors:
                break  # 如果没有出边，停止游走
            
            # 随机选择下一个节点
            next_node = random.choice(successors)
            edge = (current, next_node)
            
            # 检查边是否重复访问
            if edge in edges_visited:
                break
            
            edges_visited.add(edge)
            path.append(next_node)
            current = next_node
        
        return " ".join(path) 