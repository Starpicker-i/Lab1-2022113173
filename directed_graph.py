from collections import defaultdict

class DirectedGraph:
    def __init__(self):
        # 存储节点和边的信息
        self.nodes = set()  # 存储所有单词节点
        self.edges = defaultdict(dict)  # 存储有向边及其权重
        self.pr_values = {}  # 存储PageRank值
    
    def add_edge(self, source, target, weight=1):
        """添加或更新有向边"""
        if source not in self.edges or target not in self.edges[source]:
            self.edges[source][target] = weight
        else:
            self.edges[source][target] += weight
        
        # 更新节点集合
        self.nodes.add(source)
        self.nodes.add(target)
    
    def get_weight(self, source, target):
        """获取边的权重"""
        if source in self.edges and target in self.edges[source]:
            return self.edges[source][target]
        return 0
    
    def get_successors(self, node):
        """获取节点的所有后继节点"""
        if node in self.edges:
            return self.edges[node].keys()
        return []
    
    def get_predecessors(self, node):
        """获取节点的所有前驱节点"""
        preds = []
        for source in self.edges:
            if node in self.edges[source]:
                preds.append(source)
        return preds 