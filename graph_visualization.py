import matplotlib.pyplot as plt
import networkx as nx
import os
from directed_graph import DirectedGraph
from text_processor import TextProcessor
from graph_analyzer import GraphAnalyzer
import numpy as np

class GraphVisualizer:
    def __init__(self, graph):
        self.graph = graph
        self.nx_graph = None
        self._create_networkx_graph()
        
    def _create_networkx_graph(self):
        """将DirectedGraph对象转换为NetworkX图形对象"""
        self.nx_graph = nx.DiGraph()
        
        # 添加所有节点
        for node in self.graph.nodes:
            self.nx_graph.add_node(node)
        
        # 添加所有边及其权重
        for source in self.graph.edges:
            for target, weight in self.graph.edges[source].items():
                self.nx_graph.add_edge(source, target, weight=weight)
    
    def visualize_graph(self, output_file=None, show=True):
        """可视化有向图
        
        Args:
            output_file: 输出文件的路径（可选）
            show: 是否显示图形
        """
        if not self.graph.nodes:
            return "图是空的，无法可视化"
        
        # 使用spring布局算法计算节点位置
        pos = nx.spring_layout(self.nx_graph, seed=42, k=1.5)  # k值控制节点间距，默认为1.0
        
        plt.figure(figsize=(14, 12))  # 也适当增大图形尺寸
        
        # 绘制节点
        nx.draw_networkx_nodes(self.nx_graph, pos, 
                               node_size=800,  # 调大节点尺寸
                               node_color='lightblue', 
                               alpha=0.8)
        
        # 绘制边
        edges = self.nx_graph.edges()
        weights = [self.nx_graph[u][v]['weight'] for u, v in edges]
        
        # 归一化边权重用于绘图
        if weights:
            max_weight = max(weights)
            normalized_weights = [w / max_weight * 3 for w in weights]
        else:
            normalized_weights = []
        
        nx.draw_networkx_edges(self.nx_graph, pos, 
                               width=normalized_weights,
                               alpha=0.5, 
                               edge_color='black',  # 改为黑色箭头
                               arrows=True, 
                               arrowsize=15, 
                               arrowstyle='->')
        
        # 绘制节点标签
        nx.draw_networkx_labels(self.nx_graph, pos, 
                                font_size=10, 
                                font_family='sans-serif')
        
        # 绘制边权重标签
        edge_labels = {(u, v): f'{d["weight"]:.0f}' for u, v, d in self.nx_graph.edges(data=True)}
        nx.draw_networkx_edge_labels(self.nx_graph, pos, 
                                     edge_labels=edge_labels, 
                                     font_size=8)
        
        plt.title("Directed Graph Visualization")
        plt.axis('off')  # 隐藏坐标轴
        
        if output_file:
            plt.savefig(output_file, format='png', dpi=300, bbox_inches='tight')
            print(f"图形已保存到: {output_file}")
        
        if show:
            plt.show()
            
        plt.close()
        
        return "图形可视化完成"
    
    def visualize_path(self, path, output_file=None, show=True, path_length=None):
        """可视化路径
        
        Args:
            path: 节点列表表示的路径
            output_file: 输出文件的路径（可选）
            show: 是否显示图形
            path_length: 路径长度（如果提供）
        """
        if not self.graph.nodes:
            return "图是空的，无法可视化"
        
        if not path or len(path) < 2:
            return "路径太短或为空，无法可视化"
        
        # 创建路径边列表
        path_edges = list(zip(path[:-1], path[1:]))
        
        # 计算路径总长度（如果未提供）
        if path_length is None:
            # 计算所有边权值之和的倒数（因为我们在计算最短路径时使用了1/weight）
            path_length = 0
            for source, target in path_edges:
                weight = self.graph.get_weight(source, target)
                path_length += weight
        
        # 使用spring布局算法计算节点位置，增大k值来增加节点间距
        pos = nx.spring_layout(self.nx_graph, seed=42, k=1.5)
        
        plt.figure(figsize=(14, 12))  # 也适当增大图形尺寸
        
        # 绘制所有节点
        nx.draw_networkx_nodes(self.nx_graph, pos, 
                               node_size=800, 
                               node_color='lightgray', 
                               alpha=0.6)
        
        # 高亮路径上的节点
        path_nodes = set(path)
        nx.draw_networkx_nodes(self.nx_graph, pos, 
                               nodelist=path_nodes, 
                               node_size=800, 
                               node_color='lightblue', 
                               alpha=0.9)
        
        # 绘制所有边
        nx.draw_networkx_edges(self.nx_graph, pos, 
                               alpha=0.2, 
                               edge_color='black',
                               arrows=True, 
                               arrowsize=10)
        
        # 高亮路径上的边
        nx.draw_networkx_edges(self.nx_graph, pos, 
                               edgelist=path_edges, 
                               width=2, 
                               alpha=1, 
                               edge_color='red',
                               arrows=True, 
                               arrowsize=15)
        
        # 绘制节点标签
        nx.draw_networkx_labels(self.nx_graph, pos, 
                                font_size=10, 
                                font_family='sans-serif')
        
        # 绘制路径边权重标签
        path_edge_labels = {(u, v): f'{self.graph.get_weight(u, v)}' 
                          for u, v in path_edges}
        nx.draw_networkx_edge_labels(self.nx_graph, pos, 
                                     edge_labels=path_edge_labels, 
                                     font_size=8)
        
        # 添加路径长度标题
        plt.title(f"Shortest Path - Total Length: {path_length}")
        plt.axis('off')
        
        if output_file:
            plt.savefig(output_file, format='png', dpi=300, bbox_inches='tight')
            print(f"路径图形已保存到: {output_file}")
        
        if show:
            plt.show()
            
        plt.close()
        
        return "路径可视化完成"
    
    def visualize_random_walk(self, walk_path, output_file=None, show=True):
        """可视化随机游走路径
        
        Args:
            walk_path: 随机游走的路径字符串或节点列表
            output_file: 输出文件的路径（可选）
            show: 是否显示图形
        """
        if not self.graph.nodes:
            return "图是空的，无法可视化"
        
        # 如果输入是字符串，将其分割为节点列表
        if isinstance(walk_path, str):
            path = walk_path.split()
        else:
            path = walk_path
            
        if not path or len(path) < 2:
            return "路径太短或为空，无法可视化"
        
        # 创建路径边列表
        path_edges = list(zip(path[:-1], path[1:]))
        
        # 使用spring布局算法计算节点位置，增大k值来增加节点间距
        pos = nx.spring_layout(self.nx_graph, seed=42, k=1.5)
        
        plt.figure(figsize=(14, 12))  # 也适当增大图形尺寸
        
        # 绘制所有节点
        nx.draw_networkx_nodes(self.nx_graph, pos, 
                               node_size=800,  # 调大节点尺寸
                               node_color='lightgray', 
                               alpha=0.6)
        
        # 高亮路径上的节点，颜色从浅到深表示访问顺序
        node_colors = plt.cm.cool(np.linspace(0, 1, len(path)))
        for i, node in enumerate(path):
            nx.draw_networkx_nodes(self.nx_graph, pos, 
                                  nodelist=[node], 
                                  node_size=800,  # 调大节点尺寸
                                  node_color=[node_colors[i]], 
                                  alpha=0.9)
        
        # 绘制所有边
        nx.draw_networkx_edges(self.nx_graph, pos, 
                               alpha=0.2, 
                               edge_color='black',  # 改为黑色箭头
                               arrows=True, 
                               arrowsize=10)
        
        # 高亮路径上的边，颜色表示访问顺序
        edge_colors = plt.cm.cool(np.linspace(0, 1, len(path_edges)))
        for i, edge in enumerate(path_edges):
            nx.draw_networkx_edges(self.nx_graph, pos, 
                                  edgelist=[edge], 
                                  width=2, 
                                  alpha=1, 
                                  edge_color=[edge_colors[i]],
                                  arrows=True, 
                                  arrowsize=15)
        
        # 绘制节点标签
        nx.draw_networkx_labels(self.nx_graph, pos, 
                                font_size=10, 
                                font_family='sans-serif')
        
        # 标记路径中的首尾节点
        first_node = path[0]
        last_node = path[-1]
        nx.draw_networkx_nodes(self.nx_graph, pos, 
                              nodelist=[first_node], 
                              node_size=900,  # 调大节点尺寸
                              node_color='green', 
                              alpha=0.9)
        nx.draw_networkx_nodes(self.nx_graph, pos, 
                              nodelist=[last_node], 
                              node_size=900,  # 调大节点尺寸
                              node_color='red', 
                              alpha=0.9)
        
        plt.title("Random Walk Visualization")
        plt.axis('off')
        
        # 添加图例
        plt.legend([plt.Line2D([0], [0], color='green', marker='o', linestyle=''),
                   plt.Line2D([0], [0], color='red', marker='o', linestyle='')],
                  ['起始节点', '结束节点'],
                  numpoints=1, loc='best')
        
        if output_file:
            plt.savefig(output_file, format='png', dpi=300, bbox_inches='tight')
            print(f"随机游走图形已保存到: {output_file}")
        
        if show:
            plt.show()
            
        plt.close()
        
        return "随机游走可视化完成"
    
    def visualize_all_paths(self, paths, output_file=None, show=True, title=None):
        """可视化多条路径，使用不同颜色区分每条路径
        
        Args:
            paths: 路径列表，每个路径是节点列表
            output_file: 输出文件的路径（可选）
            show: 是否显示图形
            title: 图形标题（可选）
        """
        if not self.graph.nodes:
            return "图是空的，无法可视化"
        
        if not paths or not paths[0]:
            return "路径为空，无法可视化"
        
        # 获取路径中的所有节点和边
        path_nodes = set()
        path_edges_list = []
        
        for path in paths:
            path_nodes.update(path)
            path_edges = list(zip(path[:-1], path[1:]))
            path_edges_list.append(path_edges)
        
        # 使用spring布局算法计算节点位置
        pos = nx.spring_layout(self.nx_graph, seed=42, k=1.5)
        
        plt.figure(figsize=(14, 12))
        
        # 绘制所有节点（浅灰色）
        nx.draw_networkx_nodes(self.nx_graph, pos, 
                              node_size=800, 
                              node_color='lightgray', 
                              alpha=0.6)
        
        # 高亮路径中的所有节点（浅蓝色）
        nx.draw_networkx_nodes(self.nx_graph, pos, 
                              nodelist=list(path_nodes), 
                              node_size=800, 
                              node_color='lightblue', 
                              alpha=0.8)
        
        # 绘制所有边（浅灰色）
        nx.draw_networkx_edges(self.nx_graph, pos, 
                              alpha=0.2, 
                              edge_color='black',
                              arrows=True, 
                              arrowsize=10)
        
        # 使用不同颜色绘制每条路径的边
        colors = plt.cm.tab10.colors  # 使用tab10颜色盘，提供10种不同颜色
        
        # 创建图例信息
        legend_lines = []
        legend_labels = []
        
        for i, path_edges in enumerate(path_edges_list):
            # 使用循环颜色，确保即使路径数超过10也能分配颜色
            color = colors[i % len(colors)]
            
            # 绘制该路径的边
            nx.draw_networkx_edges(self.nx_graph, pos, 
                                  edgelist=path_edges, 
                                  width=2, 
                                  alpha=0.8, 
                                  edge_color=color,
                                  arrows=True, 
                                  arrowsize=15)
            
            # 为该路径在图例中添加一行
            line = plt.Line2D([0], [0], color=color, linewidth=2)
            legend_lines.append(line)
            legend_labels.append(f"路径 {i+1}")
        
        # 绘制节点标签
        nx.draw_networkx_labels(self.nx_graph, pos, 
                              font_size=10, 
                              font_family='sans-serif')
        
        # 设置标题
        if title:
            plt.title(title)
        else:
            plt.title("多路径可视化")
        
        plt.axis('off')
        
        # 添加图例
        plt.legend(legend_lines, legend_labels, loc='best')
        
        if output_file:
            plt.savefig(output_file, format='png', dpi=300, bbox_inches='tight')
            print(f"多路径图形已保存到: {output_file}")
        
        if show:
            plt.show()
            
        plt.close()
        
        return "多路径可视化完成"
    
    def visualize_pagerank(self, output_file=None, show=True, top_n=None):
        """可视化PageRank值
        
        Args:
            output_file: 输出文件的路径（可选）
            show: 是否显示图形
            top_n: 只显示前N个PageRank值最高的节点（可选）
        """
        if not self.graph.nodes:
            return "图是空的，无法可视化"
        
        # 确保已计算PageRank值
        if not self.graph.pr_values:
            analyzer = GraphAnalyzer(self.graph)
            analyzer.calc_page_rank()
        
        # 如果指定了top_n，只保留前N个节点
        if top_n and top_n < len(self.graph.nodes):
            # 按PageRank值排序
            sorted_pr = sorted(self.graph.pr_values.items(), key=lambda x: x[1], reverse=True)
            top_nodes = [node for node, _ in sorted_pr[:top_n]]
            
            # 创建子图
            subgraph = nx.DiGraph()
            for node in top_nodes:
                subgraph.add_node(node)
                
            # 添加这些节点之间的边
            for source in top_nodes:
                for target in self.graph.get_successors(source):
                    if target in top_nodes:
                        weight = self.graph.get_weight(source, target)
                        subgraph.add_edge(source, target, weight=weight)
            
            # 使用子图替代完整图进行可视化
            graph_to_visualize = subgraph
        else:
            # 使用完整图
            graph_to_visualize = self.nx_graph
            
        # 使用spring布局算法计算节点位置，增大k值来增加节点间距
        pos = nx.spring_layout(graph_to_visualize, seed=42, k=1.5)
        
        plt.figure(figsize=(14, 12))  # 也适当增大图形尺寸
        
        # 获取节点的PageRank值作为节点大小
        node_pr = {node: self.graph.pr_values.get(node, 0) for node in graph_to_visualize.nodes()}
        
        # 将PageRank值归一化为合适的节点大小
        pr_values = list(node_pr.values())
        if pr_values:
            max_pr = max(pr_values)
            min_pr = min(pr_values)
            node_sizes = [1500 * (pr - min_pr) / (max_pr - min_pr) + 500 if max_pr > min_pr else 800 for pr in pr_values]
        else:
            node_sizes = [800] * len(graph_to_visualize.nodes())
        
        # 根据PageRank值确定节点颜色
        node_colors = [plt.cm.viridis(pr / max(pr_values)) if pr_values else plt.cm.viridis(0) for pr in pr_values]
        
        # 绘制节点
        nx.draw_networkx_nodes(graph_to_visualize, pos, 
                              node_size=node_sizes, 
                              node_color=node_colors, 
                              alpha=0.8)
        
        # 绘制边
        edges = graph_to_visualize.edges()
        weights = [graph_to_visualize[u][v]['weight'] for u, v in edges]
        
        # 归一化边权重用于绘图
        if weights:
            max_weight = max(weights)
            normalized_weights = [w / max_weight * 3 for w in weights]
        else:
            normalized_weights = []
        
        nx.draw_networkx_edges(graph_to_visualize, pos, 
                              width=normalized_weights,
                              alpha=0.5, 
                              edge_color='black',  # 改为黑色箭头
                              arrows=True, 
                              arrowsize=15, 
                              arrowstyle='->')
        
        # 绘制节点标签
        nx.draw_networkx_labels(graph_to_visualize, pos, 
                               font_size=10, 
                               font_family='sans-serif')
        
        # 添加PageRank值标签
        pr_labels = {node: f'{pr:.4f}' for node, pr in node_pr.items()}
        nx.draw_networkx_labels(graph_to_visualize, pos, 
                               labels=pr_labels,
                               font_size=8,
                               font_color='red',
                               font_family='sans-serif',
                               verticalalignment='bottom')
        
        plt.title("PageRank Visualization - Node Size Represents Importance")
        plt.axis('off')
        
        # 添加颜色条
        sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=min(pr_values) if pr_values else 0, 
                                                                          vmax=max(pr_values) if pr_values else 1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=plt.gca())
        cbar.set_label('PageRank Value')
        
        if output_file:
            plt.savefig(output_file, format='png', dpi=300, bbox_inches='tight')
            print(f"PageRank可视化已保存到: {output_file}")
        
        if show:
            plt.show()
            
        plt.close()
        
        return "PageRank可视化完成"


def main():
    # 获取文件路径
    while True:
        file_path = input("请输入文本文件路径（或直接按回车使用默认文件'Easy Test.txt'）: ")
        if not file_path:
            file_path = "Easy Test.txt"
        
        if os.path.exists(file_path):
            break
        else:
            print("文件不存在，请重新输入！")
    
    # 处理文件并构建图
    processor = TextProcessor()
    graph = processor.process_file(file_path)
    analyzer = GraphAnalyzer(graph)
    visualizer = GraphVisualizer(graph)
    
    # 展示功能菜单
    while True:
        print("\n图形可视化功能：")
        print("1. 可视化有向图并保存")
        print("2. 可视化两个单词间的最短路径")
        print("3. 可视化随机游走路径")
        print("4. 可视化PageRank值")
        print("5. 可视化多条路径")
        print("0. 返回主程序")
        
        choice = input("请输入选项编号: ")
        
        if choice == "1":
            output_file = input("请输入保存文件名 (按回车使用默认名称 'graph.png'): ")
            if not output_file:
                output_file = "graph.png"
            visualizer.visualize_graph(output_file=output_file)
        
        elif choice == "2":
            word1 = input("请输入起始单词: ")
            word2 = input("请输入目标单词: ")
            
            # 获取最短路径
            path_result = analyzer.calc_shortest_path(word1, word2)
            print(path_result)
            
            # 如果找到了路径，可视化它
            if "Shortest path" in path_result:
                path_str = path_result.split(": ")[1]
                path = path_str.split(" -> ")
                
                output_file = input("请输入保存文件名 (按回车使用默认名称 'shortest_path.png'): ")
                if not output_file:
                    output_file = "shortest_path.png"
                
                visualizer.visualize_path(path, output_file=output_file)
        
        elif choice == "3":
            # 执行随机游走并可视化
            walk_path = analyzer.random_walk()
            print("随机游走路径: " + walk_path)
            
            # 保存到文件
            output_file_txt = "random_walk_result.txt"
            with open(output_file_txt, 'w', encoding='utf-8') as f:
                f.write(walk_path)
            print("随机游走结果已保存到文件: " + output_file_txt)
            
            # 可视化随机游走
            output_file = input("请输入保存图形文件名 (按回车使用默认名称 'random_walk.png'): ")
            if not output_file:
                output_file = "random_walk.png"
            
            visualizer.visualize_random_walk(walk_path, output_file=output_file)
        
        elif choice == "4":
            # 计算并可视化PageRank
            analyzer.calc_page_rank()
            
            top_n = input("请输入要显示的前N个重要节点数量 (按回车显示所有): ")
            if top_n and top_n.isdigit():
                top_n = int(top_n)
            else:
                top_n = None
            
            output_file = input("请输入保存图形文件名 (按回车使用默认名称 'pagerank.png'): ")
            if not output_file:
                output_file = "pagerank.png"
            
            visualizer.visualize_pagerank(output_file=output_file, top_n=top_n)
            
            # 显示排名前10的单词及其PageRank值
            pr_values = analyzer.graph.pr_values
            sorted_pr = sorted(pr_values.items(), key=lambda x: x[1], reverse=True)
            print("\nPageRank值最高的单词:")
            for i, (word, value) in enumerate(sorted_pr[:10]):
                print(f"{i+1}. {word}: {value:.6f}")
        
        elif choice == "5":
            # 获取多条路径
            paths = []
            while True:
                path = input("请输入路径（以空格分隔），或输入'done'结束: ")
                if path == 'done':
                    break
                paths.append(path.split())
            
            # 可视化多条路径
            output_file = input("请输入保存图形文件名 (按回车使用默认名称 'multi_paths.png'): ")
            if not output_file:
                output_file = "multi_paths.png"
            
            visualizer.visualize_all_paths(paths, output_file=output_file)
        
        elif choice == "0":
            break
        
        else:
            print("无效选项，请重新输入！")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"发生错误: {e}")
        print("请确保已安装所需的库: matplotlib 和 networkx")
        print("可以使用命令安装: pip install matplotlib networkx") 