import os
import sys
import codecs

from directed_graph import DirectedGraph
from text_processor import TextProcessor
from graph_analyzer import GraphAnalyzer

def main():
    # 设置控制台编码以正确显示中文
    if sys.platform.startswith('win'):
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)
    
    print("欢迎使用文本图分析程序！")
    
    # 获取文件路径
    while True:
        file_path = input("请输入文本文件路径（或直接按回车使用默认文件'Easy Test.txt'）: ")
        if not file_path:
            file_path = "lab1/Easy Test.txt"
        elif not os.path.isabs(file_path) and not file_path.startswith('lab1/'):
            file_path = os.path.join('lab1', file_path)
        
        if os.path.exists(file_path):
            break
        else:
            print("文件不存在，请重新输入！")
    
    # 处理文件并构建图
    processor = TextProcessor()
    graph = processor.process_file(file_path)
    analyzer = GraphAnalyzer(graph)
    
    # 展示功能菜单
    while True:
        print("\n请选择功能：")
        print("1. 展示有向图")
        print("2. 查询桥接词")
        print("3. 根据桥接词生成新文本")
        print("4. 计算最短路径")
        print("5. 计算PageRank值")
        print("6. 随机游走")
        print("7. 图形可视化")
        print("0. 退出程序")
        
        choice = input("请输入选项编号: ")
        
        if choice == "1":
            print(analyzer.show_directed_graph())
        
        elif choice == "2":
            word1 = input("请输入第一个单词: ")
            word2 = input("请输入第二个单词: ")
            print(analyzer.query_bridge_words(word1, word2))
        
        elif choice == "3":
            text = input("请输入文本: ")
            new_text = analyzer.generate_new_text(text)
            print("生成的新文本: " + new_text)
        
        elif choice == "4":
            # 最短路径子菜单
            print("\n最短路径计算：")
            print("1. 计算两个单词之间的最短路径")
            print("2. 计算一个单词到所有其他单词的最短路径")
            print("3. 计算两个单词之间的所有最短路径")
            print("0. 返回主菜单")
            
            path_choice = input("请选择功能: ")
            
            if path_choice == "1":
                word1 = input("请输入起始单词: ")
                word2 = input("请输入目标单词: ")
                path_result = analyzer.calc_shortest_path(word1, word2)
                if isinstance(path_result, tuple):
                    print(path_result[0])  # 只打印结果字符串
                else:
                    print(path_result)
                
            elif path_choice == "2":
                word = input("请输入起始单词: ")
                result = analyzer.calc_all_shortest_paths(word)
                
                if isinstance(result, tuple):
                    result_str, paths_dict = result
                    print(result_str)
                    
                    # 询问是否保存到文件
                    save_choice = input("是否将结果保存到文件? (y/n): ")
                    if save_choice.lower() == 'y':
                        output_file = "all_paths_from_{}.txt".format(word)
                        with open(output_file, 'w', encoding='utf-8') as f:
                            f.write(result_str)
                        print("结果已保存到文件:", output_file)
                else:
                    print(result)
            
            elif path_choice == "3":
                word1 = input("请输入起始单词: ")
                word2 = input("请输入目标单词: ")
                result = analyzer.calc_all_shortest_paths_between(word1, word2)
                
                if isinstance(result, tuple):
                    result_str, paths_with_weights = result
                    print(result_str)
                    
                    # 询问是否保存到文件
                    save_choice = input("是否将结果保存到文件? (y/n): ")
                    if save_choice.lower() == 'y':
                        output_file = "all_shortest_paths_from_{}_to_{}.txt".format(word1, word2)
                        with open(output_file, 'w', encoding='utf-8') as f:
                            f.write(result_str)
                        print("结果已保存到文件:", output_file)
                else:
                    print(result)
            
            elif path_choice == "0":
                continue
            
            else:
                print("无效选项!")
        
        elif choice == "5":
            use_tf_idf = input("是否使用TF-IDF优化初始PageRank值? (y/n): ").lower() == 'y'
            analyzer.calc_page_rank(text_processor=processor, use_tf_idf=use_tf_idf)
            word = input("请输入要查询PageRank值的单词（直接回车显示所有）: ")
            if word:
                pr_value = analyzer.get_page_rank(word)
                print("单词 '{}' 的PageRank值为: {}".format(word, pr_value))
            else:
                pr_values = analyzer.graph.pr_values
                sorted_pr = sorted(pr_values.items(), key=lambda x: x[1], reverse=True)
                print("所有单词的PageRank值（按降序排列）:")
                for word, value in sorted_pr:
                    print("  {}: {:.6f}".format(word, value))
                
                # 显示TF-IDF值（如果使用）
                if use_tf_idf:
                    print("\n单词的TF-IDF值（仅显示前10个）:")
                    sorted_tf_idf = sorted(analyzer.tf_idf_values.items(), key=lambda x: x[1], reverse=True)
                    for word, value in sorted_tf_idf[:10]:
                        print("  {}: {:.6f}".format(word, value))
        
        elif choice == "6":
            path = analyzer.random_walk()
            print("随机游走路径: " + path)
            
            # 保存到文件
            output_file = "random_walk_result.txt"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(path)
            print("随机游走结果已保存到文件: " + output_file)
            
        elif choice == "7":
            try:
                # 导入可视化模块
                from graph_visualization import GraphVisualizer
                
                visualizer = GraphVisualizer(graph)
                
                # 展示可视化子菜单
                while True:
                    print("\n图形可视化功能：")
                    print("1. 可视化有向图并保存")
                    print("2. 可视化两个单词间的最短路径")
                    print("3. 可视化随机游走路径")
                    print("4. 可视化PageRank值")
                    print("5. 查看一个单词到指定目标单词的最短路径")
                    print("6. 可视化两个单词间的所有最短路径")
                    print("0. 返回主菜单")
                    
                    vis_choice = input("请输入选项编号: ")
                    
                    if vis_choice == "1":
                        output_file = input("请输入保存文件名 (按回车使用默认名称 'graph.png'): ")
                        if not output_file:
                            output_file = "graph.png"
                        visualizer.visualize_graph(output_file=output_file)
                    
                    elif vis_choice == "2":
                        word1 = input("请输入起始单词: ")
                        word2 = input("请输入目标单词: ")
                        
                        # 获取最短路径
                        path_result = analyzer.calc_shortest_path(word1, word2)
                        
                        # 判断是否找到路径
                        if isinstance(path_result, tuple):
                            result_str, path, path_weight = path_result
                            print(result_str)
                            
                            output_file = input("请输入保存文件名 (按回车使用默认名称 'shortest_path.png'): ")
                            if not output_file:
                                output_file = "shortest_path.png"
                            
                            visualizer.visualize_path(path, output_file=output_file, path_length=path_weight)
                        else:
                            print(path_result)  # 打印错误信息
                    
                    elif vis_choice == "3":
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
                        
                    elif vis_choice == "4":
                        # 计算并可视化PageRank
                        use_tf_idf = input("是否使用TF-IDF优化初始PageRank值? (y/n): ").lower() == 'y'
                        analyzer.calc_page_rank(text_processor=processor, use_tf_idf=use_tf_idf)
                        
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
                            
                        # 显示TF-IDF值（如果使用）
                        if use_tf_idf:
                            print("\n单词的TF-IDF值（仅显示前10个）:")
                            sorted_tf_idf = sorted(analyzer.tf_idf_values.items(), key=lambda x: x[1], reverse=True)
                            for i, (word, value) in enumerate(sorted_tf_idf[:10]):
                                print(f"{i+1}. {word}: {value:.6f}")
                    
                    elif vis_choice == "5":
                        # 计算一个单词到指定目标单词的最短路径
                        source = input("请输入起始单词: ")
                        
                        # 检查单词是否在图中
                        if source not in analyzer.graph.nodes:
                            print(f"单词 '{source}' 不在图中!")
                            continue
                        
                        # 获取最短路径结果
                        result = analyzer.calc_all_shortest_paths(source)
                        
                        if not isinstance(result, tuple):
                            print(result)  # 打印错误信息
                            continue
                            
                        result_str, paths_dict = result
                        
                        # 显示前10个可达节点（或全部如果少于10个）
                        available_targets = list(paths_dict.keys())
                        print(f"从 '{source}' 可到达的单词有 {len(available_targets)} 个")
                        
                        if not available_targets:
                            print("没有可到达的单词")
                            continue
                        
                        print("可到达的单词示例（最多显示10个）:")
                        for i, target in enumerate(available_targets[:10]):
                            path, weight = paths_dict[target]
                            print(f"{i+1}. {target} (路径长度: {weight})")
                        
                        # 让用户选择目标单词
                        target = input("请输入目标单词（查看到该单词的最短路径）: ")
                        
                        if target not in paths_dict:
                            print(f"'{target}' 不在可到达的单词列表中或不是有效的单词")
                            continue
                        
                        # 获取所选目标的路径信息
                        path, weight = paths_dict[target]
                        
                        # 可视化该路径
                        output_file = input("请输入保存文件名 (按回车使用默认名称 'path_to_word.png'): ")
                        if not output_file:
                            output_file = f"path_from_{source}_to_{target}.png"
                        
                        visualizer.visualize_path(path, output_file=output_file, path_length=weight)
                        print(f"从 '{source}' 到 '{target}' 的最短路径已可视化，总权重: {weight}")
                    
                    elif vis_choice == "6":
                        # 计算并可视化两个单词间的所有最短路径
                        word1 = input("请输入起始单词: ")
                        word2 = input("请输入目标单词: ")
                        
                        # 获取所有最短路径
                        result = analyzer.calc_all_shortest_paths_between(word1, word2)
                        
                        if not isinstance(result, tuple):
                            print(result)  # 打印错误信息
                            continue
                            
                        result_str, paths_with_weights = result
                        print(result_str)  # 打印路径信息
                        
                        # 可视化所有最短路径
                        output_file = input("请输入保存文件名 (按回车使用默认名称 'all_shortest_paths.png'): ")
                        if not output_file:
                            output_file = f"all_shortest_paths_from_{word1}_to_{word2}.png"
                        
                        # 提取所有路径
                        paths = [path for path, _ in paths_with_weights]
                        
                        # 调用可视化方法
                        visualizer.visualize_all_paths(paths, output_file=output_file, 
                                                      title=f"所有从 {word1} 到 {word2} 的最短路径")
                    
                    elif vis_choice == "0":
                        break
                    
                    else:
                        print("无效选项，请重新输入！")
                        
            except ImportError as e:
                print(f"未能导入图形可视化模块: {e}")
                print("请确保已安装所需的库：matplotlib 和 networkx")
                print("可以使用以下命令安装：pip install matplotlib networkx")
        
        elif choice == "0":
            print("感谢使用，再见！")
            break
        
        else:
            print("无效选项，请重新输入！")


if __name__ == "__main__":
    main() 