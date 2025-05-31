#!/usr/bin/env python3
"""
运行代码评审和测试并将结果保存到日志文件
"""

import os
import sys
import subprocess
import datetime
import argparse

# 添加用户Python脚本目录到PATH
PYTHON_USER_SCRIPTS = os.path.join(os.path.expanduser("~"), "AppData", "Roaming", "Python", "Python311", "Scripts")
if os.path.exists(PYTHON_USER_SCRIPTS) and PYTHON_USER_SCRIPTS not in os.environ.get("PATH", ""):
    os.environ["PATH"] = PYTHON_USER_SCRIPTS + os.pathsep + os.environ.get("PATH", "")

def run_command_with_log(command, log_file, description):
    """运行命令并将输出保存到日志文件"""
    print(f"\n{'-' * 80}")
    print(f"{description}:")
    print(f"{'-' * 80}")
    
    # 写入日志文件
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f"\n{'-' * 80}\n")
        f.write(f"{description}:\n")
        f.write(f"{'-' * 80}\n")
        f.write(f"命令: {command}\n")
        f.write(f"时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    try:
        result = subprocess.run(command, shell=True, check=False, 
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True)
        
        # 打印到控制台
        print(result.stdout)
        if result.stderr:
            print(f"错误信息:\n{result.stderr}")
        
        # 写入日志文件
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(result.stdout)
            if result.stderr:
                f.write(f"\n错误信息:\n{result.stderr}")
        
        return result.returncode == 0
    except Exception as e:
        error_msg = f"执行命令时出错: {e}"
        print(error_msg)
        
        # 写入日志文件
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"\n{error_msg}\n")
        
        return False

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="运行代码评审和测试并记录日志")
    parser.add_argument('--no-review', action='store_true', help='跳过代码评审')
    parser.add_argument('--no-test', action='store_true', help='跳过单元测试')
    parser.add_argument('--log-file', type=str, default="code_assessment_log.txt", 
                        help='日志文件名 (默认: code_assessment_log.txt)')
    
    args = parser.parse_args()
    log_file = args.log_file
    
    # 创建或清空日志文件
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write(f"代码评审和测试日志\n")
        f.write(f"开始时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'=' * 80}\n\n")
    
    success = True
    
    # 运行代码评审
    if not args.no_review:
        print("运行代码评审...")
        
        # 运行pylint
        if not run_command_with_log("python -m pylint directed_graph.py graph_analyzer.py text_processor.py", log_file, "运行Pylint检查"):
            success = False
        
        # 运行flake8
        if not run_command_with_log("python -m flake8 directed_graph.py graph_analyzer.py text_processor.py", log_file, "运行Flake8检查"):
            success = False
        
        # 运行bandit
        if not run_command_with_log("python -m bandit -c .bandit directed_graph.py graph_analyzer.py text_processor.py", log_file, "运行Bandit检查"):
            success = False
    
    # 运行单元测试
    if not args.no_test:
        print("运行单元测试...")
        
        # 运行黑盒测试
        if not run_command_with_log("python -m pytest test_blackbox.py -v", log_file, "运行黑盒测试"):
            success = False
        
        # 运行白盒测试
        if not run_command_with_log("python -m pytest test_whitebox.py -v", log_file, "运行白盒测试"):
            success = False
        
        # 运行测试覆盖率分析
        if not run_command_with_log("python -m pytest --cov=directed_graph --cov=text_processor --cov=graph_analyzer --cov-report=html", log_file, "运行测试覆盖率分析"):
            success = False
    
    # 写入总结
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f"\n\n{'=' * 80}\n")
        f.write(f"结束时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        if success:
            f.write("总体评估: 成功\n")
        else:
            f.write("总体评估: 存在一些问题，请查看上面的详细信息\n")
    
    print(f"\n评估完成！结果已保存到 {log_file}")
    if not success:
        print("存在一些问题，请查看日志文件获取详细信息")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main()) 