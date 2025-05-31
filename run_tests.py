#!/usr/bin/env python3
"""
测试运行脚本
运行单元测试并生成覆盖率报告
"""

import os
import sys
import subprocess
import argparse

def run_command(command, description):
    """运行命令并打印结果"""
    print(f"\n{'-' * 80}")
    print(f"{description}:")
    print(f"{'-' * 80}")
    
    try:
        result = subprocess.run(command, shell=True, check=False, 
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True)
        print(result.stdout)
        if result.stderr:
            print(f"错误信息:\n{result.stderr}")
        
        return result.returncode == 0
    except Exception as e:
        print(f"执行命令时出错: {e}")
        return False

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="测试运行工具")
    parser.add_argument('--black', action='store_true', help='只运行黑盒测试')
    parser.add_argument('--white', action='store_true', help='只运行白盒测试')
    parser.add_argument('--no-cov', action='store_true', help='不生成覆盖率报告')
    parser.add_argument('--html', action='store_true', help='生成HTML格式的覆盖率报告')
    
    args = parser.parse_args()
    
    # 构建测试命令
    test_files = []
    if args.black:
        test_files.append("test_blackbox.py")
    elif args.white:
        test_files.append("test_whitebox.py")
    else:
        test_files = ["test_blackbox.py", "test_whitebox.py"]
    
    test_cmd = "pytest " + " ".join(test_files)
    
    # 添加覆盖率选项
    if not args.no_cov:
        test_cmd += " --cov=directed_graph --cov=text_processor --cov=graph_analyzer"
        if args.html:
            test_cmd += " --cov-report=html"
        else:
            test_cmd += " --cov-report=term"
    
    # 运行测试
    success = run_command(test_cmd, "运行单元测试")
    
    if success:
        print("\n测试成功完成！")
        if not args.no_cov and args.html:
            print("HTML覆盖率报告已生成在 htmlcov/ 目录")
    else:
        print("\n测试失败，请检查错误信息。")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main()) 