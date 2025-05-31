#!/usr/bin/env python3
"""
代码评审脚本
运行pylint、flake8和bandit进行代码静态分析
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
    parser = argparse.ArgumentParser(description="代码评审工具")
    parser.add_argument('files', nargs='*', help='要评审的文件，如果为空则评审所有Python文件')
    parser.add_argument('--pylint', action='store_true', help='只运行pylint')
    parser.add_argument('--flake8', action='store_true', help='只运行flake8')
    parser.add_argument('--bandit', action='store_true', help='只运行bandit')
    
    args = parser.parse_args()
    
    # 如果没有指定文件，则默认评审所有Python文件
    files = args.files
    if not files:
        files = [f for f in os.listdir('.') if f.endswith('.py') and not f.startswith('test_')]
    
    # 如果没有指定特定工具，则运行所有工具
    run_all = not (args.pylint or args.flake8 or args.bandit)
    
    success = True
    
    # 运行pylint
    if args.pylint or run_all:
        for file in files:
            cmd = f"pylint {file}"
            if not run_command(cmd, f"运行Pylint检查 {file}"):
                success = False
    
    # 运行flake8
    if args.flake8 or run_all:
        for file in files:
            cmd = f"flake8 {file}"
            if not run_command(cmd, f"运行Flake8检查 {file}"):
                success = False
    
    # 运行bandit
    if args.bandit or run_all:
        for file in files:
            cmd = f"bandit -c .bandit {file}"
            if not run_command(cmd, f"运行Bandit检查 {file}"):
                success = False
    
    if success:
        print("\n所有检查通过！")
        return 0
    else:
        print("\n有一些检查失败，请修复上述问题。")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 