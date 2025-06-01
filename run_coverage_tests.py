#!/usr/bin/env python3
"""
运行所有测试并生成覆盖率报告
"""

import os
import subprocess
import sys

def run_tests():
    """运行测试并生成覆盖率报告"""
    print("开始运行测试并生成覆盖率报告...")
    
    # 运行测试并生成覆盖率报告
    cmd = [
        sys.executable, "-m", "pytest", 
        "test_whitebox.py",  # 使用已有的测试文件
        "test_coverage.py",  # 使用新创建的高覆盖率测试文件
        "--cov=directed_graph", 
        "--cov=graph_analyzer", 
        "--cov=text_processor", 
        "--cov-report=term", 
        "--cov-report=html"
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print("\n覆盖率报告已生成！")
        print("HTML报告位于: htmlcov/index.html")
    except subprocess.CalledProcessError:
        print("测试运行失败，请检查错误信息")
        return False
    
    return True

if __name__ == "__main__":
    run_tests() 