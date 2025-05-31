#!/usr/bin/env python3
"""
运行所有测试并生成覆盖率报告
"""

import os
import sys
import subprocess
import datetime

def main():
    """运行测试并生成报告"""
    start_time = datetime.datetime.now()
    print(f"开始运行测试: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # 运行黑盒测试
    print("\n运行黑盒测试:")
    print("-"*80)
    subprocess.run(["python", "-m", "pytest", "test_blackbox.py", "-v"], check=False)
    
    # 运行白盒测试
    print("\n运行白盒测试:")
    print("-"*80)
    subprocess.run(["python", "-m", "pytest", "test_whitebox.py", "-v"], check=False)
    
    # 运行覆盖率提升测试
    print("\n运行覆盖率提升测试:")
    print("-"*80)
    subprocess.run(["python", "-m", "pytest", "test_coverage.py", "-v"], check=False)
    
    # 生成覆盖率报告
    print("\n生成覆盖率报告:")
    print("-"*80)
    subprocess.run([
        "python", "-m", "pytest",
        "--cov=directed_graph",
        "--cov=text_processor",
        "--cov=graph_analyzer",
        "--cov-report=html",
        "--cov-report=term",
        "test_blackbox.py", "test_whitebox.py", "test_coverage.py"
    ], check=False)
    
    end_time = datetime.datetime.now()
    duration = end_time - start_time
    print("="*80)
    print(f"测试完成: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"总耗时: {duration.total_seconds():.2f} 秒")
    print(f"覆盖率报告已生成在 htmlcov 目录中")
    print("可以通过浏览器打开 htmlcov/index.html 查看详细报告")

if __name__ == "__main__":
    main() 