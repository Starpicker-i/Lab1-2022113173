#!/usr/bin/env python3
"""
生成评估报告
分析代码评审和测试结果，生成评估报告
"""

import os
import re
import datetime
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def parse_pylint_score(log_content):
    """解析Pylint评分"""
    match = re.search(r'Your code has been rated at (\d+\.\d+)/10', log_content)
    if match:
        return float(match.group(1))
    return 0.0

def count_issues(log_content, tool_name):
    """计算特定工具的问题数量"""
    if tool_name == 'pylint':
        # 计算pylint问题
        issues = re.findall(r'[A-Za-z_]+\.py:\d+:\d+: \w\d+:', log_content)
        return len(issues)
    elif tool_name == 'flake8':
        # 计算flake8问题
        issues = re.findall(r'[A-Za-z_]+\.py:\d+:\d+: \w\d+', log_content)
        return len(issues)
    elif tool_name == 'bandit':
        # 计算bandit问题
        match = re.search(r'Total issues \(by severity\):\s+Undefined: (\d+)\s+Low: (\d+)\s+Medium: (\d+)\s+High: (\d+)', log_content, re.DOTALL)
        if match:
            return sum(int(match.group(i)) for i in range(1, 5))
    return 0

def parse_test_results(log_content):
    """解析测试结果"""
    # 黑盒测试
    blackbox_match = re.search(r'test_blackbox\.py.+?(\d+) passed', log_content)
    blackbox_passed = int(blackbox_match.group(1)) if blackbox_match else 0
    
    # 白盒测试
    whitebox_match = re.search(r'test_whitebox\.py.+?(\d+) passed', log_content)
    whitebox_passed = int(whitebox_match.group(1)) if whitebox_match else 0
    
    # 总测试
    total_match = re.search(r'(\d+) passed in \d+\.\d+s', log_content)
    total_passed = int(total_match.group(1)) if total_match else 0
    
    return {
        'blackbox_passed': blackbox_passed,
        'whitebox_passed': whitebox_passed,
        'total_passed': total_passed
    }

def parse_coverage(log_content):
    """解析覆盖率"""
    # 这里简化处理，实际项目中应该从覆盖率报告中提取更详细的信息
    match = re.search(r'Coverage HTML written to dir htmlcov', log_content)
    return True if match else False

def generate_charts(report_data, output_dir):
    """生成图表"""
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 代码评审问题图表
    plt.figure(figsize=(10, 6))
    tools = ['Pylint', 'Flake8', 'Bandit']
    issues = [report_data['pylint_issues'], report_data['flake8_issues'], report_data['bandit_issues']]
    plt.bar(tools, issues, color=['#ff9999', '#66b3ff', '#99ff99'])
    plt.title('代码评审问题数量')
    plt.xlabel('工具')
    plt.ylabel('问题数量')
    for i, v in enumerate(issues):
        plt.text(i, v + 0.5, str(v), ha='center')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'code_review_issues.png'))
    plt.close()
    
    # 测试通过率图表
    plt.figure(figsize=(10, 6))
    test_types = ['黑盒测试', '白盒测试', '总测试']
    passed = [report_data['blackbox_passed'], report_data['whitebox_passed'], report_data['total_passed']]
    total = [5, 4, 9]  # 硬编码测试数量，实际项目中应该动态获取
    pass_rate = [p/t*100 for p, t in zip(passed, total)]
    
    plt.bar(test_types, pass_rate, color=['#66b3ff', '#99ff99', '#ffcc99'])
    plt.title('测试通过率')
    plt.xlabel('测试类型')
    plt.ylabel('通过率 (%)')
    plt.ylim(0, 100)
    for i, v in enumerate(pass_rate):
        plt.text(i, v + 2, f'{v:.1f}%', ha='center')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'test_pass_rate.png'))
    plt.close()
    
    # Pylint评分图表
    plt.figure(figsize=(8, 6))
    score = report_data['pylint_score']
    plt.pie([score, 10-score], labels=[f'得分: {score}/10', ''], colors=['#66b3ff', '#f0f0f0'],
            autopct=lambda p: f'{p:.1f}%' if p > 5 else '', startangle=90)
    plt.title('Pylint代码质量评分')
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pylint_score.png'))
    plt.close()

def generate_report(log_file, output_file, charts_dir):
    """生成评估报告"""
    # 读取日志文件
    with open(log_file, 'r', encoding='utf-8') as f:
        log_content = f.read()
    
    # 解析数据
    pylint_score = parse_pylint_score(log_content)
    pylint_issues = count_issues(log_content, 'pylint')
    flake8_issues = count_issues(log_content, 'flake8')
    bandit_issues = count_issues(log_content, 'bandit')
    test_results = parse_test_results(log_content)
    has_coverage = parse_coverage(log_content)
    
    # 整合数据
    report_data = {
        'pylint_score': pylint_score,
        'pylint_issues': pylint_issues,
        'flake8_issues': flake8_issues,
        'bandit_issues': bandit_issues,
        'blackbox_passed': test_results['blackbox_passed'],
        'whitebox_passed': test_results['whitebox_passed'],
        'total_passed': test_results['total_passed'],
        'has_coverage': has_coverage,
        'date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # 生成图表
    generate_charts(report_data, charts_dir)
    
    # 生成HTML报告
    html_report = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>代码评审与测试报告</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                line-height: 1.6;
                margin: 0;
                padding: 20px;
                color: #333;
            }}
            .container {{
                max-width: 1000px;
                margin: 0 auto;
                background: #fff;
                padding: 20px;
                box-shadow: 0 0 10px rgba(0,0,0,0.1);
            }}
            h1, h2, h3 {{
                color: #2c3e50;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
            }}
            th, td {{
                padding: 12px 15px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }}
            th {{
                background-color: #f8f9fa;
            }}
            .chart {{
                margin: 20px 0;
                text-align: center;
            }}
            .chart img {{
                max-width: 100%;
                height: auto;
            }}
            .summary {{
                background-color: #f8f9fa;
                padding: 15px;
                border-radius: 5px;
                margin: 20px 0;
            }}
            .good {{
                color: #28a745;
            }}
            .warning {{
                color: #ffc107;
            }}
            .danger {{
                color: #dc3545;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>代码评审与测试报告</h1>
            <p>生成时间: {report_data['date']}</p>
            
            <div class="summary">
                <h2>总结</h2>
                <p>
                    代码质量评分: <span class="{'good' if pylint_score >= 7 else 'warning' if pylint_score >= 5 else 'danger'}">{pylint_score}/10</span><br>
                    代码问题总数: {pylint_issues + flake8_issues + bandit_issues}<br>
                    测试通过率: <span class="{'good' if test_results['total_passed'] == 9 else 'warning' if test_results['total_passed'] >= 7 else 'danger'}">{test_results['total_passed']}/9 ({test_results['total_passed']/9*100:.1f}%)</span><br>
                    覆盖率报告: {'已生成' if has_coverage else '未生成'}
                </p>
            </div>
            
            <h2>代码评审结果</h2>
            <table>
                <tr>
                    <th>工具</th>
                    <th>问题数量</th>
                    <th>评分</th>
                </tr>
                <tr>
                    <td>Pylint</td>
                    <td>{pylint_issues}</td>
                    <td>{pylint_score}/10</td>
                </tr>
                <tr>
                    <td>Flake8</td>
                    <td>{flake8_issues}</td>
                    <td>-</td>
                </tr>
                <tr>
                    <td>Bandit</td>
                    <td>{bandit_issues}</td>
                    <td>-</td>
                </tr>
            </table>
            
            <div class="chart">
                <h3>代码评审问题分布</h3>
                <img src="charts/code_review_issues.png" alt="代码评审问题分布">
            </div>
            
            <div class="chart">
                <h3>Pylint代码质量评分</h3>
                <img src="charts/pylint_score.png" alt="Pylint代码质量评分">
            </div>
            
            <h2>测试结果</h2>
            <table>
                <tr>
                    <th>测试类型</th>
                    <th>通过数量</th>
                    <th>总数</th>
                    <th>通过率</th>
                </tr>
                <tr>
                    <td>黑盒测试</td>
                    <td>{test_results['blackbox_passed']}</td>
                    <td>5</td>
                    <td>{test_results['blackbox_passed']/5*100:.1f}%</td>
                </tr>
                <tr>
                    <td>白盒测试</td>
                    <td>{test_results['whitebox_passed']}</td>
                    <td>4</td>
                    <td>{test_results['whitebox_passed']/4*100:.1f}%</td>
                </tr>
                <tr>
                    <td>总计</td>
                    <td>{test_results['total_passed']}</td>
                    <td>9</td>
                    <td>{test_results['total_passed']/9*100:.1f}%</td>
                </tr>
            </table>
            
            <div class="chart">
                <h3>测试通过率</h3>
                <img src="charts/test_pass_rate.png" alt="测试通过率">
            </div>
            
            <h2>建议</h2>
            <ul>
                <li>{'代码质量良好，可以继续保持。' if pylint_score >= 7 else '代码质量有待提高，建议解决Pylint和Flake8中的问题。'}</li>
                <li>{'所有测试都已通过，测试覆盖良好。' if test_results['total_passed'] == 9 else '有测试未通过，需要修复相关问题。'}</li>
                <li>{'代码中没有安全漏洞，继续保持。' if bandit_issues == 0 else '存在安全漏洞，建议尽快修复。'}</li>
            </ul>
        </div>
    </body>
    </html>
    """
    
    # 写入HTML报告
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_report)
    
    print(f"报告已生成: {output_file}")
    print(f"图表已保存到: {charts_dir}")

if __name__ == "__main__":
    log_file = "code_assessment_log.txt"
    output_file = "code_assessment_report.html"
    charts_dir = "charts"
    
    generate_report(log_file, output_file, charts_dir) 