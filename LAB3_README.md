# Lab3: 代码评审与单元测试

本实验实现了对Python项目的代码评审和单元测试，对应实验指导书中的要求。

## 环境准备

首先安装所需的依赖：

```bash
pip install -r requirements.txt
```

## 代码评审

代码评审使用以下工具：

1. **Pylint**: 检查代码风格和规范，相当于Java中的CheckStyle
2. **Flake8**: 结合了PyFlakes、pycodestyle和McCabe复杂度检查
3. **Bandit**: 检查安全漏洞，相当于Java中的SpotBugs

### 运行代码评审

使用以下命令运行代码评审：

```bash
# 评审所有Python文件
python code_review.py

# 评审特定文件
python code_review.py directed_graph.py graph_analyzer.py

# 只运行特定工具
python code_review.py --pylint
python code_review.py --flake8
python code_review.py --bandit
```

### 配置文件

- `.pylintrc`: Pylint配置文件
- `.flake8`: Flake8配置文件
- `.bandit`: Bandit配置文件

## 单元测试

单元测试使用pytest框架，包括黑盒测试和白盒测试：

1. **黑盒测试**: 测试函数的功能，不关注内部实现
2. **白盒测试**: 使用基本路径法测试函数的内部实现

### 运行单元测试

使用以下命令运行单元测试：

```bash
# 运行所有测试并生成覆盖率报告
python run_tests.py

# 只运行黑盒测试
python run_tests.py --black

# 只运行白盒测试
python run_tests.py --white

# 生成HTML格式的覆盖率报告
python run_tests.py --html

# 不生成覆盖率报告
python run_tests.py --no-cov
```

### 测试文件

- `test_blackbox.py`: 黑盒测试
- `test_whitebox.py`: 白盒测试

## 测试覆盖率

测试覆盖率使用pytest-cov插件，可以生成覆盖率报告：

- 命令行报告: 默认输出到终端
- HTML报告: 使用`--html`选项生成，保存在`htmlcov/`目录

## 对应实验指导书的要求

本实验实现了实验指导书中的以下要求：

1. **代码评审**:
   - Pylint: 对应Java中的CheckStyle，检查代码规范
   - Bandit: 对应Java中的SpotBugs，检查代码缺陷

2. **单元测试**:
   - 黑盒测试: 设计测试用例，测试函数功能
   - 白盒测试: 使用基本路径法，测试函数内部实现
   - 测试覆盖率: 使用pytest-cov统计测试覆盖率

## 注意事项

- 修复代码评审中发现的问题后，应该重新运行测试，确保代码质量
- 测试覆盖率不是越高越好，应该关注测试的质量和有效性
- 白盒测试需要了解代码内部实现，而黑盒测试只关注函数接口 