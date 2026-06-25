# Paper Prototype Dev

本地论文创新点原型开发项目骨架。

## 目标

- 将论文中的核心创新点拆成可验证的工程原型
- 快速建立 baseline、实验脚本和测试闭环
- 后续根据具体论文/方向补齐算法实现、数据集和评测指标

## 目录结构

```text
.
├── docs/                 # 需求、设计、实验记录
├── src/paper_prototype/  # Python 原型代码
├── cpp/                  # C/C++ 原型或性能核心
├── tests/                # 自动化测试
├── scripts/              # 运行/构建脚本
├── experiments/          # 实验配置和输出说明
└── data/                 # 本地数据占位，不纳入 git
```

## 快速开始

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
pytest -q
```

## 下一步

1. 明确项目名、论文/创新点、技术路线
2. 在 `docs/requirements.md` 写验收指标
3. 在 `docs/design.md` 写模块设计
4. 实现 baseline + 最小可复现实验
