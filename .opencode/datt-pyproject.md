# DATT → uv + pyproject.toml — Clean Migration Plan

## Intent

将 DATT 代码库从手工 `pip install -r requirements.txt` + `PYTHONPATH` 的野生状态彻底迁移到 `uv` + `pyproject.toml` 标准包管理。不留向后兼容包袱。

## Feasibility

**feasible** — 所有源文件使用绝对导入 `from DATT.xxx import ...`，移入 `DATT/` 命名空间目录后零 import 修改。依赖集合明确，版本锁定充分。

### 硬约束

- Python 上限 `< 3.11`：`gym==0.21.0` 是旧 gym API（非 gymnasium），不支持 Python 3.11+。`stable-baselines3==1.6.2` 同样绑定此版本。升级 gym→gymnasium + SB3→2.x 需改写 9 个文件中的 `gym.Env`/`gym.spaces` 引用，属独立迁移任务，不纳入本次范围。
- `uv` 会自动解析 torch 的 CUDA wheel，用户需确保系统有 CUDA 11.7+ 和 cuDNN。

## Task

重构 DATT 仓库为 pip 可安装标准包，删除所有旧管理模式残留，交付一个纯净的 `uv sync` 即用项目。

## Deliverables

1. 删除 `requirements.txt`
2. 删除 `DATT_profile.sh`
3. `DATT/` 命名空间包目录（含 `__init__.py`，收纳全部 7 个子包 + main.py）
4. `pyproject.toml`（依赖、构建、入口点、lint/test 配置）
5. 包装后的 `DATT/main.py`（`def main():` + 兼容 `if __name__`）
6. `.gitignore` 追加 uv 产物排除项
7. `uv sync` 通过的验证结果

## Acceptance

1. `requirements.txt` 和 `DATT_profile.sh` 不存在于仓库根
2. `uv sync` 成功安装全部依赖（79 packages，含 torch 1.13.1 + CUDA 11）
3. 以下命令全部成功（无需 `pip install -e .`，uv sync 已自动执行 editable install）：
   - `python -c "from DATT.quadsim.sim import QuadSim"` → OK
   - `python -c "from DATT.learning.configs import RLAlgo"` → OK
   - `python -c "from DATT.main import main"` → OK
   - `python -c "from DATT.controllers import ControllersZoo"` → OK
   - `datt --help` / `datt-train --help` / `datt-eval --help` → 全部可用

### 实际偏差

- **gym 版本**：`gym==0.21.0` 的 setup.py 依赖声明格式有 bug（`opencv-python>=3.`），无法在现代 setuptools 下构建且无 Python 3.10 wheel。改为 `gym==0.22.0` + `[tool.uv.override-dependencies]` 覆盖 SB3 的 `gym==0.21` 硬钉。API 完全兼容（同属旧 gym 包）。
- **Python 下限**：从 `>=3.8` 提升到 `>=3.9`，因为 scipy>=1.11 不再支持 3.8。
- **SB3 版本**：解析为 `1.6.1`（非 `1.6.2`），在 `>=1.6, <2` 范围内。
4. `datt-train --help` 和 `datt-eval --help` 可执行

## Child Tasks

### Child 1: Delete legacy files

- **Deliverable**: `requirements.txt` 和 `DATT_profile.sh` 从仓库中删除
- **Depends on**: 无
- **Completion signal**: 两个文件不再存在于仓库根目录

### Child 2: Update .gitignore

- **Deliverable**: `.gitignore` 新增以下条目：`dist/`, `*.egg-info/`, `.venv/`, `uv.lock` (可选)
- **Depends on**: 无
- **Completion signal**: `rg "dist/" .gitignore` 有结果

### Child 3: Create DATT/ package scaffold

- **Deliverable**: `DATT/` 目录 + `DATT/__init__.py`（空文件或仅含包注释）
- **Depends on**: 无
- **Completion signal**: `DATT/__init__.py` 存在

### Child 4: Move all subpackages into DATT/

- **Deliverable**: `learning/`, `quadsim/`, `controllers/`, `refs/`, `python_utils/`, `utils/`, `configuration/` 从仓库根移至 `DATT/` 下，使用 `git mv`
- **Depends on**: Child 3
- **Completion signal**: 仓库根不再有上述目录；`ls DATT/learning/__init__.py` 返回成功

### Child 5: Wrap main.py with main() and move to DATT/

- **Deliverable**: `DATT/main.py`，结构如下：
  ```
  # 第 1-19 行：原有 import
  def main():
      # 原 第 21-116 行（if __name__ == "__main__": 内全部代码）
  if __name__ == "__main__":
      main()
  ```
- **Depends on**: Child 3
- **Completion signal**: `from DATT.main import main` 可导入，`main` 为 callable

### Child 6: Write pyproject.toml

- **Deliverable**: `/home/rec/DATT/pyproject.toml`
- **Depends on**: Child 4（包路径需已存在）
- **Completion signal**: `uv sync` 执行成功

    内容：

    ```toml
    [build-system]
    requires = ["setuptools>=61.0", "wheel"]
    build-backend = "setuptools.build_meta"

    [project]
    name = "datt"
    version = "0.1.0"
    description = "DATT: Deep Adaptive Trajectory Tracking for Quadrotor Control"
    readme = "README.md"
    license = {text = "MIT"}
    requires-python = ">=3.8, <3.11"
    dependencies = [
        "numpy>=1.23, <2",
        "scipy>=1.11",
        "torch>=1.13, <2",
        "gym==0.21.0",
        "stable-baselines3>=1.6, <2",
        "matplotlib>=3.2",
        "pygame>=2.4",
        "tqdm>=4.64",
        "joblib>=0.14",
        "meshcat>=0.3.2",
        "Pillow>=10.1",
        "rich>=13.6",
    ]

    [project.optional-dependencies]
    dev = [
        "pytest>=8.0",
        "ruff>=0.1.0",
        "ipython>=8.0",
    ]

    [project.scripts]
    datt = "DATT.main:main"
    datt-train = "DATT.learning.train_policy:train"
    datt-eval = "DATT.learning.eval_policy:eval"

    [tool.setuptools.packages.find]
    where = ["."]
    include = ["DATT*"]

    [tool.ruff]
    line-length = 100
    target-version = "py310"

    [tool.ruff.lint]
    select = ["E", "F", "I", "W"]

    [tool.pytest.ini_options]
    testpaths = ["tests"]
    python_files = ["test_*.py"]
    python_functions = ["test_*"]
    ```

### Child 7: Verify end-to-end

- **Deliverable**: 验证报告
- **Depends on**: Child 1-6
- **Completion signal**: 以下全部通过：
    1. `uv sync` 无错误
    2. `python -c "from DATT.quadsim.sim import QuadSim; print('OK')"` 输出 `OK`
    3. `python -c "from DATT.learning.configs import RLAlgo; print('OK')"` 输出 `OK`
    4. `python -c "from DATT.main import main; print('OK')"` 输出 `OK`
    5. `python -c "from DATT.controllers import ControllersZoo; print('OK')"` 输出 `OK`

## Rules

- 不修改任何源文件的业务逻辑（main.py 仅加 `def main():` 包装层，内部代码不变）
- 所有目录移动使用 `git mv` 保留历史
- 不再保留 `requirements.txt` / `DATT_profile.sh` / PYTHONPATH 方式
- uv 创建的 `.venv/` 由 `.gitignore` 排除，不纳入仓库

## Verification

- [ ] `requirements.txt` 不存在
- [ ] `DATT_profile.sh` 不存在
- [ ] `.gitignore` 含 `dist/`, `*.egg-info/`, `.venv/`
- [ ] `DATT/__init__.py` 存在
- [ ] 根目录无 `learning/`, `quadsim/`, `controllers/`, `refs/`, `python_utils/`, `utils/`, `configuration/`
- [ ] `DATT/main.py` 含 `def main():` 且在文件末尾有 `if __name__ == "__main__": main()`
- [ ] `pyproject.toml` 存在，内容与 Child 6 一致
- [ ] `uv sync` 成功
- [ ] 5 条 import 验证全部输出 `OK`
