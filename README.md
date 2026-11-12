# ML Assignment 2 · C++ Tiny Deep Learning Framework

[排行榜](https://coideas.me/ml-assignment-2/) · [课程仓库](https://github.com/ZeroGen-Lab/ML-Assignment-2)

本作业要求仅使用 C++17 标准库实现一个小型深度学习框架，并用它训练 MLP、CNN、LSTM 三个模型。

作业采用填空形式：张量、模型结构、数据加载和训练循环均已给出，需要实现的是框架的核心部分，即各算子的前向与反向传播、自动求导引擎、网络层和优化器。
三个模型的梯度（包括 LSTM 沿时间展开后的反向传播）均由所实现的自动求导引擎计算，不得为单个模型手写梯度。

- **成绩占比**：课程总评的 35%。每组隐藏种子按误差定档，三个模型分别计 credit，详见[评分](#评分)。
- **提交次数**：每人共 **3 次**。
- **截止时间**：**2026 年 12 月 3 日 23:59:59（北京时间）**。

## 快速开始

```bash
git clone https://github.com/ZeroGen-Lab/ML-Assignment-2.git
cd ML-Assignment-2
make            # 编译，确认编译环境可用
make test       # 初始状态下仅示例 ops/add 通过（1/30），其余测试报 TODO not implemented
```

Windows 上不使用 WSL 时，用 `python tools/build.py` 和 `python tools/build.py test` 代替 `make` 和 `make test`（见[使用 Windows](#使用-windows)）。
未安装 Git 时，可在[课程仓库](https://github.com/ZeroGen-Lab/ML-Assignment-2)页面选择 **Code → Download ZIP** 下载并解压。
以下所有命令均在作业根目录（包含 `src/` 的目录）下执行。

请按[建议顺序](#建议顺序)逐项实现，并用 `make test`、`make check` 在本机自测；自测通过后，使用 `python3 tools/submit.py submit --id 学号 --name 姓名` 提交（见[提交](#提交)）。

## 学习目标

- 推导并实现常用算子的反向传播：逐元素运算、矩阵乘法、卷积、池化、softmax 交叉熵。
- 理解计算图：前向计算时记录节点，反向传播时按拓扑逆序累加梯度。
- 理解网络层如何由算子组合而成，以及 LSTM 沿时间展开后如何成为一张普通的计算图。
- 实现带 momentum 的 SGD，理解参数更新与梯度清零。
- 使用梯度检查（比较数值梯度与解析梯度）定位实现错误。

## 框架结构

```
models/      MLP、CNN、LSTM 的结构和训练程序           （已给出）
  ↓ 调用
nn / optim   Linear、Conv2d、MaxPool2d、LSTMCell、SGD  （需实现 src/nn.cpp、src/optim.cpp）
  ↓ 由算子组合
ops          各算子的前向和反向                         （需实现 src/ops.cpp）
  ↓ 记录计算图
autograd     Tensor::backward()                         （需实现 src/autograd.cpp）
  ↓
tensor       存储、形状、梯度、计算图节点 Node           （已给出）
```

所有数据均为 `float`（float32），按行优先存储。接口定义在 `include/tinydl/*.hpp` 中，头文件不得修改。

## 模块分工

| 模块 | 内容 | 状态 | 文件 |
| --- | --- | --- | --- |
| Tensor | 存储、形状、梯度缓冲、计算图节点 `Node`、随机数 | 已给出 | `include/tinydl/tensor.hpp`、`src/tensor.cpp` |
| 基础算子 | `mul`、`matmul`、`add_bias`、`relu`、`sigmoid`、`tanh`、`reshape`、`slice_cols` 的前向和反向（`add` 已给出，作为示例） | 需实现 | `src/ops.cpp` |
| 卷积与池化 | `conv2d`、`max_pool2d` 的前向和反向 | 需实现 | `src/ops.cpp` |
| 损失 | `cross_entropy`（softmax 与负对数似然，对 batch 取平均）的前向和反向 | 需实现 | `src/ops.cpp` |
| Autograd | `Tensor::backward()`：拓扑排序后按逆序调用各节点的 `backward`；不得使用递归（测试包含 2 万层的深链，递归会导致栈溢出） | 需实现 | `src/autograd.cpp` |
| 网络层 | `Linear`、`Conv2d`、`MaxPool2d`、`LSTMCell` 的 `forward()`，由上述算子组合而成 | 需实现 | `src/nn.cpp` |
| 优化器 | `zero_grad`、带 momentum 的 `SGD`（`clip_grad_norm` 已给出） | 需实现 | `src/optim.cpp` |
| 模型与训练 | 三个模型的结构、超参数和训练循环 | 已给出 | `models/` |
| 测试 | 梯度检查与单元测试、固定种子一致性检查 | 已给出 | `tests/`、`tools/check_consistency.py` |

## 需实现的文件

评测时只使用 `src/autograd.cpp`、`src/ops.cpp`、`src/nn.cpp`、`src/optim.cpp` 四个文件，其余文件一律替换为课程提供的版本。

待实现的位置均以 `todo("...")` 标出，调用时抛出异常。各函数的形状约定和语义见 `include/tinydl/*.hpp` 及源文件中的注释。
允许在这四个文件中添加辅助函数，或为 `Node` 的子类添加成员；不得修改头文件。

## 三个模型

模型结构定义在 `models/nets.hpp`，训练程序位于 `models/train_*.cpp`。

| 模型 | 结构 | 评分设置（一致性检查，见[评分](#评分)） |
| --- | --- | --- |
| MLP | Linear(784→256) → ReLU → Linear(256→128) → ReLU → Linear(128→10) | 每步 batch 32，每个样本 784 维（28×28）；lr 0.05，momentum 0.9 |
| CNN | Conv(3→16, 3×3, padding 1) → ReLU → MaxPool(2×2) → Conv(16→32, 3×3, padding 1) → ReLU → MaxPool(2×2) → Linear(2048→128) → ReLU → Linear(128→10) | 每步 batch 8，每个样本 3×32×32；lr 0.01，momentum 0.9 |
| LSTM | LSTMCell(9→64) 沿时间展开，取最后一步的隐藏状态 → Linear(64→6) | 每步 batch 32，每个样本 16 个时间步、每步 9 维；lr 0.1，momentum 0.9，梯度裁剪至范数 5 |

评分所用的输入由固定随机种子生成，无需数据集。lr、momentum 和梯度裁剪的设置在评分与真实数据训练中相同。

### 真实数据训练（可选，不计分）

如需用真实数据验证框架的训练效果，可先准备数据（见[数据](#数据)），再运行训练程序：

| 模型 | 数据集 | 训练设置 |
| --- | --- | --- |
| MLP | Fashion-MNIST | 3 epoch，batch 64 |
| CNN | CIFAR-10（仅用训练集前 20000 张） | 2 epoch，batch 32 |
| LSTM | UCI HAR（每个样本 128 个时间步） | 4 epoch，batch 32 |

其中 epoch 指完整遍历一次训练集，batch 指每次参数更新所用的样本数。

## 数据

**完成作业和评分均无需下载数据集。** 本节仅适用于真实数据训练。

从各数据集的官方地址下载原始文件，放入同一目录（如 `raw/`）并按以下名称命名，再转换为 `data/`：

- Fashion-MNIST：四个文件 `train-images-idx3-ubyte.gz` 等，分别改名为 `fashion-train-images-idx3-ubyte.gz`、`fashion-train-labels-idx1-ubyte.gz`、`fashion-t10k-images-idx3-ubyte.gz`、`fashion-t10k-labels-idx1-ubyte.gz`
- CIFAR-10：二进制版本 `cifar-10-binary.tar.gz`
- UCI HAR：压缩包，改名为 `uci-har.zip`

```
python3 tools/prepare_data.py raw data
```

缺少某个数据集的文件时，转换脚本会跳过该数据集；只训练一个模型时，仅需准备对应的数据。`data/` 已列入 `.gitignore`，请勿提交。

## 编译和自测

环境要求：支持 C++17 的 `g++` 或 `clang++`、`make` 和 `python3`。Linux 和 macOS 可直接使用；Windows 可直接编译（不需要 `make`），也可使用 WSL，见[使用 Windows](#使用-windows)。

```
make            # 生成 build/gradcheck、build/consistency 和三个训练程序
make test       # 梯度检查和单元测试（公开部分）
make check      # 固定种子一致性检查（公开种子）
./build/gradcheck ops          # 只运行一组：ops / autograd / nn / optim
./build/gradcheck matmul       # 只运行一项
./build/consistency mlp 1      # 输出一次一致性检查的原始结果（参数：mlp / cnn / lstm，种子）
./build/train_mlp  --data data # 可选：用真实数据训练 MLP（需先准备 data/）
./build/train_cnn  --data data # 可选：训练 CNN
./build/train_lstm --data data # 可选：训练 LSTM
```

没有 `make` 时（如 Windows），可用 `tools/build.py` 完成同样的操作：

```
python tools/build.py              # 等同 make
python tools/build.py test         # 等同 make test
python tools/build.py test ops     # 等同 ./build/gradcheck ops
python tools/build.py check        # 等同 make check
python tools/build.py clean        # 删除 build/
```

Windows 下直接运行程序的写法为 `build\consistency mlp 1`、`build\train_mlp --data data`。

训练程序支持以下参数：`--epochs N`（训练轮数）、`--limit N`（仅使用前 N 个训练样本，便于快速调试）、`--seed N`（随机种子）。
训练时每个 epoch 输出一行 loss，结束时输出一行 JSON，包含训练耗时、测试准确率和最终 loss。

## 建议顺序

1. `mul`、`relu`、`sigmoid`、`tanh`：逐个通过 `./build/gradcheck ops`。
2. `Tensor::backward()`：通过 `./build/gradcheck autograd`。
3. `matmul`、`add_bias`、`reshape`、`cross_entropy`，`Linear::forward`，`zero_grad`、`SGD` → `make check` 中 MLP 达到“计算准确”。
4. `conv2d`、`max_pool2d`，`Conv2d::forward`、`MaxPool2d::forward` → CNN 达到“计算准确”。
5. `slice_cols`、`LSTMCell::forward` → LSTM 达到“计算准确”。

每一步应先通过对应的梯度检查，再进行一致性检查。三个模型分别计 credit，按上述顺序每完成一个模型，即可获得该模型的 credit。

## 测试与反馈

`make test` 共 30 个测试，分为四组，分别对应四个需实现的文件。
大部分为梯度检查：对输入的每个元素分别加、减一个微小量，由输出的变化得到数值梯度，再与 backward 计算的解析梯度比较；其余测试使用可手算的小规模例子核对数值或行为。

| 组 | 对应文件 | 检查内容 |
| --- | --- | --- |
| ops（16 个） | `src/ops.cpp` | 各算子的梯度；交叉熵在 logit 很大时的数值稳定性；卷积和池化的数值与梯度（含步长 2、重叠窗口） |
| autograd（6 个） | `src/autograd.cpp` | 同一变量多次使用时的梯度累加；分叉后汇合的计算图；2 万层深链（不得栈溢出）；连续两次 backward 的梯度累加；`NoGradGuard`；不需要梯度的张量不接收梯度 |
| nn（4 个） | `src/nn.cpp` | `Linear`、Conv→ReLU→Pool、`LSTMCell` 的梯度；`LSTMCell` 的一组手算数值 |
| optim（4 个） | `src/optim.cpp` | `zero_grad`；有、无 momentum 的 SGD 与手算结果一致；没有梯度的参数不更新 |

评测服务器上另有 25 个隐藏测试，分组相同，使用更大或更极端的形状与数值。

各工具提供的反馈：

- `make test`：失败时输出 `FAIL 组/测试名: input k elem i: analytic … numeric …`，指明梯度不一致的输入及元素位置。
- `make check`：输出每个模型、每个种子的误差和档位；未达到“计算准确”时，另外输出误差最大的比较项，例如 `第 1 步后的梯度 lstm.Wih` 或 `第 5 步后的参数 fc2.b`。
  若第 1 步后的梯度均正确、仅第 5 步后的参数存在误差，问题通常在优化器；若第 1 步后某层的梯度存在误差，应检查该层使用的算子。
- `status`：总 credit，三个模型各自的 credit、隐藏种子在各档位的组数和误差；公开种子的误差（不计分）；隐藏种子运行失败的类型（如超时、某个 TODO 未实现）；未通过的公开测试名称；隐藏测试各组的通过数（如 `autograd 4/5`，不显示测试名称）。

测试结果仅用于定位错误，不计入成绩。

## 评分

仅依据固定种子一致性检查的**误差**评分，不考虑训练准确率和运行速度。每个模型的每组隐藏种子**逐组定档**，三个模型**分别计 credit**。

- **检查方法**：每个模型使用 6 组固定随机种子。种子 1 公开，即本机 `make check` 所用的种子，仅供自查，**不计分**；其余 5 组仅在评测服务器上运行，用于评分。
  每组种子生成网络的初始权重和随机输入，连续执行 5 步 forward / backward / SGD（momentum），每步一个 batch（大小见[三个模型](#三个模型)），并与标准答案比较：每一步的 loss、第 1 步后各参数的梯度、最后一步后的各参数。
  标准答案由 PyTorch 以 float32 执行相同的 5 步得到。
- **相对误差**：每个比较值按 `|提交值 − 标准值| ÷ 尺度` 计算。loss 的尺度为标准值的绝对值；梯度和参数的尺度为对应标准张量的均方根，以避免接近 0 的元素放大误差。
  某组运行崩溃、超时或输出形状不符时，该组误差记为无穷大（显示为“—”）。
- **定档**：每个模型的 5 组隐藏种子**逐组定档**。每组取该组全部比较值的**最大误差**和**误差中位数**，按下表确定档位和分值：

| 档位 | 条件（单组种子） | 含义 | 分值 |
| --- | --- | --- | --- |
| 计算准确 | 最大误差 ≤ 1e-5 且中位数 ≤ 1e-6 | 实现正确，差异仅来自浮点运算顺序 | 4 |
| 低误差 | 未达到计算准确，且最大误差 ≤ 1e-3 | 实现基本正确，但存在数值精度问题 | 3 |
| 较高误差 | 最大误差 > 1e-3 | 存在较大数值精度问题 | 2 |
| 计算错误 | 该组运行未给出有效结果 | 有函数未实现，或运行崩溃、超时 | 1 |

正确的 float32 实现与标准答案之间的差异仅来自运算顺序，实测最大约为 3e-6，低于“计算准确”的界限，无需追求零误差。

- **credit**：每个模型的 credit 为其 5 组种子分值之和，最高 20；总 credit 为三个模型之和，最高 60。
  例如 MLP 的 5 组中 4 组计算准确、1 组低误差，则 MLP 的 credit 为 4 × 4 + 3 = 19。
  排行榜显示总 credit、每个模型的 5 组种子在四个档位中的分布，以及全部 15 组隐藏种子中的最大误差。
- **作业成绩**：由三个模型的评测结果确定，具体换算方式另行公布。
  各模型相互独立。例如 MLP、CNN 全部计算准确而 LSTM 未实现时，MLP 和 CNN 仍各获得 20 credit。
- **排名**：首先比较总 credit，高者在前；相同时比较全部 15 组隐藏种子（三个模型各 5 组）中的最大误差，小者在前；仍相同时，提交时间早者在前。
  排行榜采用每人截止前排名最高的一次提交，作业成绩以该次提交为准。
- 梯度检查和单元测试不计分，评测时照常运行并显示通过数（见[测试与反馈](#测试与反馈)）。

## 提交

在作业根目录下运行以下命令。提交时仅上传 `src/` 中的四个文件。

```
python3 tools/submit.py submit --id 学号 --name 姓名 --nickname 昵称   # 提交（--nickname 可省略）
python3 tools/submit.py status --id 学号                               # 查询已用次数、排队位置和评测结果
```

提交时须用 `--name` 填写本人姓名，姓名仅用于登记成绩，不在排行榜显示。

**每人共 3 次提交机会**，请在本机通过 `make test` 和 `make check` 后再提交。`submit` 在上传前依次执行：

1. 在本机编译并运行公开种子的一致性检查（与 `make check` 相同）并显示结果；本机编译失败时不上传。
2. 显示学号、姓名和本次将使用的提交次数，输入 `yes` 后才上传。
3. 学号首次登记时，要求再次输入学号确认。成绩按学号记录，登记后不可自行修改，请务必核对。
   如发现学号填写错误，请立即联系助教，并修改保存了错误学号的脚本、命令历史或环境变量。

以下情况**不计提交次数**：

- 服务器代码检查未通过，或编译、链接失败（不产生结果，修改后可重新提交）；
- 上传的代码与此前某次提交完全相同（例如网络中断后重试），服务器直接关联到该次提交；
- 排队期间提交了新版本，排队中的旧提交被替换；
- 评测系统故障；
- 截止后、次数用尽后或因网络问题未显示“已受理”的上传。

排行榜显示学号和昵称，不显示姓名。昵称默认为“匿名同学”，最多 20 个字符，每次提交时均可修改。
排行榜中昵称为“参考实现”的条目是课程参考答案的评测结果，仅供对照，不参与成绩评定。
`status` 显示每次提交的结果和诊断信息（见[测试与反馈](#测试与反馈)）。

截止时间为 **2026-12-03 23:59:59（北京时间）**，以服务器受理上传的时间为准；截止前已受理、仍在排队的提交照常评测。
排行榜：https://coideas.me/ml-assignment-2/

## 规则

- 仅允许使用 C++17 标准库，单线程。评测编译选项为 `-O3 -march=native -std=c++17`，评测服务器为 x86-64 架构，支持 AVX-512。
- 时间与内存限制：每次一致性运行和每个测试的时限均为 5 秒，超时的运行误差记为无穷大；编译和链接总时限为 180 秒；每次运行内存上限为 2 GB。
  因服务器原因（例如负载过高导致编译超时）未能完成评测的提交记为系统故障，不计次数，可重新提交相同代码。
- 源码检查：`#include` 仅允许常用标准头文件和 `tinydl/*.hpp`，且须直接写出头文件名；禁止使用输出、文件、环境变量、进程、线程、内联汇编等功能（如 `printf`、`cout`、`fopen`、`getenv`、`system`、`exit`、`thread`、`asm`、`extern`、`remove`、`rename`、`__attribute__`），也禁止使用 `#pragma`、`##`、反斜杠续行、原始字符串 `R"..."` 和 `%:`。
  编译后还将检查目标文件引用的外部符号，仅允许标准库自身所需的符号。未通过检查的提交不产生结果，也不计次数。
- 程序不得读写评测所需以外的文件，不得访问网络。

## 常见问题

### `make` 提示不支持 `-march=native`

部分编译器（如较旧版本的 Apple clang）不支持该选项，本机编译时可去掉：`make CXXFLAGS="-O3 -std=c++17 -Wall"`。`tools/build.py` 会自动检测，无需处理。评测服务器的编译选项不受影响。

### 使用 Windows

以下两种方式任选其一。

**方式一：直接在 Windows 上编译（不需要 `make`）**

1. 安装 [Python 3](https://www.python.org/downloads/)，安装时勾选 “Add python.exe to PATH”。
2. 安装 MinGW-w64 版 `g++`（9 或更新版本）。推荐 [MSYS2](https://www.msys2.org/)：安装后打开 “MSYS2 UCRT64” 终端，运行 `pacman -S --needed mingw-w64-ucrt-x86_64-gcc`，再把 `C:\msys64\ucrt64\bin` 加入系统环境变量 PATH。已安装其他 MinGW-w64 `g++` 的同学可直接使用。不支持 Visual Studio 的 MSVC 编译器。
3. 重新打开 PowerShell 或命令提示符，运行 `g++ --version` 和 `python --version` 确认均可用。
4. 在作业目录下用 `python tools/build.py`、`python tools/build.py test`、`python tools/build.py check` 编译和自测；本文档中的 `python3` 均改为 `python`，例如 `python tools/submit.py submit --id 学号 --name 姓名`。

**方式二：使用 WSL（Ubuntu）**

与评测服务器的环境一致。以管理员身份打开 PowerShell，运行 `wsl --install` 并重启；打开 Ubuntu 后运行 `sudo apt update && sudo apt install -y g++ make python3 git`，之后按上述命令操作。建议将作业放在 WSL 的主目录（`~`）下，不要放在 `/mnt/c` 下，否则编译较慢。

两种方式的本机结果与服务器可能有微小差异，均以服务器评测为准。

### 本机 `make check` 误差很小，服务器上的误差较大

请先通过 `status` 查看具体原因：

- 代码检查未通过：使用了[规则](#规则)中禁止的头文件或函数（不计次数）；
- 隐藏种子误差大或隐藏测试失败：实现可能仅在公开种子的情形下正确，请检查广播、非方阵、batch 为 1、步长和边界等情况；
- 超时：检查 `Tensor::backward()` 是否存在重复遍历或平方复杂度的查找；
- 结果不稳定：检查未初始化的变量和数组越界。服务器以 `-O3 -march=native` 编译，此类错误更容易暴露。

### 提交时提示网络错误或“评测服务器暂时不可用”

请稍后重试；使用代理或校外网络时，可尝试更换网络。提示“请求太频繁”时，请等待一分钟后重试。
不确定上次是否上传成功时，请先用 `status` 查询；重新提交相同代码不会重复计次。

### 提交后多久出结果

通常在一分钟内完成。`submit` 会显示排队位置和预计等待时间，结果可用 `status` 查询。临近截止时提交较多，请尽早提交。

### 其他问题

请联系课程助教，并提供提交编号、`status` 的输出和简要的复现步骤。请勿在公开场合发布完整代码。
