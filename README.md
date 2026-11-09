# ML Assignment 2 · C++ Tiny Deep Learning Framework

[排行榜](https://coideas.me/ml-assignment-2/) · [课程仓库](https://github.com/ZeroGen-Lab/ML-Assignment-2) · [数据下载（Releases）](https://github.com/ZeroGen-Lab/ML-Assignment-2/releases)

用 C++17（只用标准库）实现一个小型深度学习框架，并用它训练 MLP、CNN、LSTM 三个模型。

这是一份填空式作业：张量、模型结构、数据加载、训练循环都已写好，你负责框架的核心——
各算子的前向和反向传播、自动求导引擎、网络层和优化器。
写完后，三个模型的训练（包括 LSTM 沿 128 个时间步的反向传播）全部由你的 autograd 自动完成，
不需要为任何模型手写梯度。

满分 100 分，占课程总评的 35%。**提交截止：2026 年 12 月 3 日 23:59:59（北京时间）。**

## 快速开始

```bash
git clone https://github.com/ZeroGen-Lab/ML-Assignment-2.git
cd ML-Assignment-2
make            # 第一次编译，确认编译器可用
make test       # 刚开始只有示例 ops/add 通过（1/30），其余报 TODO not implemented，这是正常的
```

不使用 Git 时，打开[课程仓库](https://github.com/ZeroGen-Lab/ML-Assignment-2)，点击 **Code → Download ZIP**，解压后进入目录。
之后的所有命令都在这个目录（含 `src/` 的那一层）里执行。

然后按[建议顺序](#建议顺序)逐个实现，用 `make test`、`make check` 在本机自测，
满意后用 `python3 tools/submit.py submit --id 你的学号` 提交（见[提交](#提交)）。

## 学习目标

- 推导并实现常用算子的反向传播：逐元素运算、矩阵乘法、卷积、池化、softmax 交叉熵。
- 理解计算图：前向时记录节点，反向时按拓扑逆序累加梯度。
- 理解网络层如何由算子组合而成，LSTM 的时间展开如何变成一张普通的计算图。
- 实现带 momentum 的 SGD，理解参数更新和梯度清零。
- 用梯度检查（数值梯度对比解析梯度）定位自己的错误。

## 框架结构

```
models/      MLP、CNN、LSTM 的结构和训练程序           （已给出）
  ↓ 调用
nn / optim   Linear、Conv2d、MaxPool2d、LSTMCell、SGD  （你写 src/nn.cpp、src/optim.cpp）
  ↓ 由算子组合
ops          各算子的前向和反向                         （你写 src/ops.cpp）
  ↓ 记录计算图
autograd     Tensor::backward()                         （你写 src/autograd.cpp）
  ↓
tensor       存储、形状、梯度、计算图节点 Node           （已给出）
```

所有数据都是 `float`（float32），按行优先存储。接口定义在 `include/tinydl/*.hpp`，头文件不能修改。

## 模块分工

| 模块 | 内容 | 谁写 | 文件 |
| --- | --- | --- | --- |
| Tensor | 存储、形状、梯度缓冲、计算图节点 `Node`、随机数 | 已给出 | `include/tinydl/tensor.hpp`、`src/tensor.cpp` |
| 基础算子 | `mul`、`matmul`、`add_bias`、`relu`、`sigmoid`、`tanh`、`reshape`、`slice_cols` 的前向和反向（`add` 已给出作示例） | 你 | `src/ops.cpp` |
| 卷积与池化 | `conv2d`、`max_pool2d` 的前向和反向 | 你 | `src/ops.cpp` |
| 损失 | `cross_entropy`（softmax + 负对数似然，取 batch 平均）的前向和反向 | 你 | `src/ops.cpp` |
| Autograd | `Tensor::backward()`：拓扑排序后按逆序调用各节点的 `backward`；不能用递归（LSTM 的计算图有十几万个节点） | 你 | `src/autograd.cpp` |
| 网络层 | `Linear`、`Conv2d`、`MaxPool2d`、`LSTMCell` 的 `forward()`，用上面的算子组合 | 你 | `src/nn.cpp` |
| 优化器 | `zero_grad`、带 momentum 的 `SGD`（`clip_grad_norm` 已给出） | 你 | `src/optim.cpp` |
| 模型与训练 | 三个模型的结构、学习率和训练循环 | 已给出 | `models/` |
| 测试 | 梯度检查与单元测试、固定种子一致性检查 | 已给出 | `tests/`、`tools/check_consistency.py` |

## 你要写的文件

只有这四个文件会被评测，其余文件评测时一律使用课程提供的版本：

| 文件 | 内容 |
| --- | --- |
| `src/autograd.cpp` | `Tensor::backward()`：拓扑排序 + 反向遍历计算图 |
| `src/ops.cpp` | 各算子的前向和 `Node::backward`（`add` 已给出作示例） |
| `src/nn.cpp` | `Linear` / `Conv2d` / `MaxPool2d` / `LSTMCell` 的 `forward()` |
| `src/optim.cpp` | `zero_grad`、`SGD`（momentum） |

每处待实现的位置都是 `todo("...")`，调用时抛异常。各函数的形状和语义见 `include/tinydl/*.hpp` 和源文件里的注释。
可以在这四个文件里加辅助函数、给 Node 子类加成员，但不要改头文件。

## 三个模型

结构定义在 `models/nets.hpp`，训练程序在 `models/train_*.cpp`。

| 模型 | 数据集 | 结构 | 训练设置 |
| --- | --- | --- | --- |
| MLP | Fashion-MNIST | Linear(784→256) → ReLU → Linear(256→128) → ReLU → Linear(128→10) | 3 epoch，batch 64，lr 0.05，momentum 0.9 |
| CNN | CIFAR-10 | Conv(3→16, 3×3) → ReLU → MaxPool → Conv(16→32, 3×3) → ReLU → MaxPool → Linear(2048→128) → ReLU → Linear(128→10) | 训练集前 20000 张，2 epoch，batch 32，lr 0.01，momentum 0.9 |
| LSTM | UCI HAR | LSTMCell(9→64) 展开 128 步，最后的隐藏状态 → Linear(64→6) | 4 epoch，batch 32，lr 0.1，momentum 0.9，梯度裁剪到范数 5 |

正确实现在一台 Mac 笔记本上（单线程）用默认设置训练的结果，供自查，不计分：

| 模型 | 测试准确率 | 训练耗时 |
| --- | --- | --- |
| MLP | 0.864 | 7 秒 |
| CNN | 0.588 | 38 秒 |
| LSTM | 0.886 | 34 秒 |

## 数据（可选）

评分不用数据集；数据只用来在本机真正训练三个模型。
数据不在仓库里（文件太大），压缩包 `tinydl-data-v1.zip`（225 MB）拆成了 5 个分卷，
请到本仓库的 [Releases](https://github.com/ZeroGen-Lab/ML-Assignment-2/releases) 页面下载全部 5 个文件
`tinydl-data-v1.zip.part-aa` 到 `tinydl-data-v1.zip.part-ae`（某个下载中断时只需重下那一个），
放在作业目录下（含 `src/` 的那一层），合并、校验、解压，得到 `data/` 目录：

```
cat tinydl-data-v1.zip.part-* > tinydl-data-v1.zip
shasum -a 256 tinydl-data-v1.zip   # Linux / WSL 用 sha256sum
# 应为 9295c3353f5267ef1410fc82250752fdfe570c6a75706f2e062e4073d2c098fe
unzip tinydl-data-v1.zip
rm tinydl-data-v1.zip tinydl-data-v1.zip.part-*   # 解压后可以删掉
```

校验值不一致说明某个分卷没下完整，重新下载后再合并。

`data/` 里是 Fashion-MNIST、CIFAR-10、UCI HAR 三个公开数据集的训练集和测试集；
来源和许可见 `data/DATA-SOURCES.md`。`data/` 已写进 `.gitignore`，不要提交。

无法从 Releases 下载时，从各数据集官方地址下载原始数据，放进一个目录（如 `raw/`）并按下面的名字命名，再转换：

- Fashion-MNIST 四个文件 `train-images-idx3-ubyte.gz` 等，改名为 `fashion-train-images-idx3-ubyte.gz`、`fashion-train-labels-idx1-ubyte.gz`、`fashion-t10k-images-idx3-ubyte.gz`、`fashion-t10k-labels-idx1-ubyte.gz`
- CIFAR-10 的二进制版本 `cifar-10-binary.tar.gz`
- UCI HAR 的压缩包，改名为 `uci-har.zip`

```
python3 tools/prepare_data.py raw data
```

## 编译和自测

需要支持 C++17 的 `g++` 或 `clang++`、`make` 和 `python3`（Linux、macOS 直接可用；Windows 请在 WSL 里做作业）。

```
make            # 生成 build/gradcheck、build/consistency 和三个训练程序
make test       # 梯度检查和单元测试（公开部分）
make check      # 固定种子一致性检查（公开种子）
./build/gradcheck ops          # 只跑一组：ops / autograd / nn / optim
./build/gradcheck matmul       # 只跑一项
./build/consistency mlp 1      # 打印一次一致性检查的原始输出（mlp / cnn / lstm，种子）
./build/train_mlp  --data data # 训练 MLP（Fashion-MNIST）
./build/train_cnn  --data data # 训练 CNN（CIFAR-10）
./build/train_lstm --data data # 训练 LSTM（UCI HAR）
```

训练程序可加 `--epochs N`（训练轮数）、`--limit N`（只用前 N 个训练样本）、`--seed N`（随机种子），
调试时用小的 `--limit` 可以很快跑完。训练过程中每个 epoch 输出一行 loss，最后一行输出 JSON：训练耗时、测试准确率、最终 loss。

## 建议顺序

1. `mul`、`relu`、`sigmoid`、`tanh`：`./build/gradcheck ops` 逐个通过。
2. `Tensor::backward()`：`./build/gradcheck autograd`。
3. `matmul`、`add_bias`、`reshape`、`cross_entropy`，`Linear::forward`，`zero_grad`、`SGD` → `make check` 里 mlp 通过。
4. `conv2d`、`max_pool2d`，`Conv2d::forward`、`MaxPool2d::forward` → cnn 通过。
5. `slice_cols`、`LSTMCell::forward` → lstm 通过。

每一步先让对应的梯度检查通过，再看一致性检查。梯度检查失败时打印 `FAIL 组/测试名: input k elem i: analytic … numeric …`，指出哪个输入的哪个元素梯度不对。

## 评分

总分 100，不看训练准确率，也不计训练速度：

- **梯度检查与单元测试 50 分**：`make test` 的 30 个公开测试，加上评测服务器上的隐藏测试（覆盖同样的函数，换不同的形状和数值，并包含更深的计算图），
  按全部测试的通过比例给分。
- **固定种子一致性检查 50 分**（MLP、CNN、LSTM 各三分之一）：用固定种子初始化网络，喂随机生成的小批量输入
  （MLP、LSTM batch 8，CNN batch 4，LSTM 序列长 16），连续做 5 步 forward / backward / SGD，
  和正确实现的结果比较三部分：第 1 步的 loss（forward）、第 1 步后每个参数的梯度（backward）、之后各步的 loss 和最终参数（training）。
  每个数值的允许误差为 `|你的值 − 参考值| ≤ 1e-3 × |参考值| + 1e-3 × scale`，loss 的 scale 为 1，张量的 scale 为参考张量的均方根；
  这个容差远大于求和顺序、编译选项带来的浮点差异，实现正确就不会失分。
  评测服务器运行公开种子和另一批不公开的种子，按通过的部分数给分；本机 `make check` 用公开种子，检查方法完全相同。

梯度检查和一致性检查分开计分：只完成了部分算子，也能拿到对应的分数。

## 提交

在作业目录下（含 `src/` 的那一层）运行，只上传 `src/` 里的四个文件：

```
python3 tools/submit.py submit --id 你的学号                   # 提交并排队评测
python3 tools/submit.py submit --id 你的学号 --nickname 昵称   # 提交，同时设置排行榜昵称
python3 tools/submit.py status --id 你的学号                   # 查看排队位置和最近一次的成绩
python3 tools/submit.py status --id 你的学号 -n 5              # 查看最近 5 次
```

第一次提交时自动登记学号，请务必填对（4 到 32 位字母或数字，成绩按学号记）。
学号打错会登记成另一个学号，成绩也记在那个学号下，自己无法修改；发现填错请立即联系助教，并改掉保存了错误学号的地方（脚本、命令历史、环境变量等）。
排行榜只显示分配的编号（如 `S0001`）和昵称，不显示学号；昵称默认“匿名同学”，最多 20 字，每次提交都可以改。

`status` 显示：代码检查和编译结果、公开测试和隐藏测试的通过数（隐藏测试按 ops / autograd / nn / optim 分组给出）、
未通过的公开测试名、三个模型一致性检查各部分的通过数和未通过的公开种子、各项得分和总分。

排队与次数：

- 每人每天最多 20 次评测，北京时间 0 点重置。
- 排队按当天已开始评测的次数轮转，次数少的先评测。
- 排队中再次提交，会替换排队中的旧提交；被替换的提交和评测系统故障不计次数。
- 截止时间 **2026-12-03 23:59:59（北京时间）**，以上传时间为准；截止前上传、仍在排队的提交照常评测。
- 课程成绩取截止前得分最高的一次提交。排行榜按总分排名，同分时提交早的在前。

排行榜：https://coideas.me/ml-assignment-2/ 。平时请先用 `make test` 和 `make check` 在本机调试。

## 规则

- 只用 C++17 标准库，编译选项 `-O3 -march=native -std=c++17`（评测服务器为 x86-64，支持 AVX-512），单线程。
- 时限：每个测试、每次一致性运行各 5 秒，每份提交总共 60 秒（含编译），超时的项记为不通过。
  每次运行内存上限 2 GB。
- 评测前会检查源码：`#include` 只允许常用标准头文件和 `tinydl/*.hpp`，而且要直接写头文件名；不得使用输出、文件、环境变量、进程、线程、内联汇编等（如 `printf`、`cout`、`fopen`、`getenv`、`system`、`exit`、`thread`、`asm`、`extern`、`remove`、`rename`、`__attribute__`），也不得使用 `#pragma`、`##`、反斜杠续行、原始字符串 `R"..."` 和 `%:`。编译后还会检查目标文件引用的外部函数，只允许标准库本身需要的那些。检查不通过记 0 分。
- 程序不得读写评测所需以外的文件、不得联网。

## 常见问题

**`make` 报不支持 `-march=native`。**
个别编译器（如较旧的 Apple clang）不认这个选项，本机可以去掉它：`make CXXFLAGS="-O3 -std=c++17 -Wall"`。评测服务器的编译选项不受影响。

**我用 Windows。**
请在 WSL（Ubuntu）里做作业，在 WSL 里安装 `g++`、`make`、`python3` 后按上面的命令操作。

**本机 `make test`、`make check` 都通过了，服务器上却有失败。**
先用 `status` 看具体原因：
- 代码检查未通过：用到了[规则](#规则)里不允许的头文件或函数；
- 隐藏测试失败：多半是只在公开形状上碰巧正确，检查广播、非方阵、batch 为 1、步长和边界等情况；
- 超时：检查 `Tensor::backward()` 是否有重复遍历或平方级的查找（LSTM 的计算图有十几万个节点）；
- 结果不稳定：检查未初始化的变量和数组越界，服务器用 `-O3 -march=native` 编译，这类错误更容易暴露。

**提交时提示网络错误或“评测服务器暂时不可用”。**
稍等再试；用了代理或校外网络时，可以换个网络。提示“请求太频繁”时等一分钟再试。没有显示“已受理”的上传不计次数，评测系统故障也不计次数。

**提交后多久出结果？**
`submit` 会显示排队位置和预计时间，用 `status` 查询结果。截止前提交的人多，请尽早提交，不要等到最后一天。

**遇到其他问题。**
向课程助教提供提交编号、`status` 的输出和简要复现步骤；不要在公开场合贴出自己的完整代码。
