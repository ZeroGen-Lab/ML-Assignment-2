# ML Assignment 2 · C++ Tiny Deep Learning Framework

[排行榜](https://coideas.me/ml-assignment-2/) · [课程仓库](https://github.com/ZeroGen-Lab/ML-Assignment-2)

用 C++17（只用标准库）实现一个小型深度学习框架，并用它训练 MLP、CNN、LSTM 三个模型。

这是一份填空式作业：张量、模型结构、数据加载、训练循环都已写好，你负责框架的核心——
各算子的前向和反向传播、自动求导引擎、网络层和优化器。
三个模型的梯度（包括 LSTM 沿 128 个时间步的反向传播）都由你的 autograd 自动求出，不为单个模型手写梯度。

占课程总评的 35%，三个模型按误差分别定档、给 credit（见[评分](#评分)）。每人共 **3 次**提交机会。
**提交截止：2026 年 12 月 3 日 23:59:59（北京时间）。**

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
`make check` 误差足够小后再用 `python3 tools/submit.py submit --id 你的学号` 提交（见[提交](#提交)）。

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

只有 `src/autograd.cpp`、`src/ops.cpp`、`src/nn.cpp`、`src/optim.cpp` 四个文件会被评测，其余文件评测时一律使用课程提供的版本。

每处待实现的位置都是 `todo("...")`，调用时抛异常。各函数的形状和语义见 `include/tinydl/*.hpp` 和源文件里的注释。
可以在这四个文件里加辅助函数、给 Node 子类加成员，但不要改头文件。

## 三个模型

结构定义在 `models/nets.hpp`，训练程序在 `models/train_*.cpp`。

| 模型 | 数据集 | 结构 | 训练设置 |
| --- | --- | --- | --- |
| MLP | Fashion-MNIST | Linear(784→256) → ReLU → Linear(256→128) → ReLU → Linear(128→10) | 3 epoch，batch 64，lr 0.05，momentum 0.9 |
| CNN | CIFAR-10 | Conv(3→16, 3×3) → ReLU → MaxPool → Conv(16→32, 3×3) → ReLU → MaxPool → Linear(2048→128) → ReLU → Linear(128→10) | 训练集前 20000 张，2 epoch，batch 32，lr 0.01，momentum 0.9 |
| LSTM | UCI HAR | LSTMCell(9→64) 展开 128 步，最后的隐藏状态 → Linear(64→6) | 4 epoch，batch 32，lr 0.1，momentum 0.9，梯度裁剪到范数 5 |

## 数据

**做作业和评分都不需要下载数据集。** 一致性检查的输入是用固定随机种子生成的，形状如下：

| 模型 | 一次检查的输入 | 输出类别 |
| --- | --- | --- |
| MLP | 32 个样本，每个 784 个数（28×28） | 10 |
| CNN | 8 个样本，每个 3×32×32 | 10 |
| LSTM | 32 个样本，每个 16 个时间步、每步 9 个数 | 6 |

（可选）想用真实数据完整训练三个模型、看看准确率，可以从各数据集官方地址下载原始数据，
放进一个目录（如 `raw/`）并按下面的名字命名，再转换成 `data/`：

- Fashion-MNIST 四个文件 `train-images-idx3-ubyte.gz` 等，改名为 `fashion-train-images-idx3-ubyte.gz`、`fashion-train-labels-idx1-ubyte.gz`、`fashion-t10k-images-idx3-ubyte.gz`、`fashion-t10k-labels-idx1-ubyte.gz`
- CIFAR-10 的二进制版本 `cifar-10-binary.tar.gz`
- UCI HAR 的压缩包，改名为 `uci-har.zip`

```
python3 tools/prepare_data.py raw data
```

缺哪个数据集的文件就跳过哪个，只想训练一个模型时只准备它的数据即可。`data/` 已写进 `.gitignore`，不要提交。

## 编译和自测

需要支持 C++17 的 `g++` 或 `clang++`、`make` 和 `python3`（Linux、macOS 直接可用；Windows 请在 WSL 里做作业）。

```
make            # 生成 build/gradcheck、build/consistency 和三个训练程序
make test       # 梯度检查和单元测试（公开部分）
make check      # 固定种子一致性检查（公开种子）
./build/gradcheck ops          # 只跑一组：ops / autograd / nn / optim
./build/gradcheck matmul       # 只跑一项
./build/consistency mlp 1      # 打印一次一致性检查的原始输出（mlp / cnn / lstm，种子）
./build/train_mlp  --data data # 可选：用真实数据训练 MLP（需要先准备 data/，见上）
./build/train_cnn  --data data # 可选：训练 CNN
./build/train_lstm --data data # 可选：训练 LSTM
```

训练程序可加 `--epochs N`（训练轮数）、`--limit N`（只用前 N 个训练样本）、`--seed N`（随机种子），
调试时用小的 `--limit` 可以很快跑完。训练过程中每个 epoch 输出一行 loss，最后一行输出 JSON：训练耗时、测试准确率、最终 loss。

## 建议顺序

1. `mul`、`relu`、`sigmoid`、`tanh`：`./build/gradcheck ops` 逐个通过。
2. `Tensor::backward()`：`./build/gradcheck autograd`。
3. `matmul`、`add_bias`、`reshape`、`cross_entropy`，`Linear::forward`，`zero_grad`、`SGD` → `make check` 里 MLP 达到“完全正确”。
4. `conv2d`、`max_pool2d`，`Conv2d::forward`、`MaxPool2d::forward` → CNN 达到“完全正确”。
5. `slice_cols`、`LSTMCell::forward` → LSTM 达到“完全正确”。

每一步先让对应的梯度检查通过，再看一致性检查。三个模型分别给 credit，按这个顺序做，做完一个就能拿到一个模型的 credit。

## 测试与反馈

`make test` 的 30 个测试分四组，每组对应一个你写的文件。多数是梯度检查：把输入的元素逐个加减一个很小的数，用输出的变化算出数值梯度，和你的 backward 算出的梯度比较；其余用能手算的小例子核对数值或行为。

| 组 | 对应文件 | 检查什么 |
| --- | --- | --- |
| ops（16 个） | `src/ops.cpp` | 每个算子的梯度；交叉熵在 logit 很大时不溢出；卷积和池化的数值与梯度（含步长 2、重叠窗口） |
| autograd（6 个） | `src/autograd.cpp` | 同一变量被多次使用时梯度累加；分叉再汇合的图；2 万层的深链不爆栈；连续两次 backward 梯度累加；`NoGradGuard`；不需要梯度的张量不接收梯度 |
| nn（4 个） | `src/nn.cpp` | `Linear`、Conv→ReLU→Pool、`LSTMCell` 的梯度；`LSTMCell` 的一个手算数值 |
| optim（4 个） | `src/optim.cpp` | `zero_grad`；有、无 momentum 的 SGD 与手算一致；没有梯度的参数不更新 |

服务器上另有 25 个隐藏测试，分组相同，换成更大或更极端的形状和数值。

各处给出的反馈：

- `make test`：失败时打印 `FAIL 组/测试名: input k elem i: analytic … numeric …`，指出哪个输入的哪个元素梯度不对。
- `make check`：每个模型、每个种子的误差和结果；没达到“完全正确”时，再打印误差最大的一项，如 `第 1 步后的梯度 lstm.Wih` 或 `第 5 步后的参数 fc2.b`。
  第 1 步后的梯度都对、只有第 5 步后的参数有误差，多半是优化器的问题；第 1 步后某层的梯度有误差，查这一层用到的算子。
- `status`：三个模型各自的档位和误差（只算隐藏种子），以及合起来的总结果，公开种子的误差（不计分），隐藏种子运行出错的类型（如超时、某个 TODO 未实现），未通过的公开测试名，隐藏测试每组的通过数（如 `autograd 4/5`，不显示测试名）。

测试只帮你定位错误，不计分。

## 评分

只看固定种子一致性检查的**误差**，不看训练准确率，也不计训练速度。三个模型**分别定档、分别给 credit**。

- **怎么检查**：3 个模型（MLP、CNN、LSTM）各用 6 组固定随机种子。种子 1 公开，就是本机 `make check` 用的那一组，只供自查、**不计分**；
  评分只用另外 5 组，它们只在评测服务器上运行。
  每组用种子生成网络初始权重和随机输入，连续做 5 步 forward / backward / SGD（momentum），每步一个 batch（大小见[数据](#数据)），
  和标准答案比较：每一步的 loss、第 1 步后每个参数的梯度、最后一步后的参数。
  标准答案是用 PyTorch float32 跑同样 5 步得到的。
- **相对误差**：每个比较的数算 `|你的值 − 标准值| ÷ 尺度`，loss 的尺度是标准值的绝对值，梯度和参数的尺度是标准张量的均方根（避免接近 0 的元素把误差放大）。
  某组运行崩溃、超时或输出的形状不对，这组的误差记为无穷大（显示为“—”）。
- **定档**：每个模型取它在 5 组隐藏种子上全部比较值的**最大误差**和**误差中位数**，按下表定档，得到这个模型的 credit：

| 档位 | 条件 | 含义 | 该模型的 credit |
| --- | --- | --- | --- |
| 完全正确 | 最大误差 ≤ 1e-5 且中位数 ≤ 1e-6 | 实现正确，只剩浮点运算顺序带来的差别 | 100% |
| 大体正确 | 不是完全正确，但最大误差 ≤ 1e-3 | 基本正确，但某处数值精度处理不好 | 60% |
| 有错误 | 最大误差 > 1e-3 | 有算错的地方 | 30% |
| 未完成 | 有运行没给出可用结果 | 有函数没实现、崩溃或超时 | 0 |

正确的 float32 实现和标准答案的差别只来自运算顺序，实测最大约 3e-6，离“完全正确”的界限还有余量，不需要追求误差为 0。
- **作业成绩**：由三个模型各自的 credit 合成，三个模型所占的比例另行公布。
  三个模型互不影响：比如 MLP、CNN 完全正确而 LSTM 未完成，MLP、CNN 照样拿满各自的 credit。
- **排名**：先比三个模型 credit 之和，越高越靠前；相同时再比最大误差、误差中位数（三个模型合在一起算），越小越靠前；再相同时提交早的在前。
  榜单采用你截止前排名最靠前的一次提交，课程成绩在截止后按上面的规则确定。
- 梯度检查和单元测试不计分，评测时照常运行并显示通过数（见[测试与反馈](#测试与反馈)）。

## 提交

在作业目录下（含 `src/` 的那一层）运行，只上传 `src/` 里的四个文件：

```
python3 tools/submit.py submit --id 你的学号                   # 提交
python3 tools/submit.py submit --id 你的学号 --nickname 昵称   # 提交，同时设置排行榜昵称
python3 tools/submit.py status --id 你的学号                   # 查看已用次数、排队位置和结果
```

**每人共 3 次提交机会**，在本机把 `make test`、`make check` 调好再提交。
`submit` 上传前会：

1. 在本机编译并运行公开种子的一致性检查（和 `make check` 相同），显示结果；本机编译失败就不上传；
2. 显示学号和“这次将用掉第几次机会”，输入 `yes` 才上传；
3. 学号第一次登记时，要求再输入一遍学号确认。成绩按学号记，登记后不能自己修改，请务必填对；
   发现填错请立即联系助教，并改掉保存了错误学号的地方（脚本、命令历史、环境变量等）。

以下情况**不计次数**：

- 服务器上代码检查不通过，或编译、链接失败（不出结果，改好后再交）；
- 上传的代码和你之前的某次提交完全相同（例如网络中断后重试），服务器直接指向那次提交；
- 排队中又提交了新版本：排队中的旧提交被替换；
- 评测系统故障；
- 截止后、次数用完后或因网络问题没有显示“已受理”的上传。

排行榜只显示分配的编号（如 `S0001`）和昵称，不显示学号；昵称默认“匿名同学”，最多 20 字，每次提交都可以改。
`status` 显示每次提交的结果和诊断信息（见[测试与反馈](#测试与反馈)）。

截止时间 **2026-12-03 23:59:59（北京时间）**，以上传时间为准；截止前上传、仍在排队的提交照常评测。
排行榜：https://coideas.me/ml-assignment-2/

## 规则

- 只用 C++17 标准库，编译选项 `-O3 -march=native -std=c++17`（评测服务器为 x86-64，支持 AVX-512），单线程。
- 时限：每次一致性运行、每个测试各 5 秒，超时的运行误差记为无穷大；编译和链接共 180 秒。
  因服务器原因（例如繁忙导致编译超时）没能评完的提交记为系统故障，不计次数，可以重新提交同样的代码。
  每次运行内存上限 2 GB。
- 评测前会检查源码：`#include` 只允许常用标准头文件和 `tinydl/*.hpp`，而且要直接写头文件名；不得使用输出、文件、环境变量、进程、线程、内联汇编等（如 `printf`、`cout`、`fopen`、`getenv`、`system`、`exit`、`thread`、`asm`、`extern`、`remove`、`rename`、`__attribute__`），也不得使用 `#pragma`、`##`、反斜杠续行、原始字符串 `R"..."` 和 `%:`。编译后还会检查目标文件引用的外部函数，只允许标准库本身需要的那些。检查不通过的提交不出结果，也不计次数。
- 程序不得读写评测所需以外的文件、不得联网。

## 常见问题

**`make` 报不支持 `-march=native`。**
个别编译器（如较旧的 Apple clang）不认这个选项，本机可以去掉它：`make CXXFLAGS="-O3 -std=c++17 -Wall"`。评测服务器的编译选项不受影响。

**我用 Windows。**
请在 WSL（Ubuntu）里做作业，在 WSL 里安装 `g++`、`make`、`python3` 后按上面的命令操作。

**本机 `make check` 误差很小，服务器上误差却变大了。**
先用 `status` 看具体原因：
- 代码检查未通过：用到了[规则](#规则)里不允许的头文件或函数（不计次数）；
- 隐藏种子误差大或隐藏测试失败：多半是只在公开种子上碰巧正确，检查广播、非方阵、batch 为 1、步长和边界等情况；
- 超时：检查 `Tensor::backward()` 是否有重复遍历或平方级的查找（LSTM 的计算图有十几万个节点）；
- 结果不稳定：检查未初始化的变量和数组越界，服务器用 `-O3 -march=native` 编译，这类错误更容易暴露。

**提交时提示网络错误或“评测服务器暂时不可用”。**
稍等再试；用了代理或校外网络时，可以换个网络。提示“请求太频繁”时等一分钟再试。
不确定刚才是否已上传成功时，先用 `status` 查看；直接再提交同样的代码也不会重复计次。

**提交后多久出结果？**
一般一分钟以内。`submit` 会显示排队位置和预计时间，用 `status` 查询结果。临近截止排队的人多，请尽早提交。

**遇到其他问题。**
向课程助教提供提交编号、`status` 的输出和简要复现步骤；不要在公开场合贴出自己的完整代码。
