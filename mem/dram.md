# DRAM 简化延迟模型说明

## 1. 背景

这个文档对应的 Python 模块是 `mem/dram_latency.py`。

目标不是复刻完整 DRAMsim3，而是做一个足够轻量、可直接嵌入 Python 仿真流程的 DRAM 延迟模型，只保留下面几类信息：

- 读延迟
- 写延迟
- 跨行访问导致的额外延迟
- 碎片化访问导致的额外延迟

不建模下面这些内容：

- controller 重排序
- bank-group 竞争
- refresh 暂停
- thermal / power 反馈
- HB-NPU 阵列内部流水线


## 2. DRAMsim3 与 HB-NPU-simulator 的差异

### 2.1 DRAMsim3 原版怎么建模

原版 `DRAMsim3` 是标准 JEDEC DRAM 事务模型：

1. 上层提交 `Transaction(addr, is_write)`
2. 每个 channel 的 controller 把事务转成普通 DRAM 命令
3. `BankState` 判断当前 bank 是 closed、open 还是 self-refresh
4. 如果是 row hit，直接发 `READ/WRITE`
5. 如果是 closed row，先发 `ACTIVATE`
6. 如果是 row conflict，先发 `PRECHARGE`，再 `ACTIVATE`
7. `Timing` 表维护命令间约束，最后在 `IssueCommand` 中结算读写完成时间

也就是说，原版重点是：

- bank 状态机
- 命令级 timing table
- controller 队列与调度


### 2.2 HB-NPU-simulator 对 DRAM 做了什么修改

`HB-NPU-simulator` 是在 DRAMsim3 基础上的分叉，但它改的不是单一一个 timing 参数，而是新增了一整条 PIM/HB 命令路径。

主要修改如下：

1. 新增命令类型：

- `LH_READ`
- `LH_READ_PRECHARGE`
- `GH_READ`
- `GH_READ_PRECHARGE`
- `PIM_WRITE`
- `PIM_WRITE_PRECHARGE`
- `PIM_ACTIVATE`

2. trace 输入扩展出 `PIM` 事务类型

3. `dram_system.cc` 中新增一个中心化上层调度器：

- 解析 workload 配置
- 解析 dataflow 配置
- 计算 tile 的 channel / bank / row / column
- 将同步生成的命令分发到每个 channel controller

4. 每个 channel controller 额外维护 3 条 PIM 专用队列：

- 权重读队列
- 输入读队列
- 输出写队列

5. `BankState` / `ChannelState` / `Timing` 为这些 PIM 命令补了状态和 timing 规则

6. 为了避免 refresh 打断 PIM，大约在 refresh 快到时会暂停 PIM 命令投递


### 2.3 HB-NPU-simulator 对 DRAM 读写的实际建模方式

它的读写已经不是单纯“像 CPU 一样发普通读写请求”。

对于 PIM 路径，模型大体是：

1. 由上层调度器按 tile 计算要访问的 row/column
2. 权重加载阶段用 `GH_READ`
3. 输入流阶段根据 dataflow 选 `GH_READ` 或 `LH_READ`
4. 输出写回阶段用 `PIM_WRITE`
5. 如果 bank 当前行没开，则先发 `PIM_ACTIVATE`
6. 如果是最后一个列访问，则发 `_PRECHARGE` 版本命令

这里最重要的一点是：

- `LH_READ` 和 `GH_READ` 在时序上非常接近
- 真正大的差别在于 HB-NPU 用了额外的专用命令路径和中心化调度
- 它还混入了 NPU 阵列内部等待计数，因此后者不是纯 DRAM 模型，而是 DRAM + PIM 阵列的组合模型


## 3. 为什么这里不直接复刻 HB-NPU-simulator

如果你的需求只是“给出读写延迟”，那么完整复刻 `HB-NPU-simulator` 的收益不高，复杂度却很高。

你真正关心的是：

- 首次访问一个 row 要多久
- 同 row 连续访问要多久
- 切换到新 row 会多出多少
- 地址碎片化以后会多出多少

因此这里采用一个更直接的近似：

- 保留 open-row / closed-row / row-conflict 三种情况
- 保留 burst 粒度
- 保留碎片化造成的重复列命令和非对齐损失
- 忽略 controller 的全局调度优化


## 4. 简化模型采用的参数

默认参数对齐 `refs/HB-NPU-simulator/configs/HBM2_8Gb_x128.ini`：

- `tRCD = 14`
- `tRP = 14`
- `tCL = 14`
- `tCWL = 4`
- `burst_cycles = 2`
- `bus_width = 64`
- `BL = 4`

由 `request_size_bytes = bus_width / 8 * BL` 可得：

- 单次 burst 传输粒度 = `64 / 8 * 4 = 32B`

由 HBM 配置和 DRAMsim3 中 HBM 的列换算方式可得：

- row buffer 大小约为 `2048B`

因此模块默认使用：

- `burst_bytes = 32`
- `row_buffer_bytes = 2048`
- `request_bytes = 32`

另外，当前模型还额外引入了一个校准项：

- `transaction_overhead = 2 cycles`

它不是 JEDEC 参数，而是为了让这个 Python 模型在当前 HBM2 配置下，与 `DRAMsim3` 的读请求回调周期对齐。


## 5. 模型公式

### 5.0 事务粒度

这里有一个非常重要的约定：

- 模块不会把一个很长的连续区间当成“一次超长 DRAM burst”
- 而是会先按 `request_bytes` 切成多个 DRAM 事务
- 然后每个事务再分别计算 row hit / row conflict / 跨行代价

这样做的原因是：

- `DRAMsim3` 的事务层本来就是固定请求粒度
- 如果把 `addr + length` 当成一次长 burst，会明显低估总延迟

例如在当前 HBM2 配置下：

- 一个 `32B` 连续读是 1 个事务
- 一个 `64B` 连续读会被拆成 2 个事务

### 5.1 连续读

若访问 `[addr, addr + size)` 连续区间，先按 `request_bytes` 切分。

对每个事务：

- row hit:
  `transaction_overhead + tCL + burst_cycles`
- closed row:
  `transaction_overhead + tRCD + tCL + burst_cycles`
- row conflict:
  `transaction_overhead + tRP + tRCD + tCL + burst_cycles`

总延迟为所有事务延迟之和。


### 5.2 连续写

同样先按 `request_bytes` 切分。

- row hit:
  `transaction_overhead + tCWL + burst_cycles`
- closed row:
  `transaction_overhead + tRCD + tCWL + burst_cycles`
- row conflict:
  `transaction_overhead + tRP + tRCD + tCWL + burst_cycles`


### 5.3 跨行额外延迟

如果一次连续区间跨过 row boundary，模块会把它拆成多个 row 内 chunk。

每次从旧 row 切到新 row 时，都会增加：

- `tRP + tRCD`

这个量会累计到：

- `AccessReport.cross_row_cycles`


### 5.4 碎片化额外延迟

如果访问不是一个连续区间，而是很多分散地址，那么每个碎片都会单独发起一次列访问。

碎片化的额外损失主要来自：

1. 重复列命令开销

- 读用 `tCL`
- 写用 `tCWL`

2. 非 burst 对齐损失

- 如果一个碎片大小不是 `burst_bytes` 的整数倍，会有未用满 burst 的浪费

3. 如果碎片落在不同 row

- 还会继续叠加 `tRP + tRCD`

这些额外量分别记录在：

- `fragmentation_cycles`
- `partial_burst_cycles`
- `cross_row_cycles`


## 6. Python 模块做了哪些修改

当前 `mem/dram_latency.py` 提供了下面这些对象：

- `DramTiming`
- `DramGeometry`
- `Fragment`
- `AccessReport`
- `SimpleDramLatencyModel`

相对最初版本，当前模块支持 3 种输入方式。

另外，相比第一版实现，当前版本又做了两个和 `DRAMsim3` 对齐的重要修改：

1. 增加了 `request_bytes`

- 连续区间会先被拆成固定 DRAM 事务
- 这样大块读写的总延迟不再被低估

2. 增加了 `transaction_overhead`

- 用来吸收 `DRAMsim3` 事务层的固定前后开销
- 在当前 HBM2 配置下，读请求能与 `DRAMsim3` 的回调周期对齐


### 6.1 方式一：连续区间

适合“非碎片化”的情况。

```python
from mem import SimpleDramLatencyModel

dram = SimpleDramLatencyModel.hb_npu_hbm2_defaults()
report = dram.read(0x1000, 256)
print(report.total_cycles)
```

语义：

- 从起始地址 `0x1000`
- 连续读取 `256B`


### 6.2 方式二：地址列表

适合“碎片化，但每个地址访问大小相同”的情况。

```python
from mem import SimpleDramLatencyModel

dram = SimpleDramLatencyModel.hb_npu_hbm2_defaults()
report = dram.read_addresses(
    [0x1000, 0x1080, 0x2000, 0x2080],
    access_size=32,
)
print(report.total_cycles)
print(report.fragmentation_cycles)
print(report.cross_row_cycles)
```

也可以直接写成：

```python
report = dram.read(
    addresses=[0x1000, 0x1080, 0x2000, 0x2080],
    access_size=32,
)
```

说明：

- `addresses` 的顺序会被保留
- 模块不会自动排序，因为访问顺序本身会影响 row hit / row conflict
- `access_size` 表示每个地址对应的读写大小
- 如果不传 `access_size`，默认按一个 burst，即 `32B`


### 6.3 方式三：显式碎片列表

适合“碎片化，而且每个碎片大小不同”的情况。

```python
from mem import SimpleDramLatencyModel

dram = SimpleDramLatencyModel.hb_npu_hbm2_defaults()
report = dram.read(
    fragments=[
        (0x1000, 16),
        (0x1080, 48),
        (0x2000, 64),
    ]
)
print(report.total_cycles)
```

这里每个 tuple 都是：

- `(fragment_addr, fragment_size)`


## 7. 返回结果说明

每次 `read()` / `write()` 返回一个 `AccessReport`。

最常用字段如下：

- `total_cycles`
  整次访问总延迟

- `setup_cycles`
  激活/预充电相关延迟总和

- `data_cycles`
  列访问 + burst 传输部分的延迟

- `cross_row_cycles`
  跨行导致的额外延迟

- `fragmentation_cycles`
  碎片化导致的额外延迟统计

- `partial_burst_cycles`
  非 burst 对齐带来的浪费

- `row_hit_count`
  命中已打开 row 的 chunk 数

- `row_miss_count`
  在没有打开 row 的情况下访问的 chunk 数

- `row_conflict_count`
  需要切换 row 的 chunk 数

- `chunks`
  每个 row 内 chunk 的详细明细


## 8. 行为约定

### 8.1 open row 会被保留

模型默认是 open-page 风格：

- 一次访问结束后，当前 row 会继续保持打开
- 下一次若还打到同一个 row，会得到 row hit

如果你想让下一次访问从 closed row 开始，可以调用：

```python
dram.close_open_row()
```

或者直接：

```python
dram.reset()
```


### 8.2 当前模型是单 bank 顺序模型

当前实现默认只跟踪一个 bank 的 open row 状态，因此它更适合：

- 上层已经把 bank/channel 并行性折算掉
- 你只想估算单条数据流的延迟
- 你只关心 row locality 和碎片化成本

如果后面需要扩展到多 bank，可以在这个版本之上再加：

- 地址到 bank 的映射
- 每个 bank 独立的 `_BankState`


## 9. 推荐使用方式

如果你的输入是：

- 一个连续区间：用 `read(addr, size)` 或 `write(addr, size)`
- 一个等粒度地址列表：用 `read_addresses(addresses, access_size)` 或 `write_addresses(...)`
- 一个变长碎片列表：用 `fragments=[(addr, size), ...]`

推荐优先顺序：

1. 连续区间时用 `addr + size`
2. 碎片化且每个碎片大小相同时用 `addresses + access_size`
3. 碎片大小不同时用 `fragments`


## 10. 一个最小示例

```python
from mem import SimpleDramLatencyModel

dram = SimpleDramLatencyModel.hb_npu_hbm2_defaults()

# 1) 连续访问
r0 = dram.read(0x0000, 32)
print("first read:", r0.total_cycles)

# 2) 同 row 命中
r1 = dram.read(0x0020, 32)
print("row hit:", r1.total_cycles)

# 3) 跨行访问
r2 = dram.read(0x0800, 32)
print("cross row total:", r2.total_cycles)
print("cross row extra:", r2.cross_row_cycles)

# 4) 碎片化访问
dram.reset()
r3 = dram.read_addresses([0x0000, 0x0040, 0x0080], access_size=16)
print("fragmented total:", r3.total_cycles)
print("fragmented extra:", r3.fragmentation_cycles)
```


## 11. 与 DRAMsim3 的再次对表确认

我重新做了一次源码级和实际运行级对表。

### 11.1 读路径

我用 `refs/HB-NPU-simulator` 源码编了一个最小 probe，测了 3 个场景：

- cold read
- row-hit read
- row-conflict read

`DRAMsim3/HB-NPU` 实测结果：

- cold read = `32`
- row-hit read = `18`
- row-conflict read = `46`

Python 模块当前结果：

- cold read = `32`
- row-hit read = `18`
- row-conflict read = `46`

这 3 项已经对齐。

### 11.2 写路径

这里要区分两种“完成”定义：

1. `DRAMsim3` 的 write callback

- 写请求进入写缓冲后很快就会回调
- 在 probe 里测到的是 `2 cycles`
- 这更像“请求被接收”

2. DRAM 基本时序上的写服务延迟

- row-hit write:
  `transaction_overhead + tCWL + burst_cycles = 2 + 4 + 2 = 8`
- cold write:
  `transaction_overhead + tRCD + tCWL + burst_cycles = 2 + 14 + 4 + 2 = 22`

因此当前 Python 模块对写选择的是：

- 对齐“基本 DRAM 写服务延迟”
- 不对齐 `DRAMsim3` 的 `write_callback=2` 这个接口语义

这样更适合你要的“读写延迟信息”。

### 11.3 结论

在当前仓库使用的 HBM2 参数下，这个 Python 模块现在可以认为：

- 对读延迟已经基本与 `DRAMsim3` 对齐
- 对跨行额外延迟与 row 状态转换逻辑对齐
- 对连续大块访问，已经按 `DRAMsim3` 事务粒度拆分，不再低估
- 对碎片化访问，能正确累计重复事务、row conflict 和 burst 未对齐损失
- 对写延迟，刻意保留“物理服务延迟”而不是 `DRAMsim3` 的快速 callback


## 12. 结论

这个模块不是 DRAMsim3 的完整替代，也不是 HB-NPU-simulator 的逐行翻译。

它做的是一件更聚焦的事：

- 用 HB-NPU 的 HBM2 参数做默认配置
- 保留 row hit / row miss / row conflict
- 保留跨行额外延迟
- 保留碎片化额外延迟
- 同时支持连续区间和地址列表两种输入

如果后面你希望再往真实硬件靠一步，最值得补的下一项不是 refresh，而是：

- 多 bank 建模
- 地址到 bank/channel 的映射
