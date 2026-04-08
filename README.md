# AI智能评估与成长系统 8.0 Beta

这是基于 7.4 稳定版继续向前推进的 **Beta 预发布骨架版**。

它的目标不是替代你已经提交的 3B 稳定版，而是用来展示：

- 下一阶段将接入更大参数量视觉模型
- 下一阶段将接入知识库增强评估
- 下一阶段将支持“稳定版 / Beta 版”双模式切换

当前这套 8.0 更适合：

- 路线展示
- 预发布演示
- 后续迭代起点
- 老师和评委看“你下一步要怎么做”

---

## 1. 8.0 相比 7.4 新增了什么

### Beta 模型模式

前端加入：

- 稳定版 V1
- Beta 大模型版

后端加入：

- `mode=stable / beta`
- `mode_requested`
- `mode_used`
- `fallback_used`

当 Beta 模型没准备好或路径无效时，会自动回退到稳定版继续完成推理，方便演示时不崩。

### 知识库骨架

新增目录：

- `knowledge_base/files/`
- `data/kb_index.json`
- `data/kb_files.json`

新增接口：

- `POST /kb/upload` 上传知识库文件
- `GET /kb/list` 查看知识库文件
- `POST /kb/search` 检索知识库片段

支持文件类型：

- txt
- md
- json
- csv
- pdf
- docx

### 知识增强评估

`/upload` 新增两个参数：

- `mode`
- `use_kb`

当 `use_kb=true` 时，后端会先对知识库做简单检索，再把命中的片段拼进 Prompt，让 Beta 版更像“基于教师标准/课程文档增强”的评估路线。

---

## 2. 项目结构

```bash
art-score-ai-8.0-beta/
├─ main.py
├─ index.html
├─ requirements.txt
├─ start_local.sh
├─ start_server.sh
├─ stop.sh
├─ .env.example
├─ data/
│  ├─ history.json
│  ├─ kb_index.json
│  └─ kb_files.json
├─ uploads/
├─ logs/
└─ knowledge_base/
   └─ files/
```

---

## 3. 运行前准备

建议仍然沿用你原来的 `artscore` 环境。

```bash
pip install -r requirements.txt
```

如果你要真的启用 Beta 大模型，可以在 `.env.example` 基础上配置：

- `STABLE_MODEL_PATH`
- `BETA_MODEL_PATH`
- `DEFAULT_BETA_MODE`
- `ENABLE_KB`

---

## 4. 启动方式

### 本地模式

```bash
chmod +x start_local.sh stop.sh
./start_local.sh
```

### 服务器模式

```bash
chmod +x start_server.sh stop.sh
./start_server.sh
```

### 停止

```bash
./stop.sh
```

---

## 5. 8.0 的定位

### 已适合展示的部分

- Beta 双模式切换界面
- 知识库上传与列表展示
- 知识库命中片段展示
- 后端知识增强骨架
- Beta 回退机制
- 大模型路径配置预留

### 还不是重点的部分

- 不追求数据库化
- 不追求多用户权限
- 不追求完整部署
- 不追求生产级 RAG
- 不追求复杂向量检索

这版的核心价值是：

> **把“下一个阶段的大模型 + 知识库方向”提前落成可讲、可看、可继续开发的代码框架。**

---

## 6. 推荐 Beta 演示说法

你可以直接这样讲：

> 7.4 是我们已经提交的稳定版，重点解决本地多模态评估闭环。
> 8.0 Beta 则提前搭好了下一阶段的扩展框架：一方面支持更大模型切换，另一方面支持知识库增强，让系统未来能根据课程标准、教师文档和示例规范做更贴合场景的评估。

---

## 7. 建议你后续怎么继续冲

1. 把 `BETA_MODEL_PATH` 替换成你下一步要试的大模型路径。
2. 先往知识库里放几份教师评分标准、课程要求、示例说明文档。
3. 先让 Beta 跑通“能搜到、能拼 Prompt、能展示命中片段”。
4. 再考虑更强的检索方式，如 embedding / rerank / Open WebUI / RAG pipeline。
