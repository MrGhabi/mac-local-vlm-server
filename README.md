# Local LLM/VLM Server

本地LLM/VLM推理服务器，基于MLX，提供OpenAI兼容API。

**环境要求：** Mac M系列芯片（M1/M2/M3/M4）
**性能优势：** MLX专为Apple Silicon优化，充分利用统一内存架构，是Mac上运行LLM/VLM的最优选择。

## 特性

- OpenAI兼容API (`/v1/chat/completions`)
- 支持本地文件路径直接传图片
- 单模型内存策略，自动卸载旧模型
- 混合引擎：VLM/LLM自动fallback
- 多端口支持

## 安装

### 1. 安装依赖

```bash
python3 -m venv ~/envs/qwen3-vl
source ~/envs/qwen3-vl/bin/activate
pip install mlx-vlm
```

### 2. 下载模型

模型存放在 `./data/hub/` 目录，使用HuggingFace Hub下载：

```bash
# 方法1：使用huggingface-cli下载（推荐）
pip install huggingface-hub
export HF_HOME="$(pwd)/data"
huggingface-cli download mlx-community/Qwen3-VL-4B-Instruct-4bit

# 方法2：首次运行时自动下载
# 启动服务时会自动从HuggingFace下载模型到./data/hub/目录
```

支持的模型：
- `mlx-community/Qwen3-VL-4B-Instruct-4bit` (2.9GB)
- `mlx-community/Qwen3-VL-8B-Instruct-4bit` (5.4GB) **← 推荐，性能和速度都不错**
- `mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit` (16GB)

### 3. 编译安装

```bash
cargo build --release
cp local_vlm_server.py vlm_infer.py target/release/
```

## 使用

### 启动服务

```bash
./target/release/llm-server start
./target/release/llm-server start --model mlx-community/Qwen3-VL-8B-Instruct-4bit
./target/release/llm-server start --port 58081
```

### 管理服务

```bash
./target/release/llm-server status
./target/release/llm-server stop
./target/release/llm-server restart
```

### 直接推理（无需启动服务）

```bash
source ~/envs/qwen3-vl/bin/activate
HF_HOME="$(pwd)/data" python3 vlm_infer.py \
  --image /path/to/image.png \
  --prompt "描述这张图片"
```

### API调用

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:58080/v1",
    api_key="dummy"
)

response = client.chat.completions.create(
    model="mlx-community/Qwen3-VL-4B-Instruct-4bit",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "描述这张图片"},
            {"type": "image_url", "image_url": {"url": "/path/to/image.png"}}
        ]
    }]
)
print(response.choices[0].message.content)
```

## 配置

编辑 `config.toml`：

```toml
[server]
port = 58080
host = "127.0.0.1"

[model]
default_model = "mlx-community/Qwen3-VL-4B-Instruct-4bit"
venv_path = "~/envs/qwen3-vl"
allowed_models = [
    "mlx-community/Qwen3-VL-4B-Instruct-4bit",
    "mlx-community/Qwen3-VL-8B-Instruct-4bit",
    "mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit",
]

[inference]
max_tokens = 256
temperature = 0.1

[paths]
pid_file = "/tmp/llm-server.pid"
log_file = "~/.llm-server.log"
```

## 性能参考

M4 Pro (48GB RAM):
- 内存占用: ~5.6GB
- 推理速度: ~67 tokens/s
- 首次加载: ~10-15s
- 后续请求: ~1-2s

## License

MIT
