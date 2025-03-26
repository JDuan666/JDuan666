import platform
import subprocess
from llama_cpp import Llama

# 初始化本地 DeepSeek 模型（GGUF 格式）
llm = Llama(model_path="./deepseek-coder-1.3b-instruct.Q4_K_M.gguf", n_ctx=2048)

def detect_cpu_flags():
    try:
        result = subprocess.check_output("lscpu", shell=True, encoding="utf-8")
        for line in result.splitlines():
            if "Flags" in line or "Features" in line:
                return line.split(":")[1].strip().split()
    except Exception:
        return []

def detect_gpu():
    try:
        subprocess.check_output(["nvidia-smi"])
        return "NVIDIA (CUDA)"
    except:
        try:
            rocm_check = subprocess.check_output("rocminfo", shell=True, encoding="utf-8")
            if "Agent" in rocm_check:
                return "AMD (ROCm)"
        except:
            pass
    if platform.system() == "Darwin" and "Apple" in platform.processor():
        return "Apple Silicon (MPS)"
    return "Unknown / CPU Only"

def query_local_llm_for_strategy(cpu_flags, gpu_type):
    prompt = f"""
你是一名 AI 优化编译专家，当前用户希望在本地部署 DeepSeek-Coder 模型。

请根据以下硬件信息，为该用户生成一份自适应的优化方案，并包含具体建议和部署方式：

CPU 支持指令集: {', '.join(cpu_flags)}
GPU 类型: {gpu_type}

要求：
1. 你不能使用已有库的默认优化方案（如 llama.cpp 预设参数、HuggingFace 默认配置）
2. 你必须从硬件特性出发，自主分析哪些优化策略最适合该环境（比如使用哪些量化策略、使用哪种 attention 实现、推理架构等）
3. 输出格式为：\n\n【优化建议】\n...\n\n【自生成部署方案（包括核心代码）】\n...\n"""

    output = llm(prompt, max_tokens=1024, temperature=0.7)
    return output["choices"][0]["text"].strip()

def main():
    print("\U0001F50D 正在检测系统硬件...")
    cpu_flags = detect_cpu_flags()
    gpu_type = detect_gpu()

    print(f"\U0001F9E0 CPU 指令集: {cpu_flags[:10]} ...")
    print(f"\U0001F3AE GPU 类型: {gpu_type}\n")

    print("🧠 正在调用本地 DeepSeek 模型生成专属优化策略...\n")
    strategy = query_local_llm_for_strategy(cpu_flags, gpu_type)
    print(strategy)

if __name__ == "__main__":
    main()
