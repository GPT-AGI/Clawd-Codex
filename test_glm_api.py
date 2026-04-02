#!/usr/bin/env python3
"""测试 GLM API 连接"""
import os
try:
    from dotenv import load_dotenv  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    def load_dotenv(*args, **kwargs):  # type: ignore
        return None

try:
    from zhipuai import ZhipuAI  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    ZhipuAI = None

# 加载环境变量
load_dotenv()

def test_glm_api():
    """测试 GLM API 是否正常工作"""
    if ZhipuAI is None:
        print("⚠️  zhipuai 未安装，跳过 GLM API 连接测试")
        return True
    api_key = os.getenv('GLM_API_KEY')
    base_url = os.getenv('GLM_BASE_URL')
    model = os.getenv('GLM_DEFAULT_MODEL', 'glm-4')

    print(f"🔍 检查配置:")
    print(f"  API Key: {api_key[:20]}..." if api_key else "  ❌ API Key 未设置")
    print(f"  Base URL: {base_url}")
    print(f"  Model: {model}")
    print()

    if not api_key:
        print("❌ 错误: GLM_API_KEY 未设置")
        return False

    try:
        print("🚀 测试 GLM API 连接...")
        client = ZhipuAI(api_key=api_key)

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": "你好，请用一句话介绍你自己"}
            ],
            temperature=0.7,
            max_tokens=500  # 增加到 500
        )

        print("✅ API 连接成功!")

        # 提取内容
        message = response.choices[0].message
        content = message.content
        reasoning = getattr(message, 'reasoning_content', None)
        finish_reason = response.choices[0].finish_reason

        print(f"\n📝 响应内容:")
        if reasoning:
            print(f"  推理过程: {reasoning[:200]}...")
        print(f"  最终回答: {content}")
        print(f"  结束原因: {finish_reason}")

        print(f"\n📊 使用情况:")
        print(f"  Prompt tokens: {response.usage.prompt_tokens}")
        print(f"  Completion tokens: {response.usage.completion_tokens}")
        print(f"  Total tokens: {response.usage.total_tokens}")

        # 检查是否有详细 token 信息
        if hasattr(response.usage, 'completion_tokens_details'):
            details = response.usage.completion_tokens_details
            print(f"  Reasoning tokens: {details.get('reasoning_tokens', 'N/A')}")

        return True

    except Exception as e:
        print(f"❌ API 调用失败: {e}")
        print(f"错误类型: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_glm_api()
    exit(0 if success else 1)
