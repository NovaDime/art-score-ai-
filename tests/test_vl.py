from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
import json
import re

model_path = "/home/xsuper/art-score-ai/models/Qwen/Qwen2.5-VL-3B-Instruct"

print("加载模型中...")

processor = AutoProcessor.from_pretrained(
    model_path,
    trust_remote_code=True,
    use_fast=False,
)

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True,
)

print("模型加载完成")


def extract_json(text):
    """从模型输出中提取 JSON"""
    try:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group())
    except:
        pass
    return None


def analyze_image(image_path, rubric=""):
    prompt = f"""
请你作为专业美术老师，对图片进行评估。

评分维度（0-100）：
- composition（构图）
- color（色彩）
- creativity（创意）
- completeness（完整度）
- theme（主题契合）

要求：
1. 必须返回 JSON
2. 不要输出任何解释文字
3. 格式如下：

{{
  "composition": 80,
  "color": 75,
  "creativity": 85,
  "completeness": 78,
  "theme": 82,
  "comment": "整体评价",
  "suggestion": "改进建议"
}}

教师评分标准参考：
{rubric}
"""

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"file://{image_path}"},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    print("开始推理...")

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=300,
    )

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    print("\n原始输出：")
    print(output_text)

    result = extract_json(output_text)

    if result is None:
        print("⚠️ JSON解析失败，返回兜底结果")
        return {
            "composition": 75,
            "color": 75,
            "creativity": 75,
            "completeness": 75,
            "theme": 75,
            "comment": "模型解析失败，使用默认评价",
            "suggestion": "建议重新上传图片或优化提示词",
        }

    return result


# ===== 测试 =====
if __name__ == "__main__":
    res = analyze_image("/home/xsuper/art-score-ai/tests/test.jpg")
    print("\n最终结构化结果：")
    print(json.dumps(res, indent=2, ensure_ascii=False))