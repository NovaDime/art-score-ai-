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
    except Exception:
        pass
    return None


def safe_int(value, default=75):
    try:
        value = int(value)
        if value < 0:
            return 0
        if value > 100:
            return 100
        return value
    except Exception:
        return default


def normalize_result(result):
    """把模型输出整理成稳定结构"""
    return {
        "image_description": str(
            result.get(
                "image_description",
                "这张图片包含一定的主体、场景和视觉信息，整体画面具备基本表达。"
            )
        ).strip(),
        "composition": safe_int(result.get("composition"), 75),
        "color": safe_int(result.get("color"), 75),
        "creativity": safe_int(result.get("creativity"), 75),
        "completeness": safe_int(result.get("completeness"), 75),
        "theme": safe_int(result.get("theme"), 75),
        "comment": str(
            result.get(
                "comment",
                "该作品整体表现较稳定，具备一定展示基础。"
            )
        ).strip(),
        "suggestion": str(
            result.get(
                "suggestion",
                "建议继续加强细节、层次与主题表达的一致性。"
            )
        ).strip(),
    }


def analyze_image(image_path, rubric=""):
    prompt = f"""
请你作为专业美术老师，对图片先做视觉内容识别，再做评估。

你必须先认真观察图片内容，明确回答这张图片里大概有什么、画面场景是什么、主体是什么、整体风格如何。
然后再从下面五个维度评分（0-100）：
- composition（构图）
- color（色彩）
- creativity（创意）
- completeness（完整度）
- theme（主题契合）

要求：
1. 必须只返回 JSON
2. 不要输出任何解释文字
3. 不要输出 markdown 代码块
4. image_description 字段必须写成自然、具体、像真的看过图一样的描述
5. 格式如下：

{{
  "image_description": "这张图片展示了什么，有哪些主要元素，画面风格如何",
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
        max_new_tokens=420,
    )

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    print("\n====== 模型原始输出 ======")
    print(output_text)

    result = extract_json(output_text)

    if result is None:
        print("⚠️ JSON解析失败，返回兜底结果")
        return {
            "image_description": "模型本次未稳定输出图片描述，建议重新测试该图片。",
            "composition": 75,
            "color": 75,
            "creativity": 75,
            "completeness": 75,
            "theme": 75,
            "comment": "模型解析失败，当前使用默认评价。",
            "suggestion": "建议重新上传图片或继续优化提示词。",
        }

    return normalize_result(result)


if __name__ == "__main__":
    test_image_path = "/home/xsuper/art-score-ai/tests/test.jpg"

    res = analyze_image(test_image_path)

    print("\n====== 最终结构化结果 ======")
    print(json.dumps(res, indent=2, ensure_ascii=False))