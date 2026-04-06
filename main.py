from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from datetime import datetime
import os
import uuid
import json
import re
import traceback

from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

app = FastAPI(title="AI智能评估与成长系统")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
DATA_DIR = os.path.join(BASE_DIR, "data")
INDEX_FILE = os.path.join(BASE_DIR, "index.html")
HISTORY_FILE = os.path.join(DATA_DIR, "history.json")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

if os.path.exists(UPLOAD_DIR):
    app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
}
MAX_FILE_SIZE = 10 * 1024 * 1024

MODEL_PATH = os.getenv(
    "MODEL_PATH",
    os.path.join(BASE_DIR, "models", "Qwen", "Qwen2.5-VL-3B-Instruct")
)

processor = None
model = None
model_load_error = None


class ReviewRecord(BaseModel):
    id: str
    original_filename: str
    saved_as: str
    image_url: str
    rubric: str
    score: int
    image_description: str
    comment: str
    suggestion: str
    dimensions: Dict[str, int]
    strengths: List[str]
    weaknesses: List[str]
    created_at: str


def load_history() -> List[Dict[str, Any]]:
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except Exception:
        return []


def save_history(records: List[Dict[str, Any]]) -> None:
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def validate_upload(file: UploadFile, content: bytes) -> str:
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext == "":
        ext = ".jpg"
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="仅支持 jpg、jpeg、png、webp 图片格式")
    if file.content_type and file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=400, detail="上传文件类型不正确，请上传图片")
    if not content:
        raise HTTPException(status_code=400, detail="上传文件为空")
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="图片不能超过 10MB")
    return ext


def clamp(value: int, low: int = 0, high: int = 100) -> int:
    return max(low, min(high, int(value)))


def safe_int(value: Any, default: int = 75) -> int:
    try:
        return clamp(int(value))
    except Exception:
        return default


def extract_json(text: str) -> Optional[Dict[str, Any]]:
    try:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group())
    except Exception:
        return None
    return None


def build_strengths_weaknesses(dimensions: Dict[str, int]) -> Dict[str, List[str]]:
    name_map = {
        "composition": "构图表达",
        "color": "色彩表现",
        "creativity": "创意亮点",
        "completeness": "完整度",
        "theme": "主题契合",
    }

    strengths_pool = {
        "composition": "画面布局比较稳定，主体位置明确，观看路径较清晰。",
        "color": "整体配色较协调，主色调统一，视觉感受比较舒服。",
        "creativity": "作品有一定新意，不是单纯堆元素，能看到自己的表达。",
        "completeness": "完成度较好，整体呈现比较完整，没有明显缺块。",
        "theme": "主题表达比较直接，观者能较快理解作品想传递的重点。",
    }

    weakness_pool = {
        "composition": "构图层次还可以再拉开，局部信息分布略显平均。",
        "color": "色彩对比度和重点区域的色彩控制还可更突出一些。",
        "creativity": "创意点已经有基础，但记忆点还可以继续强化。",
        "completeness": "细节收边和局部精修还可以继续加强。",
        "theme": "主题和视觉元素之间的呼应还能再更统一一些。",
    }

    strongest_key = max(dimensions, key=dimensions.get)
    weakest_key = min(dimensions, key=dimensions.get)

    strengths = [
        strengths_pool[strongest_key],
        f"当前在{name_map[strongest_key]}方面表现更突出，适合作为后续继续打磨的优势方向。",
    ]
    weaknesses = [
        weakness_pool[weakest_key],
        f"建议下一步重点提升{name_map[weakest_key]}，让整体说服力更进一步。",
    ]
    return {
        "strengths": strengths,
        "weaknesses": weaknesses,
    }


def load_qwen_model() -> None:
    global processor, model, model_load_error

    if processor is not None and model is not None:
        return

    try:
        print("加载 Qwen2.5-VL 模型中...")

        processor = AutoProcessor.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True,
            use_fast=False,
        )

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_PATH,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )

        model_load_error = None
        print("Qwen2.5-VL 模型加载完成")
    except Exception as e:
        model_load_error = str(e)
        print("模型加载失败：", e)
        traceback.print_exc()


@app.on_event("startup")
def startup_load_model():
    load_qwen_model()


def analyze_image_with_qwen(image_path: str, rubric: str = "") -> Dict[str, Any]:
    global processor, model, model_load_error

    if processor is None or model is None:
        load_qwen_model()

    if processor is None or model is None:
        raise RuntimeError(f"模型未成功加载：{model_load_error}")

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
2. 不要输出任何额外解释、前言、后记、markdown 代码块
3. image_description 字段必须写成自然、具体、像“AI真的看过图”一样的描述，不能只写“这是一张图片”
4. comment 字段写整体评价
5. suggestion 字段写改进建议
6. 输出格式必须严格如下：

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
                {
                    "type": "image",
                    "image": f"file://{image_path}",
                },
                {
                    "type": "text",
                    "text": prompt,
                },
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

    print("开始视觉评分推理...")

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
        raise RuntimeError("模型输出无法解析为 JSON")

    dimensions = {
        "composition": safe_int(result.get("composition"), 75),
        "color": safe_int(result.get("color"), 75),
        "creativity": safe_int(result.get("creativity"), 75),
        "completeness": safe_int(result.get("completeness"), 75),
        "theme": safe_int(result.get("theme"), 75),
    }

    score = round(sum(dimensions.values()) / 5)

    image_description = str(
        result.get(
            "image_description",
            "这张图片包含清晰的主体与场景信息，整体具有一定视觉表达和内容组织。"
        )
    ).strip()

    comment = str(
        result.get(
            "comment",
            "该作品整体表现较稳定，具备一定展示基础。"
        )
    ).strip()

    suggestion = str(
        result.get(
            "suggestion",
            "建议继续加强细节、层次与主题表达的一致性。"
        )
    ).strip()

    extra = build_strengths_weaknesses(dimensions)

    return {
        "score": score,
        "image_description": image_description,
        "comment": comment,
        "suggestion": suggestion,
        "dimensions": dimensions,
        "strengths": extra["strengths"],
        "weaknesses": extra["weaknesses"],
        "raw_output": output_text,
    }


@app.get("/")
def root() -> FileResponse:
    if not os.path.exists(INDEX_FILE):
        raise HTTPException(status_code=404, detail="未找到 index.html")
    return FileResponse(INDEX_FILE)


@app.get("/health")
def health() -> Dict[str, Any]:
    history = load_history()
    return {
        "status": "ok",
        "service": "AI智能评估与成长系统后端",
        "history_count": len(history),
        "upload_dir": UPLOAD_DIR,
        "model_loaded": processor is not None and model is not None,
        "model_path": MODEL_PATH,
        "model_load_error": model_load_error,
    }


@app.get("/history")
def history(limit: int = 10) -> JSONResponse:
    records = load_history()
    records = list(reversed(records))[: max(1, min(limit, 100))]
    return JSONResponse(content={"items": records, "count": len(records)})


@app.post("/upload")
async def upload_image(file: UploadFile = File(...), rubric: str = Form("")) -> JSONResponse:
    content = await file.read()
    ext = validate_upload(file, content)

    new_name = f"{uuid.uuid4().hex}{ext}"
    save_path = os.path.join(UPLOAD_DIR, new_name)

    with open(save_path, "wb") as f:
        f.write(content)

    try:
        review = analyze_image_with_qwen(save_path, rubric)
    except Exception as e:
        print("模型分析失败：", e)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"AI评估失败：{str(e)}")

    now = datetime.now().strftime("%Y/%m/%d %H:%M:%S")

    record = ReviewRecord(
        id=uuid.uuid4().hex,
        original_filename=file.filename or new_name,
        saved_as=new_name,
        image_url=f"/uploads/{new_name}",
        rubric=rubric,
        score=review["score"],
        image_description=review["image_description"],
        comment=review["comment"],
        suggestion=review["suggestion"],
        dimensions=review["dimensions"],
        strengths=review["strengths"],
        weaknesses=review["weaknesses"],
        created_at=now,
    )

    history_records = load_history()
    history_records.append(record.model_dump())
    save_history(history_records)

    return JSONResponse(
        content={
            "filename": file.filename,
            "saved_as": new_name,
            "saved_path": save_path,
            "image_url": f"/uploads/{new_name}",
            "score": review["score"],
            "image_description": review["image_description"],
            "comment": review["comment"],
            "suggestion": review["suggestion"],
            "dimensions": review["dimensions"],
            "strengths": review["strengths"],
            "weaknesses": review["weaknesses"],
            "created_at": now,
        }
    )


USER = {
    "username": "CGG41",
    "password": "52113",
    "name": "杠上开花",
    "student_id": "52113"
}


@app.post("/login")
def login(data: dict = Body(...)):
    username = data.get("username")
    password = data.get("password")

    print("\n====== 登录数据 ======")
    print(f"输入账号: [{username}], 输入密码: [{password}]")
    print("======================\n")

    if username == USER["username"] and password == USER["password"]:
        return {
            "success": True,
            "name": USER["name"],
            "student_id": USER["student_id"]
        }
    else:
        return {"success": False, "msg": "账号或密码错误"}