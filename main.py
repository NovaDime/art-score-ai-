from __future__ import annotations

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path
import json
import os
import re
import traceback
import uuid

try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None

try:
    from docx import Document
except Exception:
    Document = None

try:
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from qwen_vl_utils import process_vision_info
except Exception:
    AutoProcessor = None
    Qwen2_5_VLForConditionalGeneration = None
    process_vision_info = None

app = FastAPI(title="AI智能评估与成长系统 8.0 Beta")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = BASE_DIR / "uploads"
KB_DIR = BASE_DIR / "knowledge_base"
KB_FILES_DIR = KB_DIR / "files"
INDEX_FILE = BASE_DIR / "index.html"
HISTORY_FILE = DATA_DIR / "history.json"
KB_INDEX_FILE = DATA_DIR / "kb_index.json"
KB_META_FILE = DATA_DIR / "kb_files.json"

for item in [DATA_DIR, UPLOAD_DIR, KB_DIR, KB_FILES_DIR]:
    item.mkdir(parents=True, exist_ok=True)

if UPLOAD_DIR.exists():
    app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")
if KB_FILES_DIR.exists():
    app.mount("/kb_files", StaticFiles(directory=str(KB_FILES_DIR)), name="kb_files")

ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}
ALLOWED_DOC_EXTENSIONS = {".txt", ".md", ".json", ".csv", ".pdf", ".docx"}
MAX_IMAGE_SIZE = 10 * 1024 * 1024
MAX_DOC_SIZE = 20 * 1024 * 1024

STABLE_MODEL_PATH = os.getenv(
    "STABLE_MODEL_PATH",
    str(BASE_DIR / "models" / "Qwen" / "Qwen2.5-VL-3B-Instruct"),
)
BETA_MODEL_PATH = os.getenv(
    "BETA_MODEL_PATH",
    str(BASE_DIR / "models" / "Qwen" / "Qwen2.5-VL-7B-Instruct"),
)
DEFAULT_BETA_MODE = os.getenv("DEFAULT_BETA_MODE", "true").lower() == "true"
ENABLE_KB = os.getenv("ENABLE_KB", "true").lower() == "true"
DEMO_LOGIN_USER = os.getenv("APP_DEMO_USER", "CGG41")
DEMO_LOGIN_PASSWORD = os.getenv("APP_DEMO_PASSWORD", "52113")
DEMO_LOGIN_NAME = os.getenv("APP_DEMO_NAME", "杠上开花")
DEMO_LOGIN_STUDENT_ID = os.getenv("APP_DEMO_STUDENT_ID", "52113")

model_registry: Dict[str, Dict[str, Any]] = {
    "stable": {"processor": None, "model": None, "error": None, "path": STABLE_MODEL_PATH},
    "beta": {"processor": None, "model": None, "error": None, "path": BETA_MODEL_PATH},
}


class ReviewRecord(BaseModel):
    id: str
    original_filename: str
    saved_as: str
    image_url: str
    rubric: str
    mode: str
    used_kb: bool
    score: int
    image_description: str
    comment: str
    suggestion: str
    dimensions: Dict[str, int]
    strengths: List[str]
    weaknesses: List[str]
    kb_hits: List[Dict[str, Any]]
    created_at: str


class KBSearchRequest(BaseModel):
    query: str
    top_k: int = 5


def now_str() -> str:
    return datetime.now().strftime("%Y/%m/%d %H:%M:%S")


def load_json_list(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []


def save_json_list(path: Path, items: List[Dict[str, Any]]) -> None:
    path.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")


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
        "composition": "画面布局较稳，主体位置清晰，浏览路径自然。",
        "color": "配色关系较协调，主次色分配比较舒服。",
        "creativity": "作品具备一定记忆点，表达不只是堆砌元素。",
        "completeness": "整体完成度较好，收边和呈现相对完整。",
        "theme": "主题表达比较直接，观看者较容易理解中心内容。",
    }
    weakness_pool = {
        "composition": "构图层次仍可继续拉开，局部节奏略显平均。",
        "color": "色彩对比和视觉重点还可以再更明确一些。",
        "creativity": "创意点已经有雏形，但核心概念还可更强化。",
        "completeness": "细节精修和局部统一性还可进一步提升。",
        "theme": "主题与视觉元素的呼应还可以更紧密。",
    }
    strongest_key = max(dimensions, key=dimensions.get)
    weakest_key = min(dimensions, key=dimensions.get)
    return {
        "strengths": [
            strengths_pool[strongest_key],
            f"当前在{name_map[strongest_key]}方面更突出，适合作为持续强化的优势方向。",
        ],
        "weaknesses": [
            weakness_pool[weakest_key],
            f"建议下一步重点提升{name_map[weakest_key]}，让整体说服力更进一步。",
        ],
    }


def validate_image_upload(file: UploadFile, content: bytes) -> str:
    ext = Path(file.filename or "").suffix.lower() or ".jpg"
    if ext not in ALLOWED_IMAGE_EXTENSIONS:
        raise HTTPException(status_code=400, detail="仅支持 jpg、jpeg、png、webp 图片格式")
    if file.content_type and file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(status_code=400, detail="上传文件类型不正确，请上传图片")
    if not content:
        raise HTTPException(status_code=400, detail="上传文件为空")
    if len(content) > MAX_IMAGE_SIZE:
        raise HTTPException(status_code=400, detail="图片不能超过 10MB")
    return ext


def validate_doc_upload(file: UploadFile, content: bytes) -> str:
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_DOC_EXTENSIONS:
        raise HTTPException(status_code=400, detail="知识库仅支持 txt、md、json、csv、pdf、docx")
    if not content:
        raise HTTPException(status_code=400, detail="知识库文件为空")
    if len(content) > MAX_DOC_SIZE:
        raise HTTPException(status_code=400, detail="知识库文件不能超过 20MB")
    return ext


def read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def read_pdf_file(path: Path) -> str:
    if PdfReader is None:
        raise RuntimeError("当前环境缺少 pypdf，无法解析 PDF")
    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n".join(pages)


def read_docx_file(path: Path) -> str:
    if Document is None:
        raise RuntimeError("当前环境缺少 python-docx，无法解析 DOCX")
    doc = Document(str(path))
    return "\n".join(p.text for p in doc.paragraphs)


def normalize_text(text: str) -> str:
    text = text.replace("\u3000", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def read_doc_content(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in {".txt", ".md", ".json", ".csv"}:
        return normalize_text(read_text_file(path))
    if ext == ".pdf":
        return normalize_text(read_pdf_file(path))
    if ext == ".docx":
        return normalize_text(read_docx_file(path))
    raise RuntimeError("不支持的知识库文件类型")


def split_into_chunks(text: str, max_len: int = 500) -> List[str]:
    cleaned = normalize_text(text)
    if not cleaned:
        return []
    segments = re.split(r"(?<=[。！？!?\n])", cleaned)
    chunks: List[str] = []
    buf = ""
    for seg in segments:
        seg = seg.strip()
        if not seg:
            continue
        if len(buf) + len(seg) + 1 <= max_len:
            buf = f"{buf} {seg}".strip()
        else:
            if buf:
                chunks.append(buf)
            buf = seg
    if buf:
        chunks.append(buf)
    if not chunks:
        return [cleaned[:max_len]]
    return chunks


def tokenize(text: str) -> List[str]:
    text = text.lower()
    tokens = re.findall(r"[\u4e00-\u9fff]{1,4}|[a-z0-9_\-]{2,}", text)
    return tokens


def build_kb_indexes_for_file(saved_name: str, original_name: str, content: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    file_id = uuid.uuid4().hex
    chunks = split_into_chunks(content)
    created_at = now_str()
    kb_chunks: List[Dict[str, Any]] = []
    for idx, chunk in enumerate(chunks, start=1):
        kb_chunks.append({
            "chunk_id": uuid.uuid4().hex,
            "file_id": file_id,
            "filename": original_name,
            "saved_as": saved_name,
            "chunk_index": idx,
            "content": chunk,
            "tokens": tokenize(chunk),
            "created_at": created_at,
        })
    meta = {
        "file_id": file_id,
        "filename": original_name,
        "saved_as": saved_name,
        "chunk_count": len(kb_chunks),
        "created_at": created_at,
        "content_preview": content[:180],
    }
    return kb_chunks, meta


def score_chunk(query: str, chunk: Dict[str, Any]) -> int:
    query_tokens = tokenize(query)
    if not query_tokens:
        return 0
    content = (chunk.get("content") or "").lower()
    token_set = set(chunk.get("tokens") or [])
    score = 0
    for token in query_tokens:
        if token in token_set:
            score += 4
        elif token in content:
            score += 2
    if query.lower() in content:
        score += 6
    return score


def search_kb(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    if not ENABLE_KB:
        return []
    chunks = load_json_list(KB_INDEX_FILE)
    ranked = []
    for chunk in chunks:
        score = score_chunk(query, chunk)
        if score > 0:
            ranked.append({**chunk, "score": score})
    ranked.sort(key=lambda x: (x["score"], x.get("chunk_index", 0)), reverse=True)
    return ranked[:max(1, min(top_k, 8))]


def kb_context_for_prompt(query: str, top_k: int = 3) -> Tuple[str, List[Dict[str, Any]]]:
    hits = search_kb(query=query, top_k=top_k)
    if not hits:
        return "", []
    parts = []
    slim_hits = []
    for idx, item in enumerate(hits, start=1):
        parts.append(f"[知识片段{idx}] 来源：{item['filename']}\n{item['content']}")
        slim_hits.append({
            "filename": item["filename"],
            "chunk_index": item["chunk_index"],
            "content": item["content"],
            "score": item["score"],
        })
    return "\n\n".join(parts), slim_hits


def get_model_bundle(mode: str) -> Dict[str, Any]:
    mode = "beta" if mode == "beta" else "stable"
    return model_registry[mode]


def load_qwen_model(mode: str = "stable") -> None:
    if AutoProcessor is None or Qwen2_5_VLForConditionalGeneration is None or process_vision_info is None:
        get_model_bundle(mode)["error"] = "当前环境缺少 transformers / qwen_vl_utils 依赖"
        return

    bundle = get_model_bundle(mode)
    if bundle["processor"] is not None and bundle["model"] is not None:
        return

    try:
        model_path = bundle["path"]
        print(f"加载 {mode} 模型中: {model_path}")
        bundle["processor"] = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=False,
        )
        bundle["model"] = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )
        bundle["error"] = None
        print(f"{mode} 模型加载完成")
    except Exception as e:
        bundle["error"] = str(e)
        traceback.print_exc()


def startup_load_model() -> None:
    load_qwen_model("stable")
    if DEFAULT_BETA_MODE:
        load_qwen_model("beta")


app.add_event_handler("startup", startup_load_model)


def build_prompt(rubric: str, mode: str, kb_context: str = "") -> str:
    mode_hint = "基础稳定版" if mode == "stable" else "Beta 预发布大模型版"
    kb_hint = (
        f"\n你还可以参考以下知识库内容辅助判断，但不要生搬硬套：\n{kb_context}\n"
        if kb_context else
        ""
    )
    return f"""
请你作为专业美术老师，对图片先做视觉内容识别，再做评估。
当前运行模式：{mode_hint}

你必须先认真观察图片内容，明确回答这张图片里大概有什么、画面场景是什么、主体是什么、整体风格如何。
然后再从下面五个维度评分（0-100）：
- composition（构图）
- color（色彩）
- creativity（创意）
- completeness（完整度）
- theme（主题契合）
{kb_hint}
要求：
1. 必须只返回 JSON
2. 不要输出任何额外解释、前言、后记、markdown 代码块
3. image_description 字段必须写成自然、具体、像“AI真的看过图”一样的描述，不能只写“这是一张图片”
4. comment 字段写整体评价
5. suggestion 字段写改进建议
6. 如果参考了知识库，请让 comment 或 suggestion 更贴合老师标准，但不要虚构知识来源
7. 输出格式必须严格如下：

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


def run_multimodal_inference(image_path: str, prompt: str, mode: str) -> str:
    bundle = get_model_bundle(mode)
    if bundle["processor"] is None or bundle["model"] is None:
        load_qwen_model(mode)

    if bundle["processor"] is None or bundle["model"] is None:
        raise RuntimeError(f"{mode} 模型未成功加载：{bundle['error']}")

    processor = bundle["processor"]
    model = bundle["model"]
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": f"file://{image_path}"},
            {"type": "text", "text": prompt},
        ],
    }]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)
    generated_ids = model.generate(**inputs, max_new_tokens=480 if mode == "beta" else 420)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    return processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]


def analyze_image(image_path: str, rubric: str = "", mode: str = "stable", use_kb: bool = False) -> Dict[str, Any]:
    requested_mode = "beta" if mode == "beta" else "stable"
    active_mode = requested_mode
    fallback_used = False

    kb_context = ""
    kb_hits: List[Dict[str, Any]] = []
    if use_kb and ENABLE_KB:
        kb_context, kb_hits = kb_context_for_prompt(query=rubric or "图片评估 视觉分析", top_k=3)

    prompt = build_prompt(rubric=rubric, mode=active_mode, kb_context=kb_context)
    try:
        output_text = run_multimodal_inference(image_path=image_path, prompt=prompt, mode=active_mode)
    except Exception as beta_error:
        if requested_mode == "beta":
            fallback_used = True
            active_mode = "stable"
            prompt = build_prompt(rubric=rubric, mode=active_mode, kb_context=kb_context)
            output_text = run_multimodal_inference(image_path=image_path, prompt=prompt, mode=active_mode)
        else:
            raise RuntimeError(str(beta_error))

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
    extra = build_strengths_weaknesses(dimensions)

    return {
        "mode_requested": requested_mode,
        "mode_used": active_mode,
        "fallback_used": fallback_used,
        "score": score,
        "image_description": str(result.get("image_description") or "这张图片包含清晰主体和场景信息。"),
        "comment": str(result.get("comment") or "该作品整体表现较稳定。"),
        "suggestion": str(result.get("suggestion") or "建议继续强化细节和主题统一性。"),
        "dimensions": dimensions,
        "strengths": extra["strengths"],
        "weaknesses": extra["weaknesses"],
        "kb_hits": kb_hits,
        "raw_output": output_text,
    }


@app.get("/")
def root() -> FileResponse:
    if not INDEX_FILE.exists():
        raise HTTPException(status_code=404, detail="未找到 index.html")
    return FileResponse(str(INDEX_FILE))


@app.get("/health")
def health() -> Dict[str, Any]:
    history = load_json_list(HISTORY_FILE)
    kb_files = load_json_list(KB_META_FILE)
    return {
        "status": "ok",
        "service": "AI智能评估与成长系统 8.0 Beta",
        "history_count": len(history),
        "kb_file_count": len(kb_files),
        "enable_kb": ENABLE_KB,
        "stable_model": {
            "path": STABLE_MODEL_PATH,
            "loaded": get_model_bundle("stable")["processor"] is not None and get_model_bundle("stable")["model"] is not None,
            "error": get_model_bundle("stable")["error"],
        },
        "beta_model": {
            "path": BETA_MODEL_PATH,
            "loaded": get_model_bundle("beta")["processor"] is not None and get_model_bundle("beta")["model"] is not None,
            "error": get_model_bundle("beta")["error"],
        },
    }


@app.get("/config")
def config() -> Dict[str, Any]:
    return {
        "default_beta_mode": DEFAULT_BETA_MODE,
        "enable_kb": ENABLE_KB,
        "stable_model_path": STABLE_MODEL_PATH,
        "beta_model_path": BETA_MODEL_PATH,
    }


@app.post("/login")
def login(data: dict = Body(...)) -> Dict[str, Any]:
    username = str(data.get("username") or "").strip()
    password = str(data.get("password") or "").strip()
    if username == DEMO_LOGIN_USER and password == DEMO_LOGIN_PASSWORD:
        return {
            "success": True,
            "name": DEMO_LOGIN_NAME,
            "student_id": DEMO_LOGIN_STUDENT_ID,
        }
    return {"success": False, "msg": "账号或密码错误"}


@app.get("/history")
def history(limit: int = 10) -> JSONResponse:
    records = list(reversed(load_json_list(HISTORY_FILE)))[:max(1, min(limit, 100))]
    return JSONResponse(content={"items": records, "count": len(records)})


@app.post("/upload")
async def upload_image(
    file: UploadFile = File(...),
    rubric: str = Form(""),
    mode: str = Form("stable"),
    use_kb: bool = Form(False),
) -> JSONResponse:
    content = await file.read()
    ext = validate_image_upload(file, content)
    saved_name = f"{uuid.uuid4().hex}{ext}"
    save_path = UPLOAD_DIR / saved_name
    save_path.write_bytes(content)

    try:
        review = analyze_image(
            image_path=str(save_path),
            rubric=rubric,
            mode=mode,
            use_kb=use_kb,
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"AI评估失败：{str(e)}")

    created_at = now_str()
    record = ReviewRecord(
        id=uuid.uuid4().hex,
        original_filename=file.filename or saved_name,
        saved_as=saved_name,
        image_url=f"/uploads/{saved_name}",
        rubric=rubric,
        mode=review["mode_used"],
        used_kb=bool(use_kb),
        score=review["score"],
        image_description=review["image_description"],
        comment=review["comment"],
        suggestion=review["suggestion"],
        dimensions=review["dimensions"],
        strengths=review["strengths"],
        weaknesses=review["weaknesses"],
        kb_hits=review["kb_hits"],
        created_at=created_at,
    )
    history_items = load_json_list(HISTORY_FILE)
    history_items.append(record.model_dump())
    save_json_list(HISTORY_FILE, history_items)

    return JSONResponse(content={
        "filename": file.filename,
        "saved_as": saved_name,
        "image_url": f"/uploads/{saved_name}",
        "score": review["score"],
        "image_description": review["image_description"],
        "comment": review["comment"],
        "suggestion": review["suggestion"],
        "dimensions": review["dimensions"],
        "strengths": review["strengths"],
        "weaknesses": review["weaknesses"],
        "mode_requested": review["mode_requested"],
        "mode_used": review["mode_used"],
        "fallback_used": review["fallback_used"],
        "used_kb": bool(use_kb),
        "kb_hits": review["kb_hits"],
        "created_at": created_at,
    })


@app.get("/kb/list")
def kb_list() -> JSONResponse:
    items = list(reversed(load_json_list(KB_META_FILE)))
    return JSONResponse(content={"items": items, "count": len(items)})


@app.post("/kb/upload")
async def kb_upload(file: UploadFile = File(...)) -> JSONResponse:
    if not ENABLE_KB:
        raise HTTPException(status_code=400, detail="当前版本未启用知识库功能")

    content = await file.read()
    ext = validate_doc_upload(file, content)
    saved_name = f"{uuid.uuid4().hex}{ext}"
    save_path = KB_FILES_DIR / saved_name
    save_path.write_bytes(content)

    try:
        raw_text = read_doc_content(save_path)
    except Exception as e:
        save_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"知识库解析失败：{str(e)}")

    if not raw_text.strip():
        save_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="知识库文件解析后为空")

    chunks, meta = build_kb_indexes_for_file(saved_name=saved_name, original_name=file.filename or saved_name, content=raw_text)
    kb_index_items = load_json_list(KB_INDEX_FILE)
    kb_meta_items = load_json_list(KB_META_FILE)
    kb_index_items.extend(chunks)
    kb_meta_items.append(meta)
    save_json_list(KB_INDEX_FILE, kb_index_items)
    save_json_list(KB_META_FILE, kb_meta_items)

    return JSONResponse(content={
        "success": True,
        "filename": meta["filename"],
        "chunk_count": meta["chunk_count"],
        "created_at": meta["created_at"],
        "content_preview": meta["content_preview"],
    })


@app.post("/kb/search")
def kb_search(data: KBSearchRequest) -> JSONResponse:
    items = search_kb(query=data.query, top_k=data.top_k)
    cleaned = [
        {
            "filename": item["filename"],
            "chunk_index": item["chunk_index"],
            "score": item["score"],
            "content": item["content"],
        }
        for item in items
    ]
    return JSONResponse(content={"items": cleaned, "count": len(cleaned)})
