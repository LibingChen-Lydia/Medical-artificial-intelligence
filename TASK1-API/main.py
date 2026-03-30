import os
import json
import requests
import re
from dotenv import load_dotenv
from PyPDF2 import PdfReader

load_dotenv()

QWEN_API_KEY = os.getenv("DASHSCOPE_API_KEY")
QWEN_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"


def test_qwen():
    headers = {
        "Authorization": f"Bearer {QWEN_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "qwen-turbo",
        "messages": [
            {"role": "user", "content": "你好，请只回复：调用成功"}
        ],
        "temperature": 0
    }

    response = requests.post(QWEN_URL, headers=headers, json=payload, timeout=60)
    print("STATUS:", response.status_code)
    print("TEXT:", response.text)


def extract_text_from_pdf(pdf_path: str) -> str:
    try:
        reader = PdfReader(pdf_path)
        text = []
        for page in reader.pages:
            content = page.extract_text()
            if content:
                text.append(content)
        return "\n".join(text)
    except Exception as e:
        raise RuntimeError(f"PDF读取失败: {e}")


def build_messages(case_text: str):
    system_prompt = """
你是医学信息抽取专家。请从病例文本中提取结构化医学实体。
要求：
1. 严格基于文本，不得编造
2. 若字段缺失：
   - patient_info 输出 {}
   - 其他列表字段输出 []
3. 只输出 JSON，不要输出解释、备注或 markdown
4. diagnosis 必须是列表
5. treatment 必须是列表
6. symptoms 必须是列表
7. medical_history 必须是列表
8. patient_info 必须是对象
"""

    user_prompt = f"""
请从以下病例中提取结构化信息：

{case_text}

字段要求：
- patient_info：对象（如年龄、性别、其他基本信息）
- symptoms：列表
- medical_history：列表
- diagnosis：列表
- treatment：列表

请严格按照以下 JSON 结构输出：
{{
  "patient_info": {{}},
  "symptoms": [],
  "medical_history": [],
  "diagnosis": [],
  "treatment": []
}}
"""

    return [
        {"role": "system", "content": system_prompt.strip()},
        {"role": "user", "content": user_prompt.strip()}
    ]


def call_qwen_api(messages):
    headers = {
        "Authorization": f"Bearer {QWEN_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "qwen-turbo",
        "messages": messages,
        "temperature": 0
    }

    try:
        response = requests.post(QWEN_URL, headers=headers, json=payload, timeout=60)
        print("API STATUS:", response.status_code)
        print("API RAW:", response.text)
        response.raise_for_status()

        result = response.json()

        if "choices" not in result or not result["choices"]:
            raise RuntimeError(f"模型返回异常，缺少 choices: {result}")

        content = result["choices"][0].get("message", {}).get("content")
        if not content:
            raise RuntimeError(f"模型返回异常，缺少 content: {result}")

        return content
    except Exception as e:
        raise RuntimeError(f"API调用失败: {e}")


def clean_llm_json(text: str) -> str:
    text = re.sub(r"```json\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"```", "", text).strip()

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start:end + 1].strip()

    raise ValueError("未找到JSON结构")


def save_json(data: dict, output_path: str):
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        raise RuntimeError(f"保存JSON失败: {e}")


def main():
    pdf_path = "A case of portal vein recanalization and symptomatic heart failure.pdf"
    output_path = "output.json"

    if not QWEN_API_KEY:
        raise EnvironmentError("未设置 DASHSCOPE_API_KEY")

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF文件不存在: {pdf_path}")

    print("读取PDF...")
    case_text = extract_text_from_pdf(pdf_path)

    if not case_text.strip():
        raise ValueError("PDF提取结果为空，请检查PDF是否可解析")

    print("构建Prompt...")
    messages = build_messages(case_text)

    print("调用大模型API...")
    raw_output = call_qwen_api(messages)

    print("解析JSON...")
    cleaned = clean_llm_json(raw_output)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(f"模型输出不是合法JSON: {e}\n原始内容:\n{cleaned}")

    print("保存结果...")
    save_json(data, output_path)

    print("完成，结果已保存到:", output_path)


if __name__ == "__main__":
    main()
