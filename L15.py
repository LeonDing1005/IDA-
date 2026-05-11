import json
import html
import base64
import hashlib
import mimetypes
import time
from pathlib import Path

import pandas as pd
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI

try:
    import zmail
except ImportError:
    zmail = None


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CSV_PATH = BASE_DIR / "data" / "lesson_data.csv"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
OUTPUT_TABLE_DIR = BASE_DIR / "outputTable"
FONT_PATH = BASE_DIR / "chartFont" / "yahei_consola.ttf"


def get_secret_value(*keys: str) -> str:
    for key in keys:
        try:
            value = st.secrets
            for part in key.split("."):
                value = value[part]
            if value:
                return str(value).strip()
        except Exception:
            continue
    return ""

FIELD_LABEL_PROMPT = """
你是一名数据字典专家。请根据字段名、数据类型和少量样例值，为每个字段生成清晰、简短、业务化的中文字段名。

要求：
- 必须为每个输入字段返回一个中文名。
- 中文名要适合直接作为表格表头、图表标签、报告指标名。
- 不要臆造数据中不存在的业务含义；不确定时给出保守中文名。
- 返回严格 JSON，不要附加解释。

返回结构：
{{
  "columns": {{
    "raw_column_name": "中文字段名"
  }}
}}
"""

DEFAULT_ANALYSIS_PROMPT = """
你是一名资深数据分析顾问。请根据当前 CSV 的字段、字段中文名、数据类型和样例值，推荐 3 个最值得一键运行的数据分析问题。

要求：
- 必须贴合当前数据，不要使用不存在的字段。
- 每个问题都要包含清晰的分析目标、需要聚合/比较/计算的关键字段，以及最合适的图表类型要求。
- 三个问题之间要有差异，覆盖不同分析视角，例如规模趋势、结构贡献、转化效率、分组对比、异常风险等。
- 每个问题最多生成 1 张最合适图表。
- 使用中文表达，但可以在必要时括号标注原始字段名。
- 返回严格 JSON，不要附加解释。

返回结构：
{{
  "queries": [
    "<默认分析问题1>",
    "<默认分析问题2>",
    "<默认分析问题3>"
  ]
}}
"""

CHART_CODE_PROMPT = """
Font rule:
- The execution namespace provides FONT_PATH, CHINESE_FONT_PROP, and CHINESE_FONT_NAME.
- Do not use SimHei or any other system font name.
- Use CHINESE_FONT_NAME in rcParams, and pass fontproperties=CHINESE_FONT_PROP to titles, labels, legends, tick labels, and annotations when possible.

你是一名严谨的数据可视化 Python 工程师。请根据当前数据画像、原始分析问题、当前分析结果和用户反馈，生成一段可执行的 matplotlib 绘图代码。

你只能返回严格 JSON：
{{
  "chart_type": "<line|bar|horizontal_bar|stacked_bar|pie|scatter|hist|box|other>",
  "summary": "<图表口径和修正说明，中文>",
  "code": "<Python 代码字符串>",
  "export_paths": ["<可选导出 CSV 路径>"]
}}

代码要求：
- 代码运行环境已经提供：df, pd, plt, output_path, ARTIFACTS_DIR, OUTPUT_TABLE_DIR, FONT_PATH。
- 必须基于 df 重新计算图表数据。
- 必须把最终图片保存到变量 output_path 指定的位置：plt.savefig(output_path, dpi=144, bbox_inches="tight") 或 fig.savefig(output_path, dpi=144, bbox_inches="tight")。
- 不要读取外部文件，不要调用网络，不要修改原始 df；如需派生使用 tmp = df.copy()。
- 图表标题、坐标轴、图例、标注必须使用中文。
- 只生成 1 张图。
- 不要把代码放进 Markdown 代码块。
"""

PROMPT_TEMPLATE = f"""
Font rule for charts:
- Always load the Chinese font from `{FONT_PATH.as_posix()}` with matplotlib.font_manager.
- Do not use SimHei or a generic sans-serif Chinese fallback.
- Set matplotlib rcParams from the loaded local font name before drawing Chinese text.

你扮演“数据分析助理”。仅可对已经注入的 Pandas DataFrame `df` 进行只读分析与绘图：
- 严禁修改 `df` 原始数据；如需派生请使用副本，例如 `tmp = df.copy()`。
- 任何图表必须使用 matplotlib，并保存到本地目录 `{ARTIFACTS_DIR.as_posix()}/`。
- 制作图表时，优先从 `{FONT_PATH.as_posix()}` 加载中文字体。
- 除了源数据字段名、数据内容外，所有描述内容均使用中文作答。
- 对外展示的字段名、表头、图表标题、坐标轴、图例、标注都必须使用中文；如需引用原始英文字段，请写成“中文名称（原字段名）”。
- 图表必须与问题核心指标直接相关，并选择最合适的形式：转化链路用漏斗/阶梯条形图，类别对比用排序柱形图或横向条形图，构成占比用环形图或堆叠图，时间趋势用折线图，双指标关系用散点图。
- 每个分析最多只生成 1 张最合适的图表；`chart_paths` 数组最多只能包含 1 个图片路径。不要为了同一问题生成多个备选图。
- 最终仅返回一个合法 JSON，不要有额外文本。

统一返回 JSON 结构：
{{
  "type": "answer" | "table" | "chart" | "error",
  "input": "<简述用户需求>",
  "data": "<根据 type 返回对应内容>",
  "chart_paths": ["<图表文件路径>"],
  "export_paths": ["<导出表格路径>"]
}}

各 type 的 data 结构：
- type="answer":
  {{"answer": "<先写1行小标题总结，再给要点式答案；包含关键口径和数值>"}}

- type="table":
  {{
    "columns": ["<列名>", "..."],
    "rows": [["<与 columns 对齐的值>", "..."]],
    "sort": {{"by": "<列名>", "order": "asc|desc"}}
  }}
  最多返回 100 行；超过时按问题相关排序截断，并在 answer 或 warnings 中说明。
  表格/数据片段应导出到 `{OUTPUT_TABLE_DIR.as_posix()}/`。

- type="chart":
  {{
    "chart_type": "line|bar|scatter|box|hist",
    "summary": "<生成数据分析报告，并解释可视化图表>"
  }}
  图表导出规范：dpi=144, bbox_inches="tight"。

- type="error":
  {{
    "message": "<错误原因>",
    "missing_columns": ["<列名>", "..."],
    "invalid_filters": {{"列名": "提供的值"}},
    "suggestions": ["<如何改写查询或替代列>", "..."]
  }}

图表自动选择：
- 时间字段 + 序列：line
- 类别字段 + 聚合值：bar
- 两个连续数值字段：scatter
- 其余无法判断：返回 type="error"，说明原因并给出建议

JSON 规范：
- 仅使用双引号；不得出现 NaN/Infinity，请转为 null 或实际数值。
- 所有列名必须存在于 df。
- 仅返回一个 JSON，不要附加解释性文字。
"""


REPORT_PROMPT = """
你是一名资深数据分析报告专家。你的任务是：
根据用户传入的 JSON 列表，将多段数据分析结果汇总并撰写成一份结构化报告。

输出必须是一个 JSON 对象，结构如下：
{{
  "subject": "<报告总标题>",
  "sections": [
    {{
      "title": "<本段标题>",
      "insight": "<精准数据洞见：关键数值、差异、排序、异常、机会或风险>",
      "conclusion": "<结论或行动建议>",
      "chart_paths": ["<本段对应图表路径>"],
      "export_paths": ["<本段对应导出文件路径>"]
    }}
  ],
  "content_html": "<完整 HTML 报告内容>",
  "attachments": ["<附件路径1>", "<附件路径2>"]
}}

content_html 生成规则：
- 使用中文撰写。
- subject 必须由你根据所有分析结果提炼生成，要求专业、准确、能概括报告主题，不要使用“数据分析报告”这类空泛标题。
- 每个分析 JSON 对应一个 <h2> 小节。
- 每个 sections 元素的 conclusion 必须由你生成，不能为空；要给出基于数据的结论或行动建议。
- 每个 <h2> 小节必须包含“数据洞见”内容，要求精准、有效、可行动：明确指出关键数值、差异、排序、异常、机会或风险，不要写空泛套话。
- 如有核心指标，使用 <table> 展示。
- 正文中不要出现“对应图表”“附件”“文件路径”等字样，也不要把 chart_paths/export_paths 直接写进正文。
- 每个小节只写业务内容；图片由应用在正文中自动嵌入。
- 内容要专业、简洁、结构清晰。
- 不要输出 Markdown，只输出 HTML 字符串。
"""


REPORT_TITLE_CONCLUSION_PROMPT = """
你是一名资深数据分析报告编辑。请基于用户给出的分析结果和当前段落内容，生成更专业的报告标题、段落小标题和结论建议。

要求：
- 只改写 subject、每段 title、每段 conclusion。
- 不要改写 insight，不要删除 chart_paths/export_paths。
- subject 要能概括整份报告的核心主题，不要空泛。
- conclusion 要精准、有效、可行动，必须基于对应 insight 和原始分析结果。
- 不要写“详见图表”“见附件”“对应图片”等路径或文件说明。
- 返回严格 JSON，不要附加解释。

返回结构：
{{
  "subject": "<报告标题>",
  "sections": [
    {{
      "title": "<段落小标题>",
      "conclusion": "<结论建议>"
    }}
  ]
}}
"""


def ensure_directories() -> None:
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    OUTPUT_TABLE_DIR.mkdir(exist_ok=True)


def build_llm(api_key: str, temperature: float = 0) -> ChatOpenAI:
    return ChatOpenAI(
        api_key=api_key,
        model="deepseek-chat",
        base_url="https://api.deepseek.com",
        temperature=temperature,
        response_format={"type": "json_object"},
    )


@st.cache_data(show_spinner=False)
def load_dataframe(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def dataframe_signature(df: pd.DataFrame) -> str:
    parts = [f"{column}:{df[column].dtype}" for column in df.columns]
    return "|".join(parts)


def generate_field_labels(df: pd.DataFrame, api_key: str) -> dict[str, str]:
    llm = build_llm(api_key)
    sample_payload = {
        "columns": list(df.columns),
        "dtypes": {column: str(dtype) for column, dtype in df.dtypes.items()},
        "sample_rows": df.head(3).where(pd.notna(df.head(3)), None).to_dict(orient="records"),
    }
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", FIELD_LABEL_PROMPT),
            ("user", "字段信息如下：{field_info}\n请生成字段中文映射。"),
        ]
    )
    response = llm.invoke(prompt.invoke({"field_info": json.dumps(sample_payload, ensure_ascii=False)}))
    data = parse_json_output(response.content)
    labels = data.get("columns") or {}
    return {column: str(labels.get(column) or column) for column in df.columns}


def dataframe_profile_payload(df: pd.DataFrame, field_labels: dict[str, str] | None = None) -> dict:
    return {
        "columns": list(df.columns),
        "field_labels": field_labels or {},
        "dtypes": {column: str(dtype) for column, dtype in df.dtypes.items()},
        "sample_rows": df.head(5).where(pd.notna(df.head(5)), None).to_dict(orient="records"),
        "row_count": int(len(df)),
    }


def generate_default_queries(df: pd.DataFrame, field_labels: dict[str, str], api_key: str) -> list[str]:
    llm = build_llm(api_key)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", DEFAULT_ANALYSIS_PROMPT),
            ("user", "当前数据画像如下：{profile}\n请推荐 3 个默认分析问题。"),
        ]
    )
    response = llm.invoke(
        prompt.invoke({"profile": json.dumps(dataframe_profile_payload(df, field_labels), ensure_ascii=False)})
    )
    data = parse_json_output(response.content)
    queries = data.get("queries") or []
    return [str(query).strip() for query in queries if str(query).strip()][:3]


def get_field_labels(df: pd.DataFrame, api_key: str | None) -> dict[str, str]:
    signature = dataframe_signature(df)
    if (
        st.session_state.get("field_label_signature") == signature
        and isinstance(st.session_state.get("field_labels"), dict)
    ):
        return st.session_state.field_labels
    if not api_key:
        return {column: column for column in df.columns}

    with st.spinner("正在识别字段中文含义..."):
        labels = generate_field_labels(df, api_key)
    st.session_state.field_label_signature = signature
    st.session_state.field_labels = labels
    return labels


def parse_json_output(raw_output) -> dict:
    if isinstance(raw_output, dict):
        return raw_output
    if not raw_output:
        return {
            "type": "error",
            "input": "",
            "data": {"message": "模型没有返回内容", "missing_columns": [], "invalid_filters": {}, "suggestions": []},
            "chart_paths": [],
            "export_paths": [],
        }
    try:
        return json.loads(raw_output)
    except json.JSONDecodeError:
        start = raw_output.find("{")
        end = raw_output.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(raw_output[start : end + 1])
        raise


def create_dataframe_agent(csv_path: str, api_key: str):
    model = build_llm(api_key)
    df = load_dataframe(csv_path)
    return create_pandas_dataframe_agent(
        llm=model,
        df=df,
        agent_type="tool-calling",
        allow_dangerous_code=True,
        verbose=False,
    )


def missing_chart_paths(result: dict) -> list[str]:
    return [path for path in result.get("chart_paths") or [] if not normalize_path(path).exists()]


def generate_verified_chart_result(
    csv_path: str,
    api_key: str,
    field_labels: dict[str, str] | None,
    original_query: str,
    current_result: dict,
    feedback: str,
    filename_prefix: str,
) -> dict:
    ensure_directories()
    output_path = ARTIFACTS_DIR / f"{filename_prefix}_{int(time.time())}.png"
    df = load_dataframe(csv_path)
    payload = {
        "field_labels": field_labels or {},
        "data_profile": dataframe_profile_payload(df, field_labels),
        "original_query": original_query,
        "current_result": {key: value for key, value in current_result.items() if not key.startswith("_")},
        "feedback": feedback,
        "required_output_path": str(output_path),
    }
    llm = build_llm(api_key)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", CHART_CODE_PROMPT),
            ("user", "请生成可执行绘图代码，输入信息如下：{payload}"),
        ]
    )
    response = llm.invoke(prompt.invoke({"payload": json.dumps(payload, ensure_ascii=False)}))
    chart_spec = parse_json_output(response.content)
    code = chart_spec.get("code")
    if not code:
        raise RuntimeError("模型没有返回绘图代码。")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    chinese_font_prop = setup_chinese_font()
    if chinese_font_prop:
        code = code.replace('"SimHei"', "CHINESE_FONT_NAME").replace("'SimHei'", "CHINESE_FONT_NAME")
    namespace = {
        "df": df,
        "pd": pd,
        "plt": plt,
        "output_path": str(output_path),
        "ARTIFACTS_DIR": ARTIFACTS_DIR,
        "OUTPUT_TABLE_DIR": OUTPUT_TABLE_DIR,
        "FONT_PATH": FONT_PATH,
        "CHINESE_FONT_PROP": chinese_font_prop,
        "CHINESE_FONT_NAME": chinese_font_prop.get_name() if chinese_font_prop else "",
    }
    try:
        exec(code, {"__builtins__": __builtins__}, namespace)
    finally:
        plt.close("all")

    if not output_path.exists():
        raise RuntimeError(f"绘图代码执行完毕，但没有生成图片：{output_path}")

    result = {key: value for key, value in current_result.items() if not key.startswith("_")}
    data = result.get("data") if isinstance(result.get("data"), dict) else {}
    data["chart_type"] = chart_spec.get("chart_type") or data.get("chart_type") or "chart"
    if chart_spec.get("summary"):
        data["summary"] = chart_spec["summary"]
    result["type"] = "chart"
    result["data"] = data
    result["chart_paths"] = [str(output_path)]
    result["export_paths"] = unique_paths((result.get("export_paths") or []) + (chart_spec.get("export_paths") or []))
    return keep_one_chart(result)


def retry_missing_chart(agent, original_result: dict, user_query: str, missing_paths: list[str]) -> dict:
    retry_prompt = f"""
上一次分析已经完成，但返回的图表路径没有实际生成文件。

原始用户需求：
{user_query}

上一次返回的 JSON：
{json.dumps(original_result, ensure_ascii=False)}

未生成的图表路径：
{json.dumps(missing_paths, ensure_ascii=False)}

请只修复图表生成问题，并返回一个完整合法 JSON：
- 必须基于当前 df 重新选择 1 张最适合该问题的图表。
- 必须使用 matplotlib 真正执行绘图，并调用 plt.savefig(...) 保存图片。
- 图片必须保存到目录：{ARTIFACTS_DIR.as_posix()}/
- chart_paths 数组只能包含 1 个已经真实保存成功的图片路径。
- 如果需要导出支撑表格，保存到目录：{OUTPUT_TABLE_DIR.as_posix()}/
- 图表标题、坐标轴、图例、标注必须使用中文。
- 不要只编造路径；保存完成后再返回 JSON。
- 最终只返回 JSON，不要返回额外解释。
"""
    raw = agent.invoke({"input": retry_prompt})
    return keep_one_chart(parse_json_output(raw.get("output", "")))


def regenerate_chart_with_feedback(
    csv_path: str,
    api_key: str,
    field_labels: dict[str, str] | None,
    original_query: str,
    current_result: dict,
    feedback: str,
) -> dict:
    result = generate_verified_chart_result(
        csv_path=csv_path,
        api_key=api_key,
        field_labels=field_labels,
        original_query=original_query,
        current_result=current_result,
        feedback=feedback,
        filename_prefix="revise",
    )
    merged = merge_chart_update(current_result, result, feedback)
    merged["_original_query"] = original_query
    return merged


def data_analyze_agent(csv_path: str, user_query: str, api_key: str, field_labels: dict[str, str] | None = None) -> dict:
    ensure_directories()
    agent = create_dataframe_agent(csv_path, api_key)
    label_text = json.dumps(field_labels or {}, ensure_ascii=False)
    raw = agent.invoke({"input": PROMPT_TEMPLATE + "\n\n当前字段中文映射：" + label_text + "\n\n用户需求：" + user_query})
    result = keep_one_chart(parse_json_output(raw.get("output", "")))
    missing_paths = missing_chart_paths(result)
    if missing_paths:
        try:
            retry_result = generate_verified_chart_result(
                csv_path=csv_path,
                api_key=api_key,
                field_labels=field_labels,
                original_query=user_query,
                current_result=result,
                feedback=f"上一次返回了图表路径但文件没有生成：{missing_paths}。请修复图表并确保图片真实保存。",
                filename_prefix="repair",
            )
            if retry_result.get("chart_paths") and not missing_chart_paths(retry_result):
                return merge_chart_update(result, retry_result)
        except Exception as exc:
            result["_chart_retry_error"] = str(exc)
            result["_chart_retry_failed"] = True
    return result


def generate_report(analysis_log: list[dict], api_key: str) -> dict:
    llm = build_llm(api_key)
    report_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", REPORT_PROMPT),
            ("user", "输入的 JSON 列表如下：{json_list}\n请生成报告。"),
        ]
    )
    prompt_value = report_prompt.invoke({"json_list": json.dumps(analysis_log, ensure_ascii=False)})
    ret = llm.invoke(prompt_value)
    return parse_json_output(ret.content)


def rewrite_report_title_conclusions(
    subject: str,
    sections: list[dict],
    analysis_log: list[dict],
    api_key: str,
) -> tuple[str, list[dict]]:
    llm = build_llm(api_key)
    payload = {
        "current_subject": subject,
        "current_sections": [
            {
                "title": section.get("title", ""),
                "insight": section.get("insight", ""),
                "conclusion": section.get("conclusion", ""),
            }
            for section in sections
        ],
        "analysis_results": analysis_log,
    }
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", REPORT_TITLE_CONCLUSION_PROMPT),
            ("user", "当前报告信息如下：{payload}\n请生成报告标题和每段结论建议。"),
        ]
    )
    response = llm.invoke(prompt.invoke({"payload": json.dumps(payload, ensure_ascii=False)}))
    data = parse_json_output(response.content)
    next_subject = str(data.get("subject") or subject)
    generated_sections = data.get("sections") or []

    next_sections = []
    for index, section in enumerate(sections):
        generated = generated_sections[index] if index < len(generated_sections) and isinstance(generated_sections[index], dict) else {}
        updated = dict(section)
        updated["title"] = str(generated.get("title") or section.get("title") or f"分析 {index + 1}")
        updated["conclusion"] = str(generated.get("conclusion") or section.get("conclusion") or "")
        next_sections.append(updated)
    return next_subject, next_sections


def has_missing_conclusions(sections: list[dict]) -> bool:
    return any(not str(section.get("conclusion") or "").strip() for section in sections)


def sync_report_editor_state(subject: str, sections: list[dict]) -> None:
    st.session_state["report-subject"] = subject
    for index, section in enumerate(sections, start=1):
        st.session_state[f"report-section-title-{index}"] = str(section.get("title") or f"分析 {index}")
        st.session_state[f"report-section-insight-{index}"] = str(section.get("insight") or "")
        st.session_state[f"report-section-conclusion-{index}"] = str(section.get("conclusion") or "")


def queue_report_editor_state(subject: str, sections: list[dict]) -> None:
    st.session_state["pending_report_editor_state"] = {"subject": subject, "sections": sections}


def apply_pending_report_editor_state() -> None:
    pending = st.session_state.pop("pending_report_editor_state", None)
    if pending:
        sync_report_editor_state(pending.get("subject", "数据分析报告"), pending.get("sections", []))


def default_report_sections(analysis_log: list[dict]) -> list[dict]:
    sections = []
    for index, item in enumerate(analysis_log, start=1):
        data = item.get("data") or {}
        summary = ""
        if isinstance(data, dict):
            summary = data.get("summary") or data.get("answer") or data.get("message") or ""
        elif data:
            summary = str(data)
        sections.append(
            {
                "title": item.get("input") or f"分析 {index}",
                "insight": summary,
                "conclusion": "",
                "chart_paths": item.get("chart_paths") or [],
                "export_paths": item.get("export_paths") or [],
            }
        )
    return sections


def ensure_report_sections(report: dict, analysis_log: list[dict]) -> dict:
    sections = report.get("sections")
    if not isinstance(sections, list) or not sections:
        report["sections"] = default_report_sections(analysis_log)
        return report

    for index, section in enumerate(sections):
        if index < len(analysis_log):
            section.setdefault("chart_paths", analysis_log[index].get("chart_paths") or [])
            section.setdefault("export_paths", analysis_log[index].get("export_paths") or [])
        section.setdefault("title", f"分析 {index + 1}")
        section.setdefault("insight", "")
        section.setdefault("conclusion", "")
    return report


def image_data_uri(path_text: str) -> str | None:
    path = normalize_path(path_text)
    if not path.exists() or not path.is_file():
        return None
    mime_type = mimetypes.guess_type(str(path))[0] or "image/png"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{data}"


def build_report_html(subject: str, sections: list[dict]) -> str:
    parts = [f"<h1>{html.escape(subject)}</h1>"]
    for section in sections:
        title = html.escape(str(section.get("title") or "分析小节"))
        insight = html.escape(str(section.get("insight") or "")).replace("\n", "<br>")
        conclusion = html.escape(str(section.get("conclusion") or "")).replace("\n", "<br>")
        parts.append(f"<h2>{title}</h2>")
        if insight:
            parts.append("<h3>数据洞见</h3>")
            parts.append(f"<p>{insight}</p>")
        if conclusion:
            parts.append("<h3>结论建议</h3>")
            parts.append(f"<p>{conclusion}</p>")
        chart_paths = section.get("chart_paths") or []
        for chart_path in chart_paths[:1]:
            uri = image_data_uri(chart_path)
            if uri:
                parts.append(
                    '<p><img src="'
                    + uri
                    + '" style="max-width:100%;height:auto;margin:12px 0;" /></p>'
                )
    return "\n".join(parts)


def normalize_path(path_text: str) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = BASE_DIR / path
    return path


def unique_paths(paths: list[str]) -> list[str]:
    seen = set()
    result = []
    for path in paths:
        key = str(normalize_path(path)).lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(path)
    return result


def keep_one_chart(result: dict) -> dict:
    chart_paths = unique_paths(result.get("chart_paths") or [])
    result["chart_paths"] = chart_paths[:1]
    return result


def merge_chart_update(original_result: dict, chart_result: dict, feedback: str | None = None) -> dict:
    merged = dict(original_result)
    if chart_result.get("chart_paths"):
        merged["chart_paths"] = chart_result["chart_paths"][:1]

    export_paths = unique_paths((original_result.get("export_paths") or []) + (chart_result.get("export_paths") or []))
    merged["export_paths"] = export_paths

    original_data = original_result.get("data")
    chart_data = chart_result.get("data")
    if isinstance(original_data, dict):
        next_data = dict(original_data)
        if isinstance(chart_data, dict) and chart_data.get("chart_type"):
            next_data["chart_type"] = chart_data["chart_type"]
        merged["data"] = next_data
    elif isinstance(chart_data, dict):
        merged["data"] = {
            "chart_type": chart_data.get("chart_type", "chart"),
            "summary": str(original_data or chart_data.get("summary") or ""),
        }

    if feedback:
        merged["_chart_feedback"] = feedback
    return keep_one_chart(merged)


def label_for_column(column: str, field_labels: dict[str, str] | None = None) -> str:
    labels = field_labels or st.session_state.get("field_labels") or {}
    return labels.get(column, column)


def localize_dataframe_columns(df: pd.DataFrame, field_labels: dict[str, str] | None = None) -> pd.DataFrame:
    return df.rename(columns={column: label_for_column(column, field_labels) for column in df.columns})


def normalize_query_key(query: str) -> str:
    return "".join(str(query).lower().split())


def upsert_analysis_result(result: dict, query: str) -> None:
    result = keep_one_chart(result)
    result["_query_key"] = normalize_query_key(query)
    result["_original_query"] = query
    for index, item in enumerate(st.session_state.analysis_log):
        if item.get("_query_key") == result["_query_key"]:
            st.session_state.analysis_log[index] = result
            return
    st.session_state.analysis_log.append(result)


def public_analysis_log(analysis_log: list[dict]) -> list[dict]:
    return [{key: value for key, value in item.items() if not key.startswith("_")} for item in analysis_log]


def setup_chinese_font():
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    if FONT_PATH.exists():
        font_manager.fontManager.addfont(str(FONT_PATH))
        font_prop = font_manager.FontProperties(fname=str(FONT_PATH))
        font_name = font_prop.get_name()
        plt.rcParams["font.family"] = font_name
        plt.rcParams["font.sans-serif"] = [font_name]
        plt.rcParams["mathtext.fontset"] = "custom"
        plt.rcParams["mathtext.rm"] = font_name
        plt.rcParams["mathtext.it"] = font_name
        plt.rcParams["mathtext.bf"] = font_name
        plt.rcParams["axes.unicode_minus"] = False
        return font_prop
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    return None


def collect_attachments(report: dict, analysis_log: list[dict]) -> list[str]:
    attachments = []
    for item in analysis_log:
        attachments.extend(item.get("chart_paths") or [])
        attachments.extend(item.get("export_paths") or [])
    attachments.extend(report.get("attachments") or [])
    seen = set()
    result = []
    for attachment in attachments:
        if not attachment or attachment in seen:
            continue
        seen.add(attachment)
        result.append(attachment)
    return result


def render_analysis_result(
    result: dict,
    index: int | None = None,
    csv_path: str | None = None,
    api_key: str | None = None,
    field_labels: dict[str, str] | None = None,
) -> bool:
    title_prefix = f"分析 {index}" if index is not None else "分析结果"
    result_type = result.get("type", "answer")
    user_input = result.get("input")
    data = result.get("data") or {}

    title_col, delete_col = st.columns([0.92, 0.08], vertical_alignment="center")
    with title_col:
        st.subheader(title_prefix)
    with delete_col:
        delete_key = f"delete-analysis-{result.get('_query_key', index)}"
        if st.button("🗑️", key=delete_key, help="删除这条分析", type="secondary"):
            return True

    if user_input:
        st.caption(user_input)
    if result.get("_chart_retry_failed"):
        st.warning("已自动重试生成图表，但模型仍未成功保存图片文件。请尝试重新运行，或把问题描述得更具体一些。")
        if result.get("_chart_retry_error"):
            with st.expander("查看图表重试错误"):
                st.code(result["_chart_retry_error"])

    if result_type == "error":
        message = data.get("message", "分析失败")
        st.error(message)
        suggestions = data.get("suggestions") or []
        if suggestions:
            st.markdown("**建议：**")
            for suggestion in suggestions:
                st.write(f"- {suggestion}")
    elif result_type == "table":
        columns = data.get("columns") or []
        rows = data.get("rows") or []
        if columns and rows:
            localized_columns = [label_for_column(column) for column in columns]
            st.dataframe(pd.DataFrame(rows, columns=localized_columns), use_container_width=True)
        else:
            st.info("模型返回了表格类型，但没有可展示的行列数据。")
    elif result_type == "chart":
        summary = data.get("summary")
        if summary:
            st.markdown(summary)
    else:
        answer = data.get("answer") if isinstance(data, dict) else data
        st.markdown(str(answer or "模型没有返回可展示的文字。"))

    chart_paths = result.get("chart_paths") or []
    for chart_path in chart_paths:
        local_path = normalize_path(chart_path)
        if local_path.exists():
            st.image(str(local_path), caption=chart_path, use_container_width=True)
        else:
            st.warning(f"模型返回了图表路径，但未生成对应文件：{chart_path}")

        feedback_key = f"chart-feedback-{result.get('_query_key', index)}"
        button_key = f"chart-regenerate-{result.get('_query_key', index)}"
        feedback = st.text_area(
            "图表修改要求",
            placeholder="例如：柱形图排序不对；占比口径应使用 purchase_total；不要用折线图，改成按类别排序的横向柱形图。",
            key=feedback_key,
            height=80,
        )
        if st.button("按反馈重新生成图表", key=button_key):
            if not api_key or not csv_path:
                st.warning("请先在侧边栏填写 DeepSeek API Key。")
            elif not feedback.strip():
                st.warning("请先写明图表哪里不准确，或希望如何修改。")
            else:
                with st.spinner("正在按反馈重新生成图表..."):
                    try:
                        updated = regenerate_chart_with_feedback(
                            csv_path=csv_path,
                            api_key=api_key,
                            field_labels=field_labels,
                            original_query=result.get("_original_query") or result.get("input") or "",
                            current_result=result,
                            feedback=feedback.strip(),
                        )
                        updated["_query_key"] = result.get("_query_key", normalize_query_key(updated.get("input", "")))
                        updated["_original_query"] = result.get("_original_query") or result.get("input") or ""
                        if index is not None:
                            st.session_state.analysis_log[index - 1] = updated
                        st.session_state.report = None
                        st.session_state.report_content_html = ""
                        st.session_state.report_sections = []
                        st.rerun()
                    except Exception as exc:
                        st.error(f"重新生成图表失败：{exc}")

    export_paths = result.get("export_paths") or []
    if export_paths:
        st.markdown("**导出文件**")
        for export_path in export_paths:
            local_path = normalize_path(export_path)
            if local_path.exists():
                with local_path.open("rb") as file:
                    st.download_button(
                        label=f"下载 {Path(export_path).name}",
                        data=file,
                        file_name=Path(export_path).name,
                        mime="text/csv",
                        key=f"download-{title_prefix}-{export_path}",
                    )
            else:
                st.write(f"- {export_path}")

    with st.expander("查看原始 JSON"):
        st.json({key: value for key, value in result.items() if not key.startswith("_")})

    return False


def send_report_email(
    sender_email: str,
    auth_code: str,
    recipient_email: str,
    report: dict,
    attachments: list[str],
) -> None:
    if zmail is None:
        raise RuntimeError("当前环境没有安装 zmail，请先运行：pip install zmail")

    mail_content = {
        "subject": report.get("subject", "数据分析报告"),
        "content_html": report.get("content_html", ""),
        "attachments": [str(normalize_path(item)) for item in attachments if normalize_path(item).exists()],
    }
    server = zmail.server(sender_email, auth_code)
    server.send_mail(recipient_email, mail_content)


def main() -> None:
    st.set_page_config(page_title=" 数据分析助手", layout="wide")
    ensure_directories()

    st.title(" 数据分析助手")
    st.caption("基于课程 CSV 数据，通过 DeepSeek + Pandas Agent 生成分析、图表、报告，并支持手动发送邮件。")

    if "analysis_log" not in st.session_state:
        st.session_state.analysis_log = []
    if "report" not in st.session_state:
        st.session_state.report = None
    if "report_content_html" not in st.session_state:
        st.session_state.report_content_html = ""
    if "report_sections" not in st.session_state:
        st.session_state.report_sections = []
    if "default_queries_signature" not in st.session_state:
        st.session_state.default_queries_signature = ""
    if "default_queries" not in st.session_state:
        st.session_state.default_queries = []
    if "active_uploaded_csv_hash" not in st.session_state:
        st.session_state.active_uploaded_csv_hash = ""

    with st.sidebar:
        st.header("配置")
        secret_api_key = get_secret_value("DEEPSEEK_API_KEY", "deepseek.api_key", "deepseek_api_key")
        manual_api_key = st.text_input(
            "DeepSeek API Key",
            type="password",
            help="优先读取 st.secrets；这里手动填写时会覆盖 secrets。",
            placeholder="已从 st.secrets 读取" if secret_api_key else "",
        )
        api_key = manual_api_key.strip() or secret_api_key
        if secret_api_key and not manual_api_key:
            st.success("已从 st.secrets 读取 DeepSeek API Key")
        csv_path = st.text_input("CSV 文件路径", value=str(DEFAULT_CSV_PATH))
        uploaded_file = st.file_uploader("或上传 CSV", type=["csv"])
        if uploaded_file is not None:
            uploaded_bytes = uploaded_file.getvalue()
            uploaded_hash = hashlib.md5(uploaded_bytes).hexdigest()[:12]
            safe_name = Path(uploaded_file.name).name
            upload_path = BASE_DIR / "data" / f"uploaded_{uploaded_hash}_{safe_name}"
            upload_path.parent.mkdir(exist_ok=True)
            if not upload_path.exists():
                upload_path.write_bytes(uploaded_bytes)
            csv_path = str(upload_path)
            if st.session_state.active_uploaded_csv_hash != uploaded_hash:
                load_dataframe.clear()
                st.session_state.active_uploaded_csv_hash = uploaded_hash
                st.session_state.analysis_log = []
                st.session_state.report = None
                st.session_state.report_content_html = ""
                st.session_state.report_sections = []
                st.session_state.default_queries_signature = ""
                st.session_state.default_queries = []
                st.session_state.field_label_signature = ""
                st.session_state.field_labels = {}
            st.success("已使用上传的 CSV 文件")

        st.divider()
        if st.button("清空本次分析结果", use_container_width=True):
            st.session_state.analysis_log = []
            st.session_state.report = None
            st.session_state.report_content_html = ""
            st.session_state.report_sections = []
            st.rerun()

    csv_file = Path(csv_path)
    if not csv_file.exists():
        st.error(f"找不到 CSV 文件：{csv_path}")
        return

    try:
        df = load_dataframe(str(csv_file))
    except Exception as exc:
        st.error(f"读取 CSV 失败：{exc}")
        return
    field_labels = get_field_labels(df, api_key)
    current_signature = dataframe_signature(df)
    if st.session_state.default_queries_signature and st.session_state.default_queries_signature != current_signature:
        st.session_state.default_queries_signature = ""
        st.session_state.default_queries = []

    preview_tab, analyze_tab, report_tab, email_tab = st.tabs(["数据预览", "分析", "综合报告", "发送邮件"])

    with preview_tab:
        left, right, third = st.columns(3)
        left.metric("行数", f"{df.shape[0]:,}")
        right.metric("列数", f"{df.shape[1]:,}")
        third.metric("文件", csv_file.name)
        st.dataframe(localize_dataframe_columns(df.head(100), field_labels), use_container_width=True)
        with st.expander("查看字段"):
            st.write([f"{label_for_column(column, field_labels)}（{column}）" for column in df.columns])

    with analyze_tab:
        st.subheader("自定义分析")
        user_query = st.text_area(
            "输入你的分析问题",
            value="按 category 汇总 revenue_course、purchase_total 和退款数，分析哪类课程商业表现最好，并生成图表。",
            height=120,
        )
        col_a, col_b = st.columns([1, 1])
        with col_a:
            run_custom = st.button("运行自定义分析", type="primary", use_container_width=True)
        with col_b:
            run_defaults = st.button("AI 生成并运行 3 项默认分析", use_container_width=True)

        if st.session_state.default_queries:
            with st.expander("查看当前 AI 默认分析问题"):
                for idx, query in enumerate(st.session_state.default_queries, start=1):
                    st.write(f"{idx}. {query}")
                if st.button("重新生成默认分析问题"):
                    st.session_state.default_queries_signature = ""
                    st.session_state.default_queries = []
                    st.rerun()

        if run_custom:
            if not api_key:
                st.warning("请先在侧边栏填写 DeepSeek API Key。")
            elif not user_query.strip():
                st.warning("请先输入分析问题。")
            else:
                with st.spinner("正在分析，请稍等..."):
                    try:
                        result = data_analyze_agent(str(csv_file), user_query.strip(), api_key, field_labels)
                        upsert_analysis_result(result, user_query.strip())
                        st.session_state.report = None
                        st.session_state.report_content_html = ""
                        st.session_state.report_sections = []
                        st.success("分析完成")
                    except Exception as exc:
                        st.error(f"分析失败：{exc}")

        if run_defaults:
            if not api_key:
                st.warning("请先在侧边栏填写 DeepSeek API Key。")
            else:
                signature = dataframe_signature(df)
                if st.session_state.default_queries_signature != signature or not st.session_state.default_queries:
                    with st.spinner("正在根据当前数据生成默认分析问题..."):
                        st.session_state.default_queries = generate_default_queries(df, field_labels, api_key)
                        st.session_state.default_queries_signature = signature

                default_queries = st.session_state.default_queries
                if not default_queries:
                    st.error("模型没有生成默认分析问题，请稍后重试或使用自定义分析。")
                    st.stop()

                with st.expander("本次 AI 默认分析问题", expanded=True):
                    for idx, query in enumerate(default_queries, start=1):
                        st.write(f"{idx}. {query}")

                progress = st.progress(0)
                for idx, query in enumerate(default_queries, start=1):
                    with st.spinner(f"正在运行 AI 默认分析 {idx}/{len(default_queries)}..."):
                        try:
                            result = data_analyze_agent(str(csv_file), query, api_key, field_labels)
                            upsert_analysis_result(result, query)
                            st.session_state.report = None
                            st.session_state.report_content_html = ""
                            st.session_state.report_sections = []
                        except Exception as exc:
                            upsert_analysis_result(
                                {
                                    "type": "error",
                                    "input": query,
                                    "data": {
                                        "message": str(exc),
                                        "missing_columns": [],
                                        "invalid_filters": {},
                                        "suggestions": ["检查 API Key、网络连接、模型输出格式或数据字段。"],
                                    },
                                    "chart_paths": [],
                                    "export_paths": [],
                                },
                                query,
                            )
                        progress.progress(idx / len(default_queries))
                st.success("AI 默认分析流程结束")

        st.divider()
        if st.session_state.analysis_log:
            delete_index = None
            for idx, item in enumerate(st.session_state.analysis_log, start=1):
                with st.container(border=True):
                    if render_analysis_result(item, idx, str(csv_file), api_key, field_labels):
                        delete_index = idx - 1
                        break
            if delete_index is not None:
                del st.session_state.analysis_log[delete_index]
                st.session_state.report = None
                st.session_state.report_content_html = ""
                st.session_state.report_sections = []
                st.rerun()
        else:
            st.info("还没有分析结果。")

    with report_tab:
        st.subheader("综合报告")
        if not st.session_state.analysis_log:
            st.info("请先在“分析”页生成至少一个分析结果。")
        else:
            if st.button("生成综合报告", type="primary"):
                if not api_key:
                    st.warning("请先在侧边栏填写 DeepSeek API Key。")
                else:
                    with st.spinner("正在生成综合报告..."):
                        try:
                            st.session_state.report = generate_report(public_analysis_log(st.session_state.analysis_log), api_key)
                            st.session_state.report = ensure_report_sections(
                                st.session_state.report,
                                public_analysis_log(st.session_state.analysis_log),
                            )
                            st.session_state.report_sections = st.session_state.report.get("sections", [])
                            if has_missing_conclusions(st.session_state.report_sections):
                                next_subject, next_sections = rewrite_report_title_conclusions(
                                    st.session_state.report.get("subject", "数据分析报告"),
                                    st.session_state.report_sections,
                                    public_analysis_log(st.session_state.analysis_log),
                                    api_key,
                                )
                                st.session_state.report["subject"] = next_subject
                                st.session_state.report["sections"] = next_sections
                                st.session_state.report_sections = next_sections
                            st.session_state.report_content_html = build_report_html(
                                st.session_state.report.get("subject", "数据分析报告"),
                                st.session_state.report_sections,
                            )
                            st.session_state.report["content_html"] = st.session_state.report_content_html
                            sync_report_editor_state(
                                st.session_state.report.get("subject", "数据分析报告"),
                                st.session_state.report_sections,
                            )
                            st.success("报告生成完成")
                        except Exception as exc:
                            st.error(f"生成报告失败：{exc}")

            report = st.session_state.report
            if report:
                apply_pending_report_editor_state()
                if not st.session_state.report_sections:
                    report = ensure_report_sections(report, public_analysis_log(st.session_state.analysis_log))
                    st.session_state.report_sections = report.get("sections", [])

                if "report-subject" not in st.session_state:
                    sync_report_editor_state(report.get("subject", "数据分析报告"), st.session_state.report_sections)

                subject = st.text_input("报告标题", key="report-subject")
                edited_sections = []
                for idx, section in enumerate(st.session_state.report_sections, start=1):
                    with st.container(border=True):
                        st.markdown(f"#### 分析段落 {idx}")
                        st.session_state.setdefault(
                            f"report-section-title-{idx}",
                            str(section.get("title") or f"分析 {idx}"),
                        )
                        st.session_state.setdefault(
                            f"report-section-insight-{idx}",
                            str(section.get("insight") or ""),
                        )
                        st.session_state.setdefault(
                            f"report-section-conclusion-{idx}",
                            str(section.get("conclusion") or ""),
                        )
                        title = st.text_input(
                            "小标题",
                            key=f"report-section-title-{idx}",
                        )
                        insight = st.text_area(
                            "数据洞见",
                            height=120,
                            key=f"report-section-insight-{idx}",
                        )
                        conclusion = st.text_area(
                            "结论建议",
                            height=90,
                            key=f"report-section-conclusion-{idx}",
                        )
                        chart_paths = section.get("chart_paths") or []
                        if chart_paths:
                            st.markdown("**图片预览**")
                            for chart_path in chart_paths[:1]:
                                path = normalize_path(chart_path)
                                if path.exists():
                                    st.image(str(path), use_container_width=True)
                                else:
                                    st.warning("图片未生成")
                        edited_sections.append(
                            {
                                "title": title,
                                "insight": insight,
                                "conclusion": conclusion,
                                "chart_paths": chart_paths,
                                "export_paths": section.get("export_paths") or [],
                            }
                        )

                st.session_state.report_sections = edited_sections
                st.session_state.report["subject"] = subject
                st.session_state.report["sections"] = edited_sections
                st.session_state.report_content_html = build_report_html(subject, edited_sections)
                st.session_state.report["content_html"] = st.session_state.report_content_html

                if st.button("AI 生成标题和结论建议", type="secondary"):
                    if not api_key:
                        st.warning("请先在侧边栏填写 DeepSeek API Key。")
                    else:
                        with st.spinner("正在生成报告标题和结论建议..."):
                            try:
                                next_subject, next_sections = rewrite_report_title_conclusions(
                                    subject,
                                    edited_sections,
                                    public_analysis_log(st.session_state.analysis_log),
                                    api_key,
                                )
                                st.session_state.report["subject"] = next_subject
                                st.session_state.report["sections"] = next_sections
                                st.session_state.report_sections = next_sections
                                st.session_state.report_content_html = build_report_html(next_subject, next_sections)
                                st.session_state.report["content_html"] = st.session_state.report_content_html
                                queue_report_editor_state(next_subject, next_sections)
                                st.rerun()
                            except Exception as exc:
                                st.error(f"生成标题和结论建议失败：{exc}")

                st.markdown("#### 报告预览")
                st.markdown(st.session_state.report_content_html, unsafe_allow_html=True)
                attachments = collect_attachments(report, st.session_state.analysis_log)
                if attachments:
                    with st.expander("下载文件"):
                        for attachment in attachments:
                            path = normalize_path(attachment)
                            if path.exists() and path.is_file():
                                with path.open("rb") as file:
                                    st.download_button(
                                        label=f"下载 {path.name}",
                                        data=file,
                                        file_name=path.name,
                                        key=f"report-download-{attachment}",
                                    )
                with st.expander("查看报告 JSON"):
                    st.json(report)

    with email_tab:
        st.subheader("发送邮件")
        st.caption("邮件不会自动发送。请先生成综合报告，再填写邮箱信息并点击发送。")

        if st.session_state.report is None:
            st.info("请先到“综合报告”页生成报告。")
        else:
            secret_sender_email = get_secret_value("EMAIL_SENDER", "email.sender", "sender_email")
            secret_auth_code = get_secret_value("EMAIL_AUTH_CODE", "email.auth_code", "email_auth_code")
            sender_email = st.text_input("发件邮箱", value=secret_sender_email)
            manual_auth_code = st.text_input(
                "邮箱授权码",
                type="password",
                help="优先读取 st.secrets；这里手动填写时会覆盖 secrets。",
                placeholder="已从 st.secrets 读取" if secret_auth_code else "",
            )
            auth_code = manual_auth_code.strip() or secret_auth_code
            if secret_auth_code and not manual_auth_code:
                st.success("已从 st.secrets 读取邮箱授权码")
            recipient_email = st.text_input("收件邮箱")
            attachments = collect_attachments(st.session_state.report, st.session_state.analysis_log)

            if attachments:
                with st.expander("将随邮件附带的文件"):
                    for item in attachments:
                        st.write(f"- {item}")

            if st.button("发送报告邮件", type="primary"):
                if not sender_email or not auth_code or not recipient_email:
                    st.warning("请完整填写发件邮箱、授权码和收件邮箱。")
                else:
                    with st.spinner("正在发送邮件..."):
                        try:
                            send_report_email(
                                sender_email=sender_email,
                                auth_code=auth_code,
                                recipient_email=recipient_email,
                                report=st.session_state.report,
                                attachments=attachments,
                            )
                            st.success("邮件发送成功")
                        except Exception as exc:
                            st.error(f"邮件发送失败：{exc}")


if __name__ == "__main__":
    main()
