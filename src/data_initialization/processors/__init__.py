# src/data_initialization/processors/__init__.py

"""
数据初始化处理器模块

导入 settings.py 中定义的常量（与管线步骤、parse_stage 对应）：
- STAGE_LAYOUT_JSON_PARSED: layout_json_parser 完成
- STAGE_FRAGMENT_MERGED: json_fragment_merger 完成
- STAGE_REGION_DIVIDED: region_extractor 完成
- STAGE_METADATA_EXTRACTED: metadata_extractor 完成
- STAGE_IMAGE_DESCRIPTION: imagedescription_from_json 完成
- STAGE_RAG_EMBEDDING: rag 嵌入完成
"""

from src.config.settings import (
    STAGE_LAYOUT_JSON_PARSED,
    STAGE_FRAGMENT_MERGED,
    STAGE_REGION_DIVIDED,
    STAGE_METADATA_EXTRACTED,
    STAGE_IMAGE_DESCRIPTION,
    STAGE_RAG_EMBEDDING,
    PROCESS_STAGES,
)


def update_parse_stage(json_path: str, new_stage: str) -> bool:
    """
    更新 JSON 文件的 parse_stage 字段

    Args:
        json_path: JSON 文件路径
        new_stage: 新的处理阶段

    Returns:
        是否更新成功
    """
    import json
    from pathlib import Path

    path = Path(json_path)
    if not path.exists():
        return False

    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        data["metadata"]["parse_stage"] = new_stage

        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        return True
    except Exception:
        return False


def get_parse_stage(json_path: str) -> str:
    """
    获取 JSON 文件的 parse_stage 字段

    Args:
        json_path: JSON 文件路径

    Returns:
        当前的 parse_stage，默认为 STAGE_LAYOUT_JSON_PARSED
    """
    import json
    from pathlib import Path

    path = Path(json_path)
    if not path.exists():
        return STAGE_LAYOUT_JSON_PARSED

    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("metadata", {}).get("parse_stage", STAGE_LAYOUT_JSON_PARSED)
    except Exception:
        return STAGE_LAYOUT_JSON_PARSED


def is_stage_completed(json_path: str, target_stage: str) -> bool:
    """
    检查是否已完成指定处理阶段

    逻辑：如果目标阶段在当前阶段之前（或相等），说明已完成
    例如：current=image_description(3), target=metadata_extracted(1)
         → 3 >= 1 → True（已完成）

    Args:
        json_path: JSON 文件路径
        target_stage: 目标处理阶段

    Returns:
        是否已完成该阶段
    """
    current_stage = get_parse_stage(json_path)
    current_index = (
        PROCESS_STAGES.index(current_stage) if current_stage in PROCESS_STAGES else -1
    )
    target_index = (
        PROCESS_STAGES.index(target_stage) if target_stage in PROCESS_STAGES else -1
    )
    # 如果 current >= target，说明已经经过了 target 阶段
    return current_index >= target_index


def should_skip_stage(
    json_path: str, target_stage: str, skip_existing: bool = True
) -> bool:
    """
    判断是否应该跳过指定处理阶段

    Args:
        json_path: JSON 文件路径
        target_stage: 目标处理阶段
        skip_existing: 是否跳过已处理的文件

    Returns:
        是否应该跳过
    """
    if not skip_existing:
        return False
    return is_stage_completed(json_path, target_stage)
