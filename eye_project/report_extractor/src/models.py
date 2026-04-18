from __future__ import annotations
from typing import Any, Optional, Literal
from pydantic import BaseModel, Field


class BBox(BaseModel):
    x0: float = 0.0
    y0: float = 0.0
    x1: float = 0.0
    y1: float = 0.0


class TextBlock(BaseModel):
    page: int
    order: int
    text: str
    block_type: str = "unknown"   # title / paragraph / caption / unknown
    bbox: BBox = Field(default_factory=BBox)
    raw: dict[str, Any] = Field(default_factory=dict)


class TableBlock(BaseModel):
    page: int
    order: int
    title: Optional[str] = None
    table_type: Optional[str] = None   # 出血渗出 / 血管参数 / ...
    headers: list[list[str]] = Field(default_factory=list)
    rows: list[list[str]] = Field(default_factory=list)
    bbox: BBox = Field(default_factory=BBox)
    raw: dict[str, Any] = Field(default_factory=dict)


class ParsedDocument(BaseModel):
    file_name: str
    parser: Literal["monkeyocr"] = "monkeyocr"
    markdown_text: str = ""
    page_count: int = 0
    text_blocks: list[TextBlock] = Field(default_factory=list)
    table_blocks: list[TableBlock] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)