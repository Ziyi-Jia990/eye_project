# import fitz  # PyMuPDF
# from PIL import Image
# import os

# # =========================
# # 配置区
# # =========================
# PDF_PATH = "example1.pdf"          # 原始 PDF 路径
# PAGE_INDEX = 0                   # 第几页，第一页就是 0

# # 这个区域是按“页面比例”定义的
# # 根据你给的图，绿色内容框大约在页面底部
# # 你可以微调这几个值
# REGION_RATIO = {
#     "x0": 0.02,   # 左边界占页面宽度比例
#     "y0": 0.60,   # 上边界占页面高度比例
#     "x1": 0.98,   # 右边界占页面宽度比例
#     "y1": 0.88    # 下边界占页面高度比例
# }

# # 是否启用 OCR 兜底
# USE_OCR_FALLBACK = True

# # 如果你是 Windows，并且 pytesseract 找不到 tesseract.exe，就取消下面注释并改成你的路径
# # import pytesseract
# # pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# # =========================
# # 工具函数
# # =========================
# def get_clip_rect(page, ratio_dict):
#     """把相对比例区域转成 PDF 页面上的绝对坐标区域"""
#     rect = page.rect
#     x0 = rect.x0 + rect.width * ratio_dict["x0"]
#     y0 = rect.y0 + rect.height * ratio_dict["y0"]
#     x1 = rect.x0 + rect.width * ratio_dict["x1"]
#     y1 = rect.y0 + rect.height * ratio_dict["y1"]
#     return fitz.Rect(x0, y0, x1, y1)


# def extract_text_from_pdf_region(pdf_path, page_index, ratio_dict):
#     """优先直接从 PDF 文本层提取指定区域内的文字"""
#     doc = fitz.open(pdf_path)
#     page = doc[page_index]
#     clip = get_clip_rect(page, ratio_dict)

#     # 方法1：直接提取该区域文本
#     text = page.get_text("text", clip=clip).strip()

#     # 方法2：如果 text 太少，再尝试按 blocks / words 提取
#     if not text:
#         words = page.get_text("words")  # [(x0, y0, x1, y1, word, block_no, line_no, word_no), ...]
#         selected_words = []
#         for w in words:
#             wx0, wy0, wx1, wy1, word = w[:5]
#             word_rect = fitz.Rect(wx0, wy0, wx1, wy1)
#             if clip.intersects(word_rect):
#                 selected_words.append((wy0, wx0, word))

#         # 按行、列排序
#         selected_words.sort()
#         text = " ".join([w[2] for w in selected_words]).strip()

#     doc.close()
#     return text


# def render_region_to_image(pdf_path, page_index, ratio_dict, zoom=3):
#     """把指定区域渲染成图片，供 OCR 使用"""
#     doc = fitz.open(pdf_path)
#     page = doc[page_index]
#     clip = get_clip_rect(page, ratio_dict)

#     matrix = fitz.Matrix(zoom, zoom)
#     pix = page.get_pixmap(matrix=matrix, clip=clip, alpha=False)

#     img_path = "cropped_region.png"
#     pix.save(img_path)
#     doc.close()
#     return img_path


# def ocr_image(image_path, lang="chi_sim+eng"):
#     """对图片做 OCR"""
#     import pytesseract

#     img = Image.open(image_path)
#     text = pytesseract.image_to_string(img, lang=lang)
#     return text.strip()


# # =========================
# # 主逻辑
# # =========================
# def main():
#     if not os.path.exists(PDF_PATH):
#         print(f"找不到 PDF 文件: {PDF_PATH}")
#         return

#     print("1) 先尝试从 PDF 文本层直接提取...")
#     text = extract_text_from_pdf_region(PDF_PATH, PAGE_INDEX, REGION_RATIO)

#     if text:
#         print("\n===== 提取结果（PDF 文本层）=====")
#         print(text)
#         return

#     print("PDF 文本层提取失败或区域内没有可读文本。")

#     if USE_OCR_FALLBACK:
#         print("\n2) 开始 OCR 识别...")
#         img_path = render_region_to_image(PDF_PATH, PAGE_INDEX, REGION_RATIO, zoom=4)
#         text = ocr_image(img_path)

#         print("\n===== 提取结果（OCR）=====")
#         print(text if text else "OCR 也没有识别到内容")
#     else:
#         print("未启用 OCR 兜底。")


# if __name__ == "__main__":
#     main()

import paddle
print(paddle.utils.run_check())