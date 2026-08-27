# OCR 增强说明

## 现在的行为

`ReadPaperTool` 仍然优先读取 PDF 自带的文字层。只有当某一页没有可提取的
文字时，才会通过 `scholaragent.ocr.TesseractOCR` 调用本机 OCR：

1. `pdftoppm` 把指定 PDF 页渲染为临时 PNG；
2. Tesseract 按页识别文字；
3. 临时图片自动清理，识别结果回到原有的分段阅读流程；
4. 阅读文本增加 `[OCR]` 标记，页码锚点使用 `pdf-page:N:ocr`，置信度为
   `medium`，提醒后续回答不要把 OCR 结果当成无误的原文转录。

OCR 失败不会让 Agent 崩溃。结果会保留“本页没有可提取文字”和失败诊断，
这样模型和用户都能区分“确实没有文字”和“本机没有 OCR 依赖”。同一运行内的
同一页会缓存 OCR 结果，避免续读时重复调用外部进程。

## 本机配置

你这台机器已检测到：

- `E:\Tesseract\tesseract.exe`，Tesseract 5.4.0；
- `E:\Tesseract\tessdata\chi_sim.traineddata` 和 `eng.traineddata`；
- 可用的 `pdftoppm` PDF 渲染器。

项目默认会从 PATH、项目所在磁盘的 `Tesseract` 目录和 Windows 常见安装目录
发现命令。你当前机器的实际路径是 `E:\Tesseract\tesseract.exe`；比赛或答辩电脑
建议按现场安装位置显式配置 `.env`，不要依赖机器目录恰好一致：

```dotenv
SCHOLARAGENT_TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
SCHOLARAGENT_PDF_RENDERER=pdftoppm
SCHOLARAGENT_OCR_LANGUAGE=chi_sim+eng
SCHOLARAGENT_OCR_DPI=200
SCHOLARAGENT_OCR_TIMEOUT=120
SCHOLARAGENT_OCR_PSM=3
```

网页工作台的状态接口会显示 OCR 是否可用、发现到的命令、语言和分辨率。
没有安装 OCR 时，系统仍可阅读普通文字层 PDF，扫描页则按原有规则如实降级。

## 识别边界

OCR 解决的是“扫描页没有文字层”的输入问题，不等于完成版面理解：

- 复杂双栏、低清晰度、倾斜页面和中英文混排可能出现识别错误；
- 表格、图片中的文字、图表语义和数学公式仍不能仅凭普通 OCR 保证准确；
- OCR 文本适合检索和辅助总结，关键数字、公式和结论仍应回看原始页面；
- 图表解析、公式解析和自动生成完整论文卡片仍属于后续增强，不在本适配器中
  冒充“已完成”。

## 许可证与比赛使用

本仓库只调用本机已安装的外部程序，不重新分发 Tesseract 二进制。Tesseract
本身采用 Apache-2.0，Leptonica 采用 BSD-2-Clause；如果以后把它们随安装包或
比赛提交物一起分发，应同时保留对应版权、许可证和第三方声明，并逐项检查
`tessdata` 语言模型的许可证。仅在比赛现场使用已安装的工具时，通常不改变本
项目自身代码的许可证，但仍应遵守比赛对第三方依赖和离线环境的要求。

## 验证

```powershell
.venv\Scripts\python -m pytest tests\test_ocr.py -q
```

测试覆盖无文字页的 OCR 接口回退、页码证据锚点和缺少外部依赖时的可解释降级。
