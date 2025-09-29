# 📄 MCP PDF Server

A PDF file reading server based on [FastMCP](https://github.com/minimaxir/fastmcp).

Supports PDF text extraction, OCR recognition, and image extraction via the MCP protocol, with a built-in web debugger for easy testing.

---

## 🚀 Features

- **read_pdf_text**  
  Extracts normal text from a PDF (page by page).

- **read_by_ocr**  
  Uses OCR to recognize text from scanned or image-based PDFs.

- **read_pdf_images**  
  Extracts all images from a specified PDF page (Base64 encoded output).

---

## 📂 Project Structure

```
mcp-pdf-server/
├── pdf_resources/        # Directory for uploaded and processed PDF files
├── txt_server.py         # Main server entry point
└── README.md             # Project documentation
```

---

## ⚙️ Installation

Recommended Python version: 3.9+

```bash
pip install pymupdf mcp
```

> Note: To use OCR features, you may need a MuPDF build with OCR support or external OCR libraries.

---

## 🔦 Start the Server

Run the following command:

```bash
python txt_server.py
```

You should see logs like:

```
Serving on http://127.0.0.1:6231
```

---

## 🌐 Web Debugging Interface

Open your browser and visit:

```
http://127.0.0.1:6231
```

- Select a tool from the left panel
- Fill in parameters on the right panel
- Click "Run" to test the tool

No coding required — easily debug and test via the web UI.

---

## 🛠️ API Tool List

| Tool | Description | Input Parameters | Returns |
|:-----|:------------|:-----------------|:--------|
| `read_pdf_text` | Extracts normal text from PDF pages | `file_path`, `start_page`, `end_page` | List of page texts |
| `read_by_ocr` | Recognizes text via OCR | `file_path`, `start_page`, `end_page`, `language`, `dpi` | OCR extracted text |
| `read_pdf_images` | Extracts images from a PDF page | `file_path`, `page_number` | List of images (Base64 encoded) |

---

## 📝 Example Usage

Extract text from pages 1 to 5:

```bash
mcp run read_pdf_text --args '{"file_path": "pdf_resources/example.pdf", "start_page": 1, "end_page": 5}'
```

Perform OCR recognition on page 1:

```bash
mcp run read_by_ocr --args '{"file_path": "pdf_resources/example.pdf", "start_page": 1, "end_page": 1, "language": "eng"}'
```

Extract all images from page 3:

```bash
mcp run read_pdf_images --args '{"file_path": "pdf_resources/example.pdf", "page_number": 3}'
```

---

## 📢 Notes

- Files must be placed inside the `pdf_resources/` directory, or an absolute path must be provided.
- OCR functionality requires appropriate OCR support in the environment.
- When processing large files, adjust memory and timeout settings as needed.

---

## 📜 License

This project is licensed under the MIT License.  
For commercial use, please credit the original source.

---