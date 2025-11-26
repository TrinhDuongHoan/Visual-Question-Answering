# Vietnamese Visual Question Answering (VQA) 🇻🇳

Dự án xây dựng mô hình Hỏi đáp trên hình ảnh (Visual Question Answering) cho ngôn ngữ tiếng Việt, sử dụng kiến trúc Encoder-Decoder hiện đại.

## 🧠 Architecture (Kiến trúc)

Mô hình được thiết kế theo cơ chế **Fusion Encoder-Decoder**:
* **Image Encoder:** `ViT (Vision Transformer)` - Trích xuất đặc trưng hình ảnh.
* **Question Encoder:** `PhoBERT` - Trích xuất đặc trưng ngữ nghĩa câu hỏi tiếng Việt.
* **Fusion Strategy:** Kết hợp (Concatenate/Element-wise product) đặc trưng ảnh và câu hỏi.
* **Decoder:** `GPT` - Sinh câu trả lời tự nhiên dựa trên đặc trưng tổng hợp.

## 📂 Project Structure

```bash
├── configs/             # Configuration files
├── notebooks/           # Jupyter Notebooks for experiments
├── src/                 # Source code modules
│   ├── dataset.py       # Data loading & preprocessing
│   ├── model.py         # ViT-PhoBERT-GPT architecture
│   └── utils.py         # Helper functions
└── ...