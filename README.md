# Lab Day 17: LangGraph Memory Agent

Sinh viên: `Dong Manh Hung`  
Mã sinh viên: `2A202600465`

## Mục tiêu bài lab

Xây dựng một AI Agent bằng Python trên LangGraph có khả năng sử dụng 4 loại bộ nhớ:

- `Short-term memory`: `ConversationBufferMemory`
- `Long-term memory`: Redis
- `Episodic memory`: file JSON
- `Semantic memory`: ChromaDB

Ngoài ra agent cần có:

- `Memory Router` để quyết định nên truy xuất loại bộ nhớ nào
- `Context Window Management` để tự cắt tỉa context khi gần hết token
- `Priority-based eviction` theo 4 cấp độ ưu tiên
- `Benchmark` trên 10 hội thoại thực tế

## Cấu trúc project

- `src/memory_lab/backends.py`: cài đặt 4 memory backends
- `src/memory_lab/router.py`: bộ điều hướng memory
- `src/memory_lab/context.py`: quản lý token budget và eviction
- `src/memory_lab/agent.py`: workflow agent bằng LangGraph
- `src/memory_lab/benchmark.py`: benchmark, metrics và sinh báo cáo
- `data/benchmark_conversations.json`: 10 hội thoại benchmark
- `data/knowledge_base.json`: dữ liệu semantic memory
- `reports/benchmark_report.md`: báo cáo kết quả benchmark

## Nội dung đã hoàn thành

### Bước 1: Thiết lập 4 loại bộ nhớ

- Short-term memory dùng `ConversationBufferMemory`
- Long-term memory kết nối theo mô hình Redis backend
- Episodic memory lưu nhật ký trải nghiệm trong file JSON
- Semantic memory lưu tri thức vector trong ChromaDB

### Bước 2: Memory Router

- Agent phân loại câu hỏi theo nhóm sở thích, kiến thức và kinh nghiệm cũ
- Từ đó chọn truy xuất `long_term`, `semantic`, `episodic` hoặc kết hợp với `short_term`

### Bước 3: Context Window Management

- Có cơ chế `auto-trim` khi vượt ngân sách token
- Có `priority-based eviction` để giữ lại thông tin quan trọng trước
- Mức ưu tiên cao nhất dành cho `system prompt` và `user input`

### Bước 4: Benchmark

- Đã xây dựng bộ benchmark gồm `10` hội thoại
- Đo các chỉ số:
  - `Response relevance`
  - `Context utilization`
  - `Token efficiency`
  - `Memory hit rate`
  - `Token budget breakdown`

## Kết quả benchmark

Kết quả chi tiết nằm tại [reports/benchmark_report.md](/home/hung/code/AI_CODE_VIN/lab/day17/reports/benchmark_report.md:1).

Tóm tắt:

- `Memory agent`: relevance `0.9333`, context utilization `1.0000`, hit rate `1.0000`
- `Baseline agent`: relevance `0.0667`, context utilization `0.0000`, hit rate `0.0000`

## Kết luận

Agent có bộ nhớ cho khả năng truy xuất thông tin tốt hơn agent thường. Việc kết hợp `short-term`, `long-term`, `episodic` và `semantic memory` giúp câu trả lời liên quan hơn, tận dụng ngữ cảnh tốt hơn và quản lý token hiệu quả hơn.
# Day17-DongManhHung-2A202600465
