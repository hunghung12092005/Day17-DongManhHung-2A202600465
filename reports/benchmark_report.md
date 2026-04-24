# Benchmark Report

Sinh viên: `Dong Manh Hung`  
Mã sinh viên: `2A202600465`

## Bảng so sánh metrics

| Agent | Response relevance | Context utilization | Token efficiency | Avg total tokens | Memory hit rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| Memory agent | 0.9333 | 1.0000 | 5.7250 | 265.50 | 1.0000 |
| Baseline agent | 0.0667 | 0.0000 | 1.0101 | 66.00 | 0.0000 |

## Phân tích hit rate

- Memory agent hit rate: 100.00%
- Baseline agent hit rate: 0.00%
- Memory agent có khả năng truy xuất đúng backend tốt hơn nhờ router + retrieval chuyên biệt.

## Token budget breakdown

| Bucket | Avg tokens |
| --- | ---: |
| system_tokens | 37.00 |
| history_tokens | 0.00 |
| memory_tokens | 136.70 |
| user_tokens | 13.00 |
| response_tokens | 78.80 |
| total_tokens | 265.50 |

## Kết luận

- Agent có memory tạo câu trả lời bám ngữ cảnh tốt hơn baseline.
- Semantic, episodic và long-term memory giúp tăng relevance mà không cần giữ toàn bộ lịch sử trong prompt.
- Priority-based eviction giúp kiểm soát token budget và tránh lãng phí context window.
