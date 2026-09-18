# Report Agent

Report Agent là ứng dụng Streamlit sử dụng LangGraph để tạo báo cáo phân tích từ tệp CSV hoặc Excel. Hệ thống kết hợp LLM với các phép tính Pandas đã kiểm chứng, tạo biểu đồ, kiểm tra bằng chứng và xuất báo cáo PDF.

## Chức năng chính

- Tải và xem trước dữ liệu CSV, XLSX hoặc XLS.
- Phát hiện header, phân loại sheet và chuẩn hóa tên cột chưa rõ nghĩa.
- Sinh câu hỏi phân tích phù hợp với cấu trúc dữ liệu.
- Thực hiện phép tính bằng Pandas thay vì để LLM tự tính số liệu.
- Sinh insight, kiểm chứng bằng chứng và loại nội dung trùng lặp.
- Tạo biểu đồ và báo cáo bằng tiếng Việt hoặc tiếng Anh.
- Hỗ trợ Google Gemini và API tương thích OpenAI.
- Xuất, xem trước và tải báo cáo PDF từ giao diện Streamlit.

## Luồng xử lý

```text
data_analysis
    ↓
question_framing
    ↓
insight_generation
    ↓
evidence_validation
    ↓
report_planning
    ↓
visualization
    ↓
report_drafting
    ↓
safety_check
    ↓
report_finalization
```

Mỗi node nhận và cập nhật `GraphState`. Các conditional edge trong LangGraph chỉ cho workflow đi tiếp khi node trước hoàn thành đúng trạng thái.

## Cấu trúc thư mục

```text
gen_report_agent/
├── app.py                         # Entry point để chạy Streamlit
├── requirements.txt              # Thư viện Python
├── packages.txt                  # Gói hệ thống cho môi trường Linux
├── .env                           # API key và cấu hình model, không commit
├── .gitignore
├── README.md
└── src/
    ├── streamlit_app.py           # Giao diện tải dữ liệu và hiển thị báo cáo
    ├── config.py                  # Cấu hình thư mục dữ liệu cục bộ
    ├── data_io.py                 # Đọc CSV, Excel, header và phân vùng dữ liệu
    ├── llm.py                     # Khởi tạo Gemini hoặc OpenAI-compatible LLM
    ├── privacy.py                 # Nhận diện và hạn chế cột dữ liệu cá nhân
    ├── graph/
    │   ├── builder.py             # Khai báo node, edge và biên dịch StateGraph
    │   └── state.py               # Hợp đồng GraphState dùng chung
    ├── schemas/
    │   └── messages.py            # Các Pydantic model của pipeline
    └── agents/
        ├── data_profiler.py       # Lập hồ sơ dữ liệu và phân loại sheet
        ├── question_framer.py     # Sinh câu hỏi và tính kết quả bằng Pandas
        ├── insight_engine.py      # Sinh insight từ kết quả đã kiểm chứng
        ├── evidence_validator.py  # Kiểm tra provenance và gộp insight trùng
        ├── report_planner.py      # Lập cấu trúc chủ đề và bảng bằng chứng
        ├── chart_generator.py     # Chọn và tạo biểu đồ
        ├── report_writer.py       # Soạn nội dung báo cáo
        ├── quality_gate.py        # Kiểm tra tính nhất quán và an toàn
        ├── report_formatting.py   # Nhãn, màu sắc và định dạng dùng chung
        └── report_exporter.py     # Xuất PDF bằng WeasyPrint hoặc ReportLab
```

Khi ứng dụng chạy, thư mục sau được tạo tự động:

```text
local_app_data/
├── uploads/    # Tệp người dùng tải lên
├── charts/     # Hình biểu đồ được sinh
└── reports/    # Báo cáo PDF đầu ra
```

## Yêu cầu hệ thống

- Python 3.11 trở lên.
- `pip` và môi trường ảo Python.
- Kết nối Internet để gọi LLM.
- API key của Google Gemini hoặc một API tương thích OpenAI.

Ứng dụng dùng `streamlit[pdf]>=1.57`, vì vậy nên cài đúng phiên bản trong `requirements.txt` để sử dụng trình xem PDF và các thành phần giao diện hiện tại.

## Cài đặt trên Windows

Mở PowerShell tại thư mục project:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Nếu PowerShell chặn việc kích hoạt môi trường ảo, chạy lệnh sau trong phiên hiện tại rồi kích hoạt lại:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

## Cài đặt trên Linux hoặc macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Trên Ubuntu hoặc Debian, WeasyPrint có thể cần các thư viện hệ thống trong `packages.txt`:

```bash
sudo apt update
sudo apt install -y libcairo2-dev libpango1.0-dev libgdk-pixbuf-2.0-dev libffi-dev shared-mime-info
```

Nếu WeasyPrint không khả dụng, project vẫn có đường xuất PDF dự phòng bằng ReportLab.

## Cấu hình biến môi trường

Tạo tệp `.env` tại thư mục gốc. Chỉ cần cấu hình nhà cung cấp mà bạn sử dụng.

### Google Gemini

```env
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=gemini-2.5-flash
```

### API tương thích OpenAI

```env
LLM_PROVIDER=proxyllm
PROXYLLM_API_KEY=your_api_key
PROXYLLM_BASE_URL=https://your-api-host.example/v1
PROXYLLM_MODEL=gpt-4o-mini
```

Không đưa `.env` hoặc API key lên Git. Tệp `.env` đã được khai báo trong `.gitignore`.

## Chạy ứng dụng

Sau khi kích hoạt môi trường ảo, chạy từ thư mục gốc:

```bash
streamlit run app.py
```

Hoặc:

```bash
python -m streamlit run app.py
```

Streamlit thường mở ứng dụng tại:

```text
http://localhost:8501
```

## Cách sử dụng

1. Chọn nhà cung cấp LLM ở thanh bên.
2. Kiểm tra hoặc thay đổi tên model.
3. Chọn ngôn ngữ báo cáo.
4. Tải lên tệp CSV hoặc Excel.
5. Nhập mục tiêu phân tích.
6. Nhấn **Tạo báo cáo**.
7. Theo dõi tiến trình các node.
8. Xem trước hoặc tải báo cáo PDF.

Dữ liệu nên có ít nhất 5 dòng và 2 cột. Với workbook Excel, hệ thống sẽ phân loại từng sheet thành dữ liệu, hướng dẫn, metadata hoặc sheet không hợp lệ trước khi phân tích.

## Định dạng dữ liệu

- CSV: được hỗ trợ trực tiếp bằng Pandas.
- XLSX: được đọc bằng `openpyxl`.
- XLS: giao diện cho phép tải lên nhưng môi trường có thể cần cài thêm `xlrd`:

```bash
pip install xlrd
```

Các cột có header rõ ràng được giữ nguyên. Cột rỗng hoặc có tên generic như `Unnamed`, `Column 1` hoặc `Cột 2` có thể được đặt lại tên khi có đủ bằng chứng từ kiểu dữ liệu, mẫu giá trị hoặc các sheet tương ứng.

## Cấu hình model

Cấu hình tập trung nằm trong `src/llm.py`:

- Gemini dùng `ChatGoogleGenerativeAI`.
- ProxyLLM hoặc OpenAI-compatible API dùng `ChatOpenAI`.
- Timeout mặc định là 120 giây.
- Số lần retry của client là 2.
- Temperature mặc định là `0.2`.

Nhà cung cấp và model được chọn trên giao diện sẽ được ghi vào `GraphState` và truyền tới các node cần gọi LLM.

## Xem sơ đồ LangGraph

Có thể lấy graph trực tiếp từ workflow đã biên dịch:

```python
from graph.builder import create_graph_workflow

workflow = create_graph_workflow()
graph = workflow.get_graph()

print(graph.draw_mermaid())
```

Khi chạy đoạn mã độc lập từ thư mục gốc, cần thêm `src` vào `PYTHONPATH` hoặc chạy trong môi trường đã cấu hình source root.

PowerShell:

```powershell
$env:PYTHONPATH = "src"
python your_script.py
```

Linux hoặc macOS:

```bash
PYTHONPATH=src python your_script.py
```

## Xử lý lỗi thường gặp

### Thiếu API key

```text
Thiếu GEMINI_API_KEY trong .env
```

hoặc:

```text
Thiếu PROXYLLM_API_KEY hoặc PROXYLLM_BASE_URL trong .env
```

Kiểm tra tên biến trong `.env`, sau đó khởi động lại Streamlit.

### Không đọc được Excel

Đảm bảo đã cài `openpyxl`. Với định dạng `.xls` cũ, cài thêm `xlrd`.

### Không hiển thị PDF

Đảm bảo phiên bản Streamlit đáp ứng `streamlit[pdf]>=1.57` và cài lại dependency:

```bash
pip install -r requirements.txt
```

### Xóa cache Streamlit

```bash
streamlit cache clear
```

### Đổi cổng chạy

```bash
streamlit run app.py --server.port=8502
```

## Lưu ý bảo mật

- Không commit `.env` và API key.
- Tệp tải lên và báo cáo được lưu trong `local_app_data/` trên máy chạy server.
- Không coi ràng buộc widget là lớp bảo mật; dữ liệu nhạy cảm vẫn cần được kiểm tra ở backend.
- Nên xóa định kỳ `local_app_data/uploads` và `local_app_data/reports` nếu dữ liệu có tính riêng tư.

#   R e p o r t _ A g e n t  
 