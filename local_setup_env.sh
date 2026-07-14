# 1. Cài đặt uv bằng Homebrew
brew install uv

# 2. Tạo môi trường ảo (uv sẽ tự xử lý pip và ensurepip hoàn hảo)
uv venv .venv --python 3.11
# 3. kiểm tra và xóa các package tensorflow đã cài đặt
uv pip uninstall tensorflow tensorflow-macos tensorflow-metal
# Cập nhật công cụ cài đặt
source .venv/bin/activate
# Cài đặt file requirements của bạn
uv pip install -r requirements.txt

# Cài đặt lại gói tensorflow chuẩn (uv sẽ tự chọn bản phù hợp nhất cho Apple Silicon)
uv pip install tensorflow
