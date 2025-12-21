import sys
import os

path = r'd:\Kỳ 1 năm 4\On thi\BIGDATA\hasaki-ml-streamlit\app\pages\03_Segmentation.py'
if not os.path.exists(path):
    print(f"File not found: {path}")
    sys.exit(1)

with open(path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

found = False
with open(path, 'w', encoding='utf-8') as f:
    for line in lines:
        if 'Chiến Lược Giá (Ưu Tiên Ràng Buộc Tồn Kho)' in line and 'st.subheader' in line:
            # Preserve indentation
            indent_len = len(line) - len(line.lstrip())
            indent = line[:indent_len]
            new_line = indent + "st.markdown('### <i class=\"fa-solid fa-coins\" style=\"color: #28a745;\"></i> Chiến Lược Giá (Ưu Tiên Ràng Buộc Tồn Kho)', unsafe_allow_html=True)\n"
            f.write(new_line)
            found = True
        else:
            f.write(line)

if found:
    print("Successfully replaced line.")
else:
    print("Target line not found.")
