import json
import os
from pathlib import Path
import ocrmypdf

def main():
    path = "data/papers/8320/Mcgrath and Heiens - 2003 - Beware the Internet panacea How tried and true st.pdf"
    ocrmypdf.ocr(path, 'output.pdf', deskew=True)
    # folder_path: Path = Path("data/processed/success")
    #
    # success_list: list[str] = os.listdir(folder_path)
    # for file_path in [folder_path / fp for fp in success_list]:
    #     with open(file_path, 'r', encoding='UTF-8') as buffer:
    #         data = json.load(buffer)
    #     print("")


if __name__ == "__main__":
    main()
