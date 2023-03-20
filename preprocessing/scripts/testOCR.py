import cv2
import layoutparser as lp
import numpy as np
from pdf2image import pdf2image

def main():
    path = "data/papers/8320/Mcgrath and Heiens - 2003 - Beware the Internet panacea How tried and true st.pdf"
    pdf_layout = lp.load_pdf(path)
    # print(pdf_layout[0])  # the layout for page 0
    pdf_layout, pdf_images = lp.load_pdf(path, load_images=True)
    img = lp.draw_box(pdf_images[1], pdf_layout[1])
    img.show()
    # folder_path: Path = Path("data/processed/success")
    #
    # success_list: list[str] = os.listdir(folder_path)
    # for file_path in [folder_path / fp for fp in success_list]:
    #     with open(file_path, 'r', encoding='UTF-8') as buffer:
    #         data = json.load(buffer)
    #     print("")


if __name__ == "__main__":
    main()
