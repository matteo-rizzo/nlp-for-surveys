import json
import os
from pathlib import Path
import layoutparser as lp
import cv2
import numpy as np
from pdf2image import pdf2image


def main():
    path = "data/papers/8320/Mcgrath and Heiens - 2003 - Beware the Internet panacea How tried and true st.pdf"
    img = np.asarray(pdf2image.convert_from_path(path)[0])

    model = lp.Detectron2LayoutModel('lp://PubLayNet/mask_rcnn_X_101_32x8d_FPN_3x/config',
                                     extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.5],
                                     label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"})
    layout_result = model.detect(img)
    lp.draw_box(img, layout_result, box_width=5, box_alpha=0.2, show_element_type=True)
    # folder_path: Path = Path("data/processed/success")
    #
    # success_list: list[str] = os.listdir(folder_path)
    # for file_path in [folder_path / fp for fp in success_list]:
    #     with open(file_path, 'r', encoding='UTF-8') as buffer:
    #         data = json.load(buffer)
    #     print("")


if __name__ == "__main__":
    main()
