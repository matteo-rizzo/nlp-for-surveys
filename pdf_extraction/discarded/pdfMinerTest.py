# from pdfminer.converter import PDFPageAggregator
# from pdfminer.layout import LAParams, LTTextLineHorizontal
# from pdfminer.pdfdocument import PDFDocument
# from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
# from pdfminer.pdfpage import PDFPage
# from pdfminer.pdfpage import PDFTextExtractionNotAllowed
# from pdfminer.pdfparser import PDFParser
#
#
# def main():
#     path = "data/papers/8320/Mcgrath and Heiens - 2003 - Beware the Internet panacea How tried and true st.pdf"
#
#     # Open the PDF file in binary mode
#     with open(path, 'rb') as fp:
#         # Create a PDF parser object
#         parser = PDFParser(fp)
#
#         # Create a PDF document object
#         document = PDFDocument(parser)
#
#         # Check if the document allows text extraction, otherwise raise an exception
#         if not document.is_extractable:
#             raise PDFTextExtractionNotAllowed
#
#         # Create a PDF resource manager object
#         rsrcmgr = PDFResourceManager()
#
#         # Set the parameters for analysis
#         laparams = LAParams(line_overlap=0.5, char_margin=100, word_margin=0.1, boxes_flow=0.5)
#
#         # Create a PDF device object
#         device = PDFPageAggregator(rsrcmgr, laparams=laparams)
#
#         # Create a PDF interpreter object
#         interpreter = PDFPageInterpreter(rsrcmgr, device)
#
#         # Loop through each page and analyze its layout to extract the TOC
#         for page in PDFPage.create_pages(document):
#             interpreter.process_page(page)
#             layout = device.get_result()
#
#             # Loop through each element on the page and extract the TOC
#             for element in layout:
#                 if isinstance(element, LTTextLineHorizontal):
#                     # Extract text and position for the text line
#                     text = element.get_text().strip()
#                     x, y, w, h = element.bbox
#
#                     # Identify the TOC entries by their position and font size
#                     if 12 <= h <= 20 and 5 <= x <= 50:
#                         print(f"TOC entry: {text}")
#
#
# if __name__ == "__main__":
#     main()

from pdfminer.high_level import extract_text

def main():
    text = extract_text("data/papers/8244/Christodoulou and Langley - 2020 - A gaming simulation approach to understanding blue.pdf")
    print(text)


if __name__ == "__main__":
    main()
