import os
from pathlib import Path
from typing import Optional

import ujson
from tqdm import tqdm


def extract_conclusions(data: dict) -> Optional[str]:
    full_text: str = data["Full Text"]

    for i in range(len(full_text) - 1, -1, -1):
        if "conclusion" in full_text[i].lower():
            extracted_conclusion: str = " ".join(full_text[i:])
            return extracted_conclusion
        elif "results" in full_text[i].lower():
            extracted_conclusion: str = " ".join(full_text[i:])
            return extracted_conclusion
        # elif "conclusions" in full_text[i].lower():
        #     extracted_conclusion: str = " ".join(full_text[i:])
        #     # TODO: check length
        #     return extracted_conclusion
    return ""


def main():
    prefix = "data/papers"
    out = Path("data/with_conclusion")
    out.mkdir(exist_ok=True)
    paper_paths: list[Path] = [Path(f"{prefix}/{x}/clean_content.json") for x in os.listdir(prefix)]
    skipped: int = 0
    counter = 0
    for content in tqdm(paper_paths, desc="Adding conclusions"):
        if not content.exists():
            skipped += 1
            continue
        with open(content, "r", encoding="UTF-8") as f:
            data = ujson.load(f)
        # This seemed like a good idea, but it's not :(
        #     path_to_pdf = [f for f in os.listdir(content.parent) if f.endswith('.pdf')][0]
        #     pdf_file: PdfDocument = PdfDocument(content.parent / path_to_pdf)
        #     toc = []
        #     # This finds TOC if embedded in pdf already.
        #     for item in pdf_file.get_toc():
        #         if item.n_kids == 0:
        #             state = "*"
        #         elif item.is_closed:
        #             state = "-"
        #         else:
        #             state = "+"
        #         toc.append((state, item.title))
        # data["toc"] = toc
        text = extract_conclusions(data)
        if not text:
            counter += 1
        data["conclusions"] = text
        with open(out / f"{content.parent.name}.json", "w", encoding="UTF-8") as f:
            ujson.dump(data, f, indent=2)
    print(f"Skipped {skipped} files (could not extract text).")
    print(f"Not found {counter}.")


if __name__ == "__main__":
    main()
