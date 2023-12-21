from Utils.paths import *
import os


def Convert(
        r_LLM_dataGenerate_txt=os.path.join(LLM_data_expand_path, "LLM_dataGenerate.txt"),
        r_LLM_dataGenerate_bio=os.path.join(LLM_data_expand_path, "LLM_dataGenerate.bio")
):
    with open(r_LLM_dataGenerate_txt, "r", encoding='utf-8') as f:
        cont = f.read()
    cont = cont.replace("\n", '')
    for i in range(100, 1, -1):
        cont = cont.replace(f"{i}. ", "")
    with open(r_LLM_dataGenerate_bio, 'w', encoding='utf-8') as f:
        for i in cont:
            if len(i.replace(" ", "")) == 0:
                continue
            f.write(f"{i} O\n")
            if i in '.。':
                f.write("\n")
    print(f"successfully write {r_LLM_dataGenerate_txt} to {r_LLM_dataGenerate_bio}")


if __name__ == '__main__':
    Convert()
