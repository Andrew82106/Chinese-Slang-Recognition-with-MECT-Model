from Utils.paths import *


def merge_LLM_data():
    """
    合并LLM增强data和wiki data
    """
    with open(LLM_data_expand_bio_file_path, "r", encoding='utf-8') as llm:
        with open(wiki_file_old_path, "r", encoding='utf-8') as origin_wiki:
            with open(wiki_file_path, "w", encoding='utf-8') as new_wiki:
                llm_txt = llm.read()
                origin_wiki_txt = origin_wiki.read()
                Summary = origin_wiki_txt + llm_txt
                new_wiki.write(Summary)

    with open(LLM_data_expand_bio_file_path, "r", encoding='utf-8') as llm:
        with open(wiki_file_old_path, "r", encoding='utf-8') as origin_wiki:
            with open(wiki_file_path, "r", encoding='utf-8') as new_wiki:
                llm_txt = llm.read()
                origin_wiki_txt = origin_wiki.read()
                new_wiki_txt = new_wiki.read()
                if len(llm_txt) + len(origin_wiki_txt) == len(new_wiki_txt):
                    print(f"successfully merge leveled data and original wiki data (length message:{len(new_wiki_txt)})")
                else:
                    raise Exception(f"合并失败，长度不对应:{len(llm_txt)}, {len(origin_wiki_txt)}, {len(new_wiki_txt)}:")
