def Convert():
    with open("./LLM_dataGenerate.txt", "r", encoding='utf-8') as f:
        cont = f.read()
    cont = cont.replace("\n", '')
    for i in range(1, 30, 1):
        cont = cont.replace(f"{i}. ", "")
    with open("./LLM_dataGenerate.bio", 'w', encoding='utf-8') as f:
        for i in cont:
            if len(i) == 0:
                continue
            f.write(f"{i} O\n")
            if i in '.。':
                f.write("\n")


if __name__ == '__main__':
    Convert()