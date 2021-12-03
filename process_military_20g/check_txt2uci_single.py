from tqdm import tqdm
uci_url = '/home/lcl/LightLDA/military_20g/20g/docword.military20_new.txt'

last_doc_id = 0
with open(uci_url,'r') as uci_file:
    pbar = tqdm(total=6086741)
    tot = 0
    last_doc_id = 0
    tmp = []
    for line in uci_file:
        doc_tokens = line.strip().split(" ")
        if len(doc_tokens) == 3:
            doc_index = int(doc_tokens[0])
            word_index = doc_tokens[1]
            word_freq = doc_tokens[2]
            if (doc_index - last_doc_id) > 1:
                print("Before is {}, this line is {}".format(tmp[-2:],doc_tokens))
            last_doc_id = doc_index
            tmp.append(doc_tokens)
        else:
            print("Can't split 3 data".format(line))
            pass
        tot += 1
        pbar.set_postfix({'rate':tot/6086741})
        pbar.update(1)
    pbar.close()
