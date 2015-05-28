'''tokenizor utils'''
#-*- coding: utf-8 -*-

def words2index(source_file, index_file, index_dic_file):
    source_f = open(source_file, 'r')
    index_f = open(index_file, 'w')
    index_dic_f = open(index_dic_file, 'w')

    index_dic = {}
    index = 1
    for line in source_f:
        tptup = line.strip().split()
        if len(tptup) == 0:
            continue
        index_list = []
        for word in tptup:
            if not index_dic.has_key(word):
                index_dic[word] = index
                index += 1
            index_list.append(str(index_dic[word]))

        index_f.write(" ".join(index_list) + "\n")

    for word, index in index_dic.items():
        index_dic_f.write(word+"\t"+str(index) + "\n")

    source_f.close()
    index_f.close()
    index_dic_f.close()

if __name__ == '__main__':
    import sys
    if not len(sys.argv) == 4:
        print "usage: python", __file__, "source_file index_file index_dic_file"
    else:
        words2index(sys.argv[1], sys.argv[2], sys.argv[3])
        