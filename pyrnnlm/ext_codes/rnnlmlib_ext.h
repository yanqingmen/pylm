/* 
 * File:   rnnlmlib_ext.h
 * Author: Administrator
 *
 * Created on 2014年8月1日, 下午3:18
 */

#ifndef RNNLMLIB_EXT_H
#define	RNNLMLIB_EXT_H
#include "rnnlmlib.h"

static vocab_word empty_word={-1,"",-1,-1};

enum layer{
    layer_input,
    layer_hiden,
    layer_hidenc,
    layer_output
};


class EasyCRnnLM:public CRnnLM{
    public:
        //get the neu value of given index in certain layer
        real getNeuValue(layer layerid, int index);
        //get the words number of given class id
        int getWordsNumInClass(int classid);
        //get the word of given index
        vocab_word getWord(int index);
        //return the vocab size
        int getVocabSize();
        //get the word index(in vocab) of given classid and wordid(within the class)
        int getWordInClass(int classid, int wordid);
        //get the word id after sampling
        int getNextWord(int last_word);
        //calculate score of given sentence
        double calSentScore(char* sent);
        //calculate score for newword, with current hiden statement
//        double calWordScore(char* word);
        //update hiden neu statement by given word
//        void updateNeu(char* word);
};


#endif	/* RNNLMLIB_EXT_H */

