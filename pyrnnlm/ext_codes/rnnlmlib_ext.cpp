/* 
 * File:   rnnlmlib_ext.cpp
 * Author: Administrator
 *
 * Created on 2014年8月1日, 下午3:18
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <cfloat>
#include "fastexp.h"
#include "rnnlmlib_ext.h"
#include "sentbuffer.h"

real EasyCRnnLM::getNeuValue(layer layerid, int index){
    if(layerid == layer_input && index < this->layer0_size && index >= 0)   return this->neu0[index].ac;
    else if(layerid == layer_hiden && index < this->layer1_size && index >= 0)   return this->neu1[index].ac;
    else if(layerid == layer_hidenc && index < this->layerc_size && index >= 0)  return this->neuc[index].ac;
    else if(layerid == layer_output && index < this->layer2_size && index >= 0)  return this->neu2[index].ac;
    else return -1;
}

int EasyCRnnLM::getWordsNumInClass(int classid){
    if(classid < this->class_size  && classid >= 0) return this->class_cn[classid];
    else return -1;
}

vocab_word EasyCRnnLM::getWord(int index){
    if(index < this->vocab_size  && index >= 0) return this->vocab[index];
    else return empty_word;
}

int EasyCRnnLM::getVocabSize(){
    return this->vocab_size;
}

int EasyCRnnLM::getWordInClass(int classid, int wordid){
    if(classid < 0 || classid >= this->class_size) return -1;
    else{
        if(wordid <0 || wordid >= this->class_cn[classid]) return -1;
        else return this->class_words[classid][wordid];
    }
}

int EasyCRnnLM::getNextWord(int last_word){
    int i,word,cla,c,b,a=0;
    real f,g,sum,val;

    if(neu0 == NULL) restoreNet();

    copyHiddenLayerToInput();

    if(independent && (last_word==0)) netReset();

    if (last_word!=-1) neu0[last_word].ac=0;
    for (a=MAX_NGRAM_ORDER-1; a>0; a--) history[a]=history[a-1];
    history[0] = last_word;

    computeNet(last_word,0);

    f = random(0,1);
    g = 0;
    i = vocab_size;

    while ((g<f) && (i<layer2_size)) {
        g+=neu2[i].ac;
        i++;
    }
    cla=i-1-vocab_size;
        
    if (cla>class_size-1) cla=class_size-1;
    if (cla<0) cla=0;

    for (c=0; c<class_cn[cla]; c++) neu2[class_words[cla][c]].ac=0;
    matrixXvector(neu2, neu1, syn1, layer1_size, class_words[cla][0], class_words[cla][0]+class_cn[cla], 0, layer1_size, 0);

    //apply direct connections to words
    if (word!=-1) if (direct_size>0) {
        unsigned long long hash[MAX_NGRAM_ORDER];

        for (a=0; a<direct_order; a++) hash[a]=0;

        for (a=0; a<direct_order; a++) {
            b=0;
            if (a>0) if (history[a-1]==-1) break;
            hash[a]=PRIMES[0]*PRIMES[1]*(unsigned long long)(cla+1);

            for (b=1; b<=a; b++) hash[a]+=PRIMES[(a*PRIMES[b]+b)%PRIMES_SIZE]*(unsigned long long)(history[b-1]+1);
            hash[a]=(hash[a]%(direct_size/2))+(direct_size)/2;
        }

        for (c=0; c<class_cn[cla]; c++) {
            a=class_words[cla][c];

            for (b=0; b<direct_order; b++) if (hash[b]) {
                neu2[a].ac+=syn_d[hash[b]];
                    hash[b]++;
                hash[b]=hash[b]%direct_size;
            } else break;
        }
    }

    //activation 2   --softmax on words
     sum=0;
     real maxAc=-FLT_MAX;
     for (c=0; c<class_cn[cla]; c++) {
      	a=class_words[cla][c];
        if (neu2[a].ac>maxAc) maxAc=neu2[a].ac;
     }
     for (c=0; c<class_cn[cla]; c++) {
        a=class_words[cla][c];
        sum+=fasterexp(neu2[a].ac-maxAc);
     }
     for (c=0; c<class_cn[cla]; c++) {
        a=class_words[cla][c];
        neu2[a].ac=fasterexp(neu2[a].ac-maxAc)/sum; //this prevents the need to check for overflow
     }

    f=random(0, 1);
    g=0;

    for (c=0; c<class_cn[cla]; c++) {
        a=class_words[cla][c];
        g+=neu2[a].ac;
        if (g>f) break;
    }
    word=a;
        
    if (word>vocab_size-1) word=vocab_size-1;
    if (word<0) word=0;

    return word; 
}

double EasyCRnnLM::calSentScore(char* sent){
    sentbuffer newbuf(sent);
    int a, b, word, last_word, wordcn;
    double d, score=0.0;
    
    if(neu0 == NULL) restoreNet();
    
    last_word=0;
    wordcn=0;
    copyHiddenLayerToInput();
    
    if (bptt>0) for (a=0; a<bptt+bptt_block; a++) bptt_history[a]=0;
    for (a=0; a<MAX_NGRAM_ORDER; a++) history[a]=0;
    if (independent) netReset();
    
    char crrtword[MAX_STRING];
    while (1) {
        if(newbuf.isend()) break;
        newbuf.nextword(crrtword);
        if(crrtword[0]==0) break;
        word = searchVocab(crrtword);
        computeNet(last_word,word);
        wordcn++;
        if(word!=-1){
            score += log10(neu2[vocab[word].class_index+vocab_size].ac * neu2[word].ac);
        }
        else{
            score +=-8.0; //penalty for missing words
        }
        
        if (last_word!=-1) neu0[last_word].ac=0;
        last_word=word;
        for (a=MAX_NGRAM_ORDER-1; a>0; a--) history[a]=history[a-1];
        history[0]=last_word;
        
        if (independent && (word==0)) netReset();

        copyHiddenLayerToInput();
    }
    
    return score/wordcn;
}
