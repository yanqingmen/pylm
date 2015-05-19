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

 int EasyCRnnLM::getDirectSize(){
    return this->direct_size;
 }

real EasyCRnnLM::getDirectSynValue(int index){
    if(index < this->direct_size && index >= 0){
        return this->syn_d[index];
    }
    else{
        return -1;
    }
}

real EasyCRnnLM::getNeuValue(layer layerid, int index){
    if(layerid == layer_input && index < this->layer0_size && index >= 0)   return this->neu0[index].ac;
    else if(layerid == layer_hiden && index < this->layer1_size && index >= 0)   return this->neu1[index].ac;
    else if(layerid == layer_hidenc && index < this->layerc_size && index >= 0)  return this->neuc[index].ac;
    else if(layerid == layer_output && index < this->layer2_size && index >= 0)  return this->neu2[index].ac;
    else return -1;
}

real EasyCRnnLM::getSynValue(weight synid, int f_index, int s_index){
    if(synid == input2hiden && f_index < this->layer0_size && f_index >=0 && s_index < this->layer1_size && s_index >= 0){
        return syn0[f_index+s_index*this->layer0_size].weight;
    }
    else if(synid == hiden2comp && f_index < this->layer1_size && f_index >= 0 && s_index < this->layerc_size && s_index >= 0){
        return syn1[f_index+s_index*this->layer1_size].weight;
    }
    else if(synid == hiden2out && f_index < this->layer1_size && f_index >= 0 && s_index < this->layer2_size && s_index >= 0){
        return syn1[f_index+s_index*this->layer1_size].weight;
    }
    else if(synid == comp2out && f_index < this->layerc_size && f_index >= 0 && s_index < this->layer2_size && s_index >= 0){
        return sync[f_index+s_index*this->layerc_size].weight;
    }
    else return -1;
}

int EasyCRnnLM::getLayerSize(layer layerid){
    if(layerid == layer_input) return this->layer0_size;
    else if(layerid == layer_hiden) return this->layer1_size;
    else if(layerid == layer_hidenc) return this->layerc_size;
    else if(layerid == layer_output) return this->layer2_size;
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

void EasyCRnnLM::computeNetInfo(int last_word, int word){
    int a, b, c;
    real val;
    double sum;   //sum is used for normalization: it's better to have larger precision as many numbers are summed together here
    
    if (last_word!=-1) neu0[last_word].ac=1;

    //propagate 0->1
    for (a=0; a<layer1_size; a++) neu1[a].ac=0;
    for (a=0; a<layerc_size; a++) neuc[a].ac=0;
    
    matrixXvector(neu1, neu0, syn0, layer0_size, 0, layer1_size, layer0_size-layer1_size, layer0_size, 0);

    for (b=0; b<layer1_size; b++) {
        a=last_word;
        if (a!=-1) neu1[b].ac += neu0[a].ac * syn0[a+b*layer0_size].weight;
    }

    //activate 1      --sigmoid
    for (a=0; a<layer1_size; a++) {
    if (neu1[a].ac>50) neu1[a].ac=50;  //for numerical stability
        if (neu1[a].ac<-50) neu1[a].ac=-50;  //for numerical stability
        val=-neu1[a].ac;
        neu1[a].ac=1/(1+fasterexp(val));
    }

    //  neu1 info
    printf("neu1 info:\n");
    for (a=0; a<layer1_size; a++) printf("%.4f ", neu1[a].ac);
    printf("\n");
    
    if (layerc_size>0) {
    matrixXvector(neuc, neu1, syn1, layer1_size, 0, layerc_size, 0, layer1_size, 0);
    //activate compression      --sigmoid
    for (a=0; a<layerc_size; a++) {
        if (neuc[a].ac>50) neuc[a].ac=50;  //for numerical stability
            if (neuc[a].ac<-50) neuc[a].ac=-50;  //for numerical stability
            val=-neuc[a].ac;
            neuc[a].ac=1/(1+fasterexp(val));
    }

    // neuc info
    printf("neuc info:\n");
    for (a=0; a<layerc_size; a++) printf("%.4f ", neuc[a].ac);
    printf("\n");
    }
        
    //1->2 class
    for (b=vocab_size; b<layer2_size; b++) neu2[b].ac=0;
    
    if (layerc_size>0) {
    matrixXvector(neu2, neuc, sync, layerc_size, vocab_size, layer2_size, 0, layerc_size, 0);
    }
    else
    {
    matrixXvector(neu2, neu1, syn1, layer1_size, vocab_size, layer2_size, 0, layer1_size, 0);
    }

    // class info
    printf("classes info after matrixXvector");
    for (a=vocab_size; a<layer2_size; a++) printf("%.4f ", neu2[a].ac);
    printf("\n");

    //apply direct connections to classes
    if (direct_size>0) {
    unsigned long long hash[MAX_NGRAM_ORDER];   //this will hold pointers to syn_d that contains hash parameters
    
    for (a=0; a<direct_order; a++) hash[a]=0;
    
    for (a=0; a<direct_order; a++) {
        b=0;
        if (a>0) if (history[a-1]==-1) break;   //if OOV was in history, do not use this N-gram feature and higher orders
        hash[a]=PRIMES[0]*PRIMES[1];
                
        for (b=1; b<=a; b++) hash[a]+=PRIMES[(a*PRIMES[b]+b)%PRIMES_SIZE]*(unsigned long long)(history[b-1]+1); //update hash value based on words from the history
        hash[a]=hash[a]%(direct_size/2);        //make sure that starting hash index is in the first half of syn_d (second part is reserved for history->words features)
    }
    
    for (a=vocab_size; a<layer2_size; a++) {
        for (b=0; b<direct_order; b++) if (hash[b]) {
        neu2[a].ac+=syn_d[hash[b]];     //apply current parameter and move to the next one
        hash[b]++;
        } else break;
    }

    // hash info for classes apply direct
    printf("hash info for classes after apply direct\n");
    for (int i = 0; i < MAX_NGRAM_ORDER; ++i) printf("%d ", hash[i]);
    printf("\n");

    // classes info after apply direct
    printf("classes info after apply direct\n");
    for (a=vocab_size; a<layer2_size; a++) printf("%.4f ", neu2[a].ac);
    printf("\n");
    }

    //activation 2   --softmax on classes
    // 20130425 - this is now a 'safe' softmax

    sum=0;
    real maxAc=-FLT_MAX;
    for (a=vocab_size; a<layer2_size; a++)
        if (neu2[a].ac>maxAc) maxAc=neu2[a].ac; //this prevents the need to check for overflow
    for (a=vocab_size; a<layer2_size; a++)
        sum+=fasterexp(neu2[a].ac-maxAc);
    for (a=vocab_size; a<layer2_size; a++)
        neu2[a].ac=fasterexp(neu2[a].ac-maxAc)/sum;

    // classes info after softmax
    printf("classes info after softmax\n");
    for (a=vocab_size; a<layer2_size; a++) printf("%.4f ", neu2[a].ac);
    printf("\n");
 
    if (gen>0) return;  //if we generate words, we don't know what current word is -> only classes are estimated and word is selected in testGen()

    
    //1->2 word
    
    if (word!=-1) {
        for (c=0; c<class_cn[vocab[word].class_index]; c++) neu2[class_words[vocab[word].class_index][c]].ac=0;
        if (layerc_size>0) {
        matrixXvector(neu2, neuc, sync, layerc_size, class_words[vocab[word].class_index][0], class_words[vocab[word].class_index][0]+class_cn[vocab[word].class_index], 0, layerc_size, 0);
    }
    else
    {
        matrixXvector(neu2, neu1, syn1, layer1_size, class_words[vocab[word].class_index][0], class_words[vocab[word].class_index][0]+class_cn[vocab[word].class_index], 0, layer1_size, 0);
    }
    }

    printf("words info after matrixXvector\n");
    printf("begin: %d, end: %d\n", class_words[vocab[word].class_index][0], class_words[vocab[word].class_index][0]+class_cn[vocab[word].class_index]);
    for (a=class_words[vocab[word].class_index][0]; a<class_words[vocab[word].class_index][0]+class_cn[vocab[word].class_index]; a++) printf("%.4f ", neu2[a].ac);
    printf("\n");
    
    //apply direct connections to words
    if (word!=-1) if (direct_size>0) {
    unsigned long long hash[MAX_NGRAM_ORDER];
        
    for (a=0; a<direct_order; a++) hash[a]=0;
    
    for (a=0; a<direct_order; a++) {
        b=0;
        if (a>0) if (history[a-1]==-1) break;
        hash[a]=PRIMES[0]*PRIMES[1]*(unsigned long long)(vocab[word].class_index+1);
                
        for (b=1; b<=a; b++) hash[a]+=PRIMES[(a*PRIMES[b]+b)%PRIMES_SIZE]*(unsigned long long)(history[b-1]+1);
        hash[a]=(hash[a]%(direct_size/2))+(direct_size)/2;
    }
    
    for (c=0; c<class_cn[vocab[word].class_index]; c++) {
        a=class_words[vocab[word].class_index][c];
        
        for (b=0; b<direct_order; b++) if (hash[b]) {
        neu2[a].ac+=syn_d[hash[b]];
        hash[b]++;
        hash[b]=hash[b]%direct_size;
        } else break;
    }
    // hash info for classes apply direct
    printf("hash info for words after apply direct\n");
    for (int i = 0; i < MAX_NGRAM_ORDER; ++i) printf("%d ", hash[i]);
    printf("\n");


    printf("words info after apply direct connections\n");
    for (a=class_words[vocab[word].class_index][0]; a<class_words[vocab[word].class_index][0]+class_cn[vocab[word].class_index]; a++) printf("%.4f ", neu2[a].ac);
    printf("\n");
    }

    //activation 2   --softmax on words
    // 130425 - this is now a 'safe' softmax
    sum=0;
    if (word!=-1) { 
        maxAc=-FLT_MAX;
        for (c=0; c<class_cn[vocab[word].class_index]; c++) {
            a=class_words[vocab[word].class_index][c];
            if (neu2[a].ac>maxAc) maxAc=neu2[a].ac;
        }
        for (c=0; c<class_cn[vocab[word].class_index]; c++) {
            a=class_words[vocab[word].class_index][c];
            sum+=fasterexp(neu2[a].ac-maxAc);
        }
        for (c=0; c<class_cn[vocab[word].class_index]; c++) {
            a=class_words[vocab[word].class_index][c];
            neu2[a].ac=fasterexp(neu2[a].ac-maxAc)/sum; //this prevents the need to check for overflow
        }
    }

    printf("words info after softmax\n");
    for (a=class_words[vocab[word].class_index][0]; a<class_words[vocab[word].class_index][0]+class_cn[vocab[word].class_index]; a++) printf("%.4f ", neu2[a].ac);
    printf("\n");
}
