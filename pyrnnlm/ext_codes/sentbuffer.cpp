# include "sentbuffer.h"

sentbuffer::sentbuffer(char* sentence){
    this->sent = sentence;
    this->index = 0;
}

void sentbuffer::nextword(char* word){
    int a=0,ch;
    
    while(this->sent[this->index]!=0){
        ch = this->sent[this->index];
        this->index++;
        if (ch==13) continue;

	if ((ch==' ') || (ch=='\t') || (ch=='\n')) {
            if(a>0) break;
            else continue;
        }
        
        word[a]=ch;
        a++;
        
        if(a>=100){
            a--;
        }
    }
    
    word[a]=0;
}

bool sentbuffer::isend(){
    if(this->sent[this->index]==0)
        return true;
    else return false;
}