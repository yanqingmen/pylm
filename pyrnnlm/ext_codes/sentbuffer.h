/* 
 * File:   sentbuffer.h
 * Author: administrator
 *
 * Created on 2013年3月1日, 上午11:06
 */

#ifndef SENTBUFFER_H
#define	SENTBUFFER_H
# include "stdlib.h"

class sentbuffer{
public:
    sentbuffer(char* sentence);
    void nextword(char* word);
    bool isend();
private:
    char* sent;
    int index;
};



#endif	/* SENTBUFFER_H */

