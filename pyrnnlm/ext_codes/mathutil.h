/* 
 * File:   mathutil.h
 * Author: Administrator
 *
 * Created on 2014年8月1日, 下午3:07
 */

#ifndef MATHUTIL_H
#define	MATHUTIL_H
#include <math.h>

static inline float 
exp10(float x){
    return pow(10,x);
}

static inline double 
exp10(double x){
    return pow(10,x);
}



#endif	/* MATHUTIL_H */

