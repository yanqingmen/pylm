#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <cfloat>
#include "rnnlmlib.h"


void fastmatrixXvector(struct neuron *dest, struct neuron *srcvec, struct synapse *srcmatrix, int matrix_width, int from, int to, int from2, int to2, int type){
	 int a, b;
	if (type==0) {		//ac mod
		for (b=from; b<to; b++) {
		    for (a=from2; a<to2; a++) {
	    		dest[b].ac += srcvec[a].ac * srcmatrix[a+b*matrix_width].weight;
	    	    }
		}
	    }
	    else 		//er mod
	    if (type==1) {
		for (a=from2; a<to2; a++) {
		    for (b=from; b<to; b++) {
	    		dest[a].er += srcvec[b].er * srcmatrix[a+b*matrix_width].weight;
	    	    }
	    	}
	    }
}

bool neuEqual(struct neuron *fvec, struct neuron *svec, int begin, int end){
	for(int i=begin; i<end;  i++){
		if(fvec[i].ac != svec[i].ac){
			return false;
		}
	}
	return true;
}

void testmatrixXvector(){
	int rows=5, width=10, from=2, to=8,  from2=1, to2=4;
	neuron *srcvec = (struct neuron *)calloc( width, sizeof(struct neuron));
	synapse *srcmatrix = (struct synapse *)calloc(rows*width, sizeof(struct synapse));
	neuron *fvec = (struct neuron *)calloc(rows, sizeof(struct neuron));
	neuron *svec = (struct neuron *)calloc(rows, sizeof(struct neuron));
	CRnnLM rnnlm;

	for(int i=0; i<width; i++){
		srcvec[i].ac = rnnlm.random(-0.1, 0.1);
	}

	for(int i=0; i<rows*width; i++){
		srcmatrix[i].weight = rnnlm.random(-0.1, 0.1);
	}

	rnnlm.matrixXvector(fvec, srcvec, srcmatrix, width, from, to, from2, to2, 0);
	fastmatrixXvector(svec, srcvec, srcmatrix, width, from, to, from2, to2, 0);
	printf("check matrix * vector result: %d \n", neuEqual(fvec, svec, from2, to2));
	 free(srcvec);
	 free(srcmatrix);
	 free(fvec);
	 free(svec);
}

int main(int argc, char **argv){
	 testmatrixXvector();
}
