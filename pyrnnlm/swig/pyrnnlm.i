%module pyrnnlm

%{
#define SWIG_FILE_WITH_INIT
#include "../rnnlm/rnnlmlib.h"
#include "../ext_codes/rnnlmlib_ext.h"
%}

%include "../rnnlm/rnnlmlib.h"
%include "../ext_codes/rnnlmlib_ext.h"