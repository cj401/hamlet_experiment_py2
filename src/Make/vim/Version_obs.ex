:set viminfo=
:set ul=0
:set ttyfast
:%s/\([A-Za-z0-9_\-\.]*\)\.\<c\> *$/    $(OBJ_DIR)\1.o \\/ge
:%s/\([A-Za-z0-9_\-\.]*\)\.\<C\> *$/    $(CXX_OBJ_DIR)\1.o \\/ge
:%s/\([A-Za-z0-9_\-\.]*\)\.\<cpp\> *$/    $(CXX_OBJ_DIR)\1.o \\/ge
:%s/\([A-Za-z0-9_\-\.]*\)\.\<cxx\> *$/    $(CXX_OBJ_DIR)\1.o \\/ge
:%s/\([A-Za-z0-9_\-\.]*\)\.\<cc\> *$/    $(CXX_OBJ_DIR)\1.o \\/ge
:$s/ *\\ *//ge
:1s/.*/VERSION_OBS = \\&/
:wq
