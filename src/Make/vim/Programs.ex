:set viminfo=
:set ul=0
:set ttyfast
:%s/\([A-Za-z0-9_\-\.]*\)\.\<c\>/$(LD_OBJ_DIR)\1/e
:%s/\([A-Za-z0-9_\-\.]*\)\.\<C\>/$(LD_OBJ_DIR)\1/e
:%s/\([A-Za-z0-9_\-\.]*\)\.\<cxx\>/$(LD_OBJ_DIR)\1/e
:%s/\([A-Za-z0-9_\-\.]*\)\.\<cpp\>/$(LD_OBJ_DIR)\1/e
:%s/\([A-Za-z0-9_\-\.]*\)\.\<cc\>/$(LD_OBJ_DIR)\1/e
:%s/^/        /e
:%s/\([^ ][^ ]*\) */\1 \\/e
:%s/ *$//ge
:%s/ *$//ge
:$s/ *\\ *//e
:1s/.*/PROGRAMS = \\&/e
:wq
