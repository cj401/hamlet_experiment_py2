:set viminfo=
:set ul=0
:set ttyfast
:%s/\([A-Za-z0-9_\-\.]*\)\.\<c\>/$(WORK_BIN_DIR)\1/e
:%s/\([A-Za-z0-9_\-\.]*\)\.\<C\>/$(WORK_BIN_DIR)\1/e
:%s/\([A-Za-z0-9_\-\.]*\)\.\<cxx\>/$(WORK_BIN_DIR)\1/e
:%s/\([A-Za-z0-9_\-\.]*\)\.\<cpp\>/$(WORK_BIN_DIR)\1/e
:%s/\([A-Za-z0-9_\-\.]*\)\.\<cc\>/$(WORK_BIN_DIR)\1/e
:%s/^/        /e
:%s/\([^ ][^ ]*\) */\1 \\/e
:%s/ *$//ge
:%s/ *$//ge
:$s/ *\\ *//e
:1s/.*/WORK_PROGRAMS = \\&/e
:wq
