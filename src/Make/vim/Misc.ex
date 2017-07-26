:set viminfo=
:set ul=0
:set ttyfast
:%s/\([A-Za-z0-9_\-\.]*\)\.\<c\>/$(MISC_BIN_DIR)\1/e
:%s/\([A-Za-z0-9_\-\.]*\)\.\<C\>/$(MISC_BIN_DIR)\1/e
:%s/\([A-Za-z0-9_\-\.]*\)\.\<cxx\>/$(MISC_BIN_DIR)\1/e
:%s/\([A-Za-z0-9_\-\.]*\)\.\<cpp\>/$(MISC_BIN_DIR)\1/e
:%s/\([A-Za-z0-9_\-\.]*\)\.\<cc\>/$(MISC_BIN_DIR)\1/e
:%s/^/        /e
:%s/\([^ ][^ ]*\) */\1 \\/e
:%s/ *$//ge
:%s/ *$//ge
:$s/ *\\ *//e
:1s/.*/MISC_PROGRAMS = \\&/e
:wq
