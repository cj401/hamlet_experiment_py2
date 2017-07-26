:set viminfo=
:set ul=0
:set ttyfast
:%s/\.\<cpp\>/.ln/ge
:%s/\.\<cc\>/.ln/ge
:%s/\.\<cxx\>/.ln/ge
:%s/\.\<c\>/.ln/ge
:%s/^/        /
:%s/\([^ ][^ ]*\) */\1 \\/
:%s/ *$//ge
:%s/[A-Za-z0-9_\-\.]*\.ln/$(OBJ_DIR)&/e
:%s/ *$//ge
:1s/.*/LINT_OBS = \\&/e
:wq
