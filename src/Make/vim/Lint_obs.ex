:set viminfo=
:set ul=0
:set ttyfast
:%s/\(.*\)\.cp* *$/    \1.ln \\/ge
:$s/ *\\ *//ge
:%s/[A-Za-z0-9._-\.]*\.ln/$(OBJ_DIR)&/e
:1s/.*/LINT_OBS = \\&/
:wq
