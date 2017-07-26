:set viminfo=
:set ul=0
:set ttyfast
:%s/\.o/.ln/ge
:%s/\<[A-Za-z0-9._-]*\.ln\>/$(OBJ_DIR)&/e
:%s/^#.*//ge
:%s/ \([^ :]\)/ \\            \1/ge
:%s/^\([^ ]\)/\1/ge
:s/^.*: *$//ge
:%s/^.*\/lib\/\([A-Za-z0-9_\-\.]*\)\/\(\$(OBJ_DIR)[A-Za-z0-9_\-]*.ln\)/$(\U\1\e_DIR)\2/e
:%s/^lib\/\([A-Za-z0-9_\-\.]*\)\/\(\$(OBJ_DIR)[A-Za-z0-9_\-]*.ln\)/$(\U\1\e_DIR)\2/e
:%s/ [^ ]*\/lib\/\([A-Za-z0-9_\-\.]*\)\/\([A-Za-z0-9_\-]*\.[hit]p*\)/$(\U\1\e_DIR)\2/e
:%s/ lib\/\([A-Za-z0-9_\-\.]*\)\/\([A-Za-z0-9_\-]*\.[hit]p*\)/$(\U\1\e_DIR)\2/e
:1s/.*/&/e
:wq
