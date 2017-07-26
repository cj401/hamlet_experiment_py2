:set viminfo=
:set ul=0
:set ttyfast
:%s/^#.*//ge
:%s/\.h$/.h /e
:%s/\$(\([A-Z][A-Z0-9_]*\)_DIR)\([a-zA-Z0-9][a-zA-Z0-9_]*\)\.h/$(PCH_\1_DIR)\2.h$(PCH_SUFFIX)/ge
:%s/ \([a-zA-Z0-9][a-zA-Z0-9_]*\)\.h/ $(PCH_DIR)\1.h$(PCH_SUFFIX)/ge
:%s/\.h$(PCH_SUFFIX)  *$/.h$(PCH_SUFFIX)/e
:%s/ \([^ :]\)/ \\            \1/ge
:%s/^\([^ ]\)/\1/ge
:s/^.*: *$//ge
:%s/^.*\/lib\/\([A-Za-z0-9_\-]*\)\/\([A-Za-z0-9_\-\.]*.h:\)/$(\U\1\e_DIR)\2/e
:%s/^lib\/\([A-Za-z0-9_\-]*\)\/\([A-Za-z0-9_\-\.]*.h:\)/$(\U\1\e_DIR)\2/e
:%s/^.*\/lib\/\([A-Za-z0-9_\-]*\)\/\(\$([A-Z_]*OBJ_DIR)[A-Za-z0-9_\-\.]*\.o:\)/$(\U\1\e_DIR)\2/e
:%s/^lib\/\([A-Za-z0-9_\-]*\)\/\(\$([A-Z_]*OBJ_DIR)[A-Za-z0-9_\-\.]*\.o:\)/$(\U\1\e_DIR)\2/e
:%s/ [^ ]*\/lib\/\([A-Za-z0-9_\-]*\)\/\([A-Za-z0-9_\-\.]*\.[hit]p*\)/$(\U\1\e_DIR)\2/e
:%s/ lib\/\([A-Za-z0-9_\-]*\)\/\([A-Za-z0-9_\-\.]*\.[hit]p*\)/$(\U\1\e_DIR)\2/e
:1s/.*/&/e
:wq
