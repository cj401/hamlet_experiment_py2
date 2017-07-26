:set viminfo=
:set ul=0
:set ttyfast
:%s/ *$/ /e
:%s/ *: */ : /e
:%s/-/-xxx-hyphen-yyy-/ge
:%s/ [^ ]*\/lib\/\([A-Za-z0-9_\-]*\)\/\([a-z_A-Z0-9\-\.]*\.\<[Ccohit]p*x*c*\>\)/ $(\U\1\e_DIR)\2/ge
:%s/^[^ ]*\/lib\/\([A-Za-z0-9_\-]*\)\/\([a-z_A-Z0-9\-\.]*\.\<[Ccohit]p*x*c*\>\)/$(\U\1\e_DIR)\2/ge
:%s/ lib\/\([A-Za-z0-9_\-]*\)\/\([a-z_A-Z0-9\-\.]*\.\<[Ccohit]p*x*c*\>\)/ $(\U\1\e_DIR)\2/ge
:%s/^lib\/\([A-Za-z0-9_\-]*\)\/\([a-z_A-Z0-9\-\.]*\.\<[Ccohit]p*x*c*\>\)/$(\U\1\e_DIR)\2/ge
:%s/ \([A-Za-z0-9_\-]*\)\/\([a-z_A-Z0-9\-\.]*\.\<[Ccohit]p*x*c*\>\)/ $(\U\1\e_DIR)\2/ge
:%s/^\([A-Za-z0-9_\-]*\)\/\([a-z_A-Z0-9\-\.]*\.\<[Ccohit]p*x*c*\>\)/$(\U\1\e_DIR)\2/ge
:%s/ \.\.\/\([A-Za-z1-9_\-]*\)\/\([A-Za-z0-9_\-\.]*\.[Ccohit]p*x*c*\) *$/ $(\U\1\e_DIR)\2 /ge
:%s/^\.\.\/\([A-Za-z1-9_\-]*\)\/\([A-Za-z0-9_\-\.]*\.[Ccohit]p*x*c*\) *$/$(\U\1\e_DIR)\2 /ge
:%s/ \/[^ ]*//ge
:%s/-xxx-hyphen-yyy-/-/ge
:%s/-XXX-HYPHEN-YYY-/___/ge
:%s/ *$//ge
:wq
