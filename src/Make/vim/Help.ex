:set viminfo=
:set ul=0
:set ttyfast
:1s/.*/
:g/fake_out_glob/d
:%s/\.\<help\>/&.made/ge
:%s/^/        /e
:%s/\([^ ][^ ]*\) */\1 \\/e
:%s/^#.*//ge
:%s/[A-Za-z0-9_\-]/$(MAKE_DOC_DIR)&/e
:%s/ *$//ge
:$s/ *\\ */
:1s/.*/
:$s/.*/&
:wq