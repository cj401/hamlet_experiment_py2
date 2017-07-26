:set viminfo=
:set ul=0
:set ttyfast
:1s/.*/&/e
:g/fake_out_glob/d
:%s/\.\<help\>/&.made/ge
:%s/^/        /e
:%s/\([^ ][^ ]*\) */\1 \\/e
:%s/^#.*//ge
:%s/[A-Za-z0-9_\-]/$(MAKE_DOC_DIR)&/e
:%s/ *$//ge
:$s/ *\\ *//e
:1s/.*/MAN_FROM_HELP_FILES = \\/e
:$s/.*/&/e
:wq
