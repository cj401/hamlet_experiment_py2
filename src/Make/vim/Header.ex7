:set viminfo=
:set ul=0
:set ttyfast
:%s/_incl\.o/_incl.h/ge
:%s/gen\.o *:/gen.h:/ge
:%s/^\([a-z]\)\/\1_gen\.h\( *:.*\) \1\/\1_gen\.h/\1\/\1_gen.h\2/ge
:%s/^\([a-z]\)\/\1_gen\.h\( *:.*\) .*\/\1\/\1_gen\.h/\1\/\1_gen.h\2/ge
:%s/^.*\/\([a-z]\)\/\1_gen\.h\( *:.*\) \1\/\1_gen\.h/\1\/\1_gen.h\2/ge
:%s/^.*\/\([a-z]\)\/\1_gen\.h\( *:.*\) .*\/\1\/\1_gen\.h/\1\/\1_gen.h\2/ge
:%s/^\(.*\)\/gen\.h\( *:.*\) [^ ]\+\/gen\.h/\1\/gen.h\2/ge
:%s/^\(.*\)\/gen\.h\( *:.*\) gen\.h/\1\/gen.h\2/ge
:%s/^gen\.h\( *:.*\) [^ ]\+\/gen.h/gen.h\1/ge
:%s/^gen\.h\( *:.*\) gen\.h/gen.h\1/ge
:wq
