:/^ *# *ifndef  *.*_INCLUDED
:/^ *# *define  *.*_INCLUDED
:/^ *# *ifdef __cplusplus
:/extern
:/^ *# *endif
:s/.*/&/
:s/.*/&/
:s/.*/&/
:s/.*/&\/\*/
:s/.*/& \* A list of all exported includes maintained by the fix_incl script\./
:s/.*/& \* (Do not edit this comment. The fix_incl script uses it\.)/
:s/.*/&\*\//
:s/.*/&/
:$
:?^ *# *endif
:?^ *# *endif
:?^ *}
:?^ *# *ifdef __cplusplus
:s/.*/&/
-
:s/.*/&/
-
:s/.*/\*\/&/
-
:s/.*/ \* (Do not edit this comment. The fix_incl script uses it\.)&/
-
:s/.*/ \* End of list of all exported includes maintained by the fix_incl script\.&/
-
:s/.*/\/\*&/
-
:s/.*/&/
:wq
