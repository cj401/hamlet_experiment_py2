
#include "l/l_incl.h"


/*
#define DEBUG
*/

int main(int argc, char** argv)
{
    char  line[ 10000 ];
    char* line_pos;
    char* possible_code_start;
    char  name[ 1000 ];
    int   slash_found;
    int   star_found;
    int   blank_found;
    int   in_typedef = FALSE;
    int   first_item_in_typedef = TRUE;
    int   comment_beg_col;
    int   comment_end_col;
    int   i;
    Queue_element* typdef_queue_head = NULL;
    Queue_element* typdef_queue_tail = NULL;
    char  author[ 1000 ];
    char  documentor[ 1000 ];
    int   next_line_is_author = FALSE;
    int   next_line_is_documentor = FALSE;
    int   have_documentor = FALSE;
    int   in_comment = FALSE;
    int   comment_end;
    int   first_pound_found = FALSE; /* Ignore copyright header. */
    int   prev_line_not_formatted = FALSE;
    char  save_line_break_line[ 1000 ];
    /*
    // Count is a state variable which is 0 to start,  one at the begining of a
    // block comment, two after the first line in the block, three in the
    // middle of the block, four at the end of the block and five after the
    // block, but before the code, and six in code.
    */
    int   count = 0;
    int   has_index_field = FALSE; 
    FILE* index_fp      = NULL;
    char  index_file_name[ MAX_FILE_NAME_SIZE ]; 
    int doing_html = FALSE;
    int option = NOT_SET;
    char value_buff[ MAX_FILE_NAME_SIZE ]; 
    extern int kjb_debug_level; 

#ifdef DEBUG
    kjb_debug_level = 2;
#endif 


    BUFF_CPY(name, "(unknown)");
    BUFF_CPY(author, "Kobus Barnard");
    BUFF_CPY(documentor, "Kobus Barnard");
    BUFF_CPY(index_file_name, "index"); 

    while ((option = kjb_getopts(argc, argv, "-hi:", NULL,
                                 value_buff, sizeof(value_buff)))
               != EOF
          )
    {
        switch (option)
        {
            case ERROR :
                set_error("Program has invalid arguments.");
                return ERROR;
            case 'h' :
                doing_html = TRUE;
                break;
            case 'i' :
                BUFF_CPY(index_file_name, value_buff);
                break;
        }
    }   

    while (BUFF_FGET_LINE(stdin, line) != EOF)
    {
        line_pos        = line;
        slash_found     = FALSE;
        star_found      = FALSE;
        blank_found     = FALSE;
        comment_beg_col = FALSE;
        comment_end_col = FALSE;

        /* Skip copyright header--it looks too much like a documenation block.  */

        /* 
         * Technically, we remove all comments that preceed the first #include.
        */

        trim_end(line);

        comment_beg_col = find_char_pair(line, '/', '*');
        comment_end_col = find_char_pair(line, '*', '/');

        /*
        // Check  for a comment in first column, followed by code (as is
        // often the case in complex #ifdef's in Kobus's code). These
        // screw up c2man, so replace the comments with blanks.
        */
        if (    (comment_beg_col == 1)
             && (comment_end_col)
             && (    (line[ comment_end_col + 1 ] != '\0')
                  || (comment_end_col < 16)
                )
           )
        {
            for (i=0; i<comment_end_col + 1; i++)
            {
                line[ i ] = ' ';
            }
        }

        line_pos = line;
        trim_beg(&line_pos);

        /*
        // Additional hacks added after the program was essentally done.
        // Some tests as to the nature of the line. The result of these tests
        // are not necessarily used.
        */
        comment_beg_col = find_char_pair(line, '/', '*');
        comment_end_col = find_char_pair(line, '*', '/');

        if (comment_beg_col == 1)
        {
            in_comment = TRUE;
        }

        if (comment_end_col > comment_beg_col)
        {
            comment_end = TRUE;
        }
        else
        {
            comment_end = FALSE;
        }

        if (in_comment)
        {
            if (! first_pound_found) 
            {
                if (comment_end) 
                {
                    in_comment = FALSE;
                }
                continue;
            }
        }
        else if (line[ 0 ] == '#')
        {
            first_pound_found = TRUE;
        }

        possible_code_start = line_pos;

        /*
        // Risky, now that we don't look for a paren on the same line.
        */
        trim_beg(&possible_code_start);

        if (*line_pos == '\0') blank_found = TRUE;

        if (*line_pos == '/') slash_found = TRUE;
        while (*line_pos == '/') line_pos++;

        trim_beg(&line_pos);

        if (*line_pos == '*') star_found = TRUE;
        while (*line_pos == '*') line_pos++;

        trim_beg(&line_pos);

#ifdef DEBUG
        kjb_fprintf(stderr, "----------------\n");
        kjb_fputs(stderr, "LINE: ->");
        kjb_fputs(stderr, line);
        kjb_fprintf(stderr, "\n");
        kjb_fputs(stderr, "LPOS: ->");
        kjb_fputs(stderr, line_pos);
        kjb_fprintf(stderr, "\n");
        kjb_fputs(stderr, "CODE: ->");
        kjb_fputs(stderr, possible_code_start);
        kjb_fprintf(stderr, "\n");
        dbc(*possible_code_start);
        dbi(count);
        dbi(comment_beg_col);
        dbi(comment_end_col);
#endif

        if ((prev_line_not_formatted) && (*line_pos != '|'))
        {
            put_line(save_line_break_line);

            if (doing_html)
            {
                put_line("</pre>"); 
            }
        }

        prev_line_not_formatted = FALSE; /* Until noted otherwise. */

        if (line[ 0 ] == '#')
        {
            put_line(line);

            /* Deal with continued defines. */
            while (last_char(line) == '\\')
            {
                if (BUFF_FGET_LINE(stdin, line) == EOF) break;

                put_line(line);
            }
        }
        else if (in_typedef)
        {
            insert_at_end_of_queue(&typdef_queue_head, &typdef_queue_tail,
                                   (void*)kjb_strdup(line));

            if (*line_pos == '}')
            {
                Queue_element* cur_elem = typdef_queue_head;

                put_line("\n);");
                in_typedef = FALSE;
                first_item_in_typedef = TRUE;


                while (cur_elem != NULL)
                {
                    put_line((char*)(cur_elem->contents));
                    cur_elem = cur_elem->next;
                }

                free_queue(&typdef_queue_head, &typdef_queue_tail, kjb_free);
                /* Swallow the next line. */
                /* BUFF_FGET_LINE(stdin, line); */
            }
            else if (*line_pos == '{')
            {
                put_line("(");
            }
            else if (blank_found || slash_found)
            {
                /*EMPTY*/
                ; /* Do nothing. */
            }
            else
            {
                char_for_char_translate(line, ';', ' ');
                trim_end(line);

                if (first_item_in_typedef)
                {
                    first_item_in_typedef = FALSE;
                }
                else
                {
                    kjb_puts(",\n");
                }

                kjb_puts(line);
            }
        }
        else if (    (count == 5)
                  && (HEAD_CMP_EQ(possible_code_start, "typedef"))
                )
        {
            char typedef_type[ 100 ];

            count = 6;
            dbs("Start of typdef");
            dbi(count);


            line_pos += 7;
            trim_beg(&line_pos);
           
            /* Jump over "struct" or "union". */
            BUFF_GET_TOKEN(&line_pos, typedef_type); 

            if ((STRCMP_EQ(typedef_type, "struct")) || (STRCMP_EQ(typedef_type, "union")))
            {
                dbs(typedef_type); 

                trim_beg(&line_pos);
                trim_end(line_pos);

                if ((FIND_CHAR_YES(line_pos, ';')) || (FIND_CHAR_YES(line_pos, '{')))
                {
                    p_stderr("\nWarning: Typedef struct or union needs to continue onto next line with the '{' to be documented.\n\n");
                    kjb_puts(line);
                    kjb_puts("\n");
                }
                else
                {
                    /* OK, this looks generic enough for dumb parsing. So, we
                     * translate 
                     *      typedef struct xxx
                     * to
                     *      typedef_struct xxx_mock_function
                    */

                    kjb_puts("typedef_");
                    kjb_puts(typedef_type);
                    kjb_puts(" ");

                    kjb_puts(line_pos);
                    kjb_puts("_mock_function");
                    kjb_puts("\n");

                    insert_at_end_of_queue(&typdef_queue_head, &typdef_queue_tail,
                                           (void*)kjb_strdup(line));
                    in_typedef = TRUE;
                }
            }
            else
            {
                kjb_puts(line);
            }
        }
        /*
        // Test for start of code following a comment block. Note that we have
        // to test for code other than a function header, as any stuff between
        // the block and the first code is stripped out because it confuses
        // c2man.
        */
        else if (    (count == 5)
                  && (isalpha(*possible_code_start))
                     /*
                     // This no longer works, as some code starts have the paren
                     // on the next line.
                     //
                     && (FIND_CHAR_YES(possible_code_start, '('))
                     */
                )
        {
            count = 6;

            dbs("Start of code");
            dbi(count);

            fput_line(stdout, line);
        }
        /*
        // Next test is for anything after the comment block but before the
        // code.  Comments here confuse c2man, so make them invisible.
        */
        else if (count == 5)
        {
            dbs("End of code skipping non # line before code");
            dbi(count);
        }
        /*
        // Next test is for lines with ==>. These are to be protected from
        // c2man, so remove them.
        */
        else if (HEAD_CMP_EQ(line_pos, "==>"))
        {
            /*EMPTY*/
            ; /* Do nothing */
        }
        /*
        // Next test for the begining of the block.
        */
        else if (HEAD_CMP_EQ(line_pos, "====="))
        {
            count = 1;
            char_for_char_translate(line, (int)'=', (int)' ');

            /*
            // It may be the case that we have the slash and star on a previous
            // line. Pretend that we got it all in this line, as the start of
            // the comment will not have been output.
            */
            line[ 0 ] = '/';
            line[ 1 ] = '*';
            fput_line(stdout, line);
        }
        /*
        // If the above failed and count is 0, then we are not in a comment
        // block. In general, other comments should not be seen by c2man.
        */
        else if ((count == 0) && (in_comment))
        {
            /*EMPTY*/
            ; /* Do nothing */
        }
        /*
        // This test is for the end of the block. If the end comes after a
        // middle (IE, count is 3), the we pretend that we had a disclaimer,
        // author and a copyright section,
        */
        else if (    (count > 0) && (count <= 3)
                  && (HEAD_CMP_EQ(line_pos, "-----"))
                )
        {
            if (    (! has_index_field)
                 && (! IC_HEAD_CMP_EQ(name, "STATIC")) 
                 && (! IC_HEAD_CMP_EQ(name, "(unknown)")) 
               )
            {
                if (index_fp == NULL)
                {
                    NPETE(index_fp = kjb_fopen(index_file_name, "w"));
                }

                kjb_fprintf(index_fp, "%-35s 0. Not yet categorized via the 'Index' field.\n", name);
            }

            dbi(comment_end_col);
            dbs(line);

            if (comment_end_col) 
            {
                line[ comment_end_col - 1 ] = '\0';
            }

            dbs(line_pos);
            char_for_char_translate(line, (int)'-', (int)' ');
            dbs(line);

            if (count == 3)
            {
                *line_pos = '\0';

                kjb_puts(line);
                kjb_puts("\n");

                kjb_puts(line);
                kjb_puts("Disclaimer:");
                kjb_puts("\n");

                kjb_puts(line);
                kjb_puts(
                         "    This software is not adequatedly tested. ");
                kjb_puts("It is recomended that ");
                kjb_puts("\n");

                kjb_puts(line);
                kjb_puts("    results are checked independantly where appropriate.");
                kjb_puts("\n");

                kjb_puts(line);
                kjb_puts("\n");

                kjb_puts(line);
                kjb_puts("\n");

                kjb_puts(line);
                kjb_puts("Author:");
                kjb_puts("\n");

                kjb_puts(line);
                kjb_puts("    ");
                kjb_puts(author);
                kjb_puts("\n");

                kjb_puts(line);
                kjb_puts("\n");

                kjb_puts(line);

                /*
                 * Documentor or Documenter? Usage varies. Seems to be a trend
                 * towards Documenter"
                */
                /* kjb_puts("Documentor:"); */
                kjb_puts("Documenter:"); 

                kjb_puts("\n");

                kjb_puts(line);
                kjb_puts("    ");
                kjb_puts(documentor);
                kjb_puts("\n");
                if (comment_end_col) kjb_puts("*/");
                kjb_puts("\n");

                BUFF_CPY(name, "(unknown)");
            }

            if (comment_end_col)
            {
                count = 5;
            }
            else 
            {
               count = 4;
            }

            /*
             * Kobus: 06/02/15:
             *
             * Bug found! We need to reset the defaults, otherwise, the author
             * of one routine is propagated to the rest of the routines in the
             * file.
             */
            BUFF_CPY(name, "(unknown)");
            BUFF_CPY(author, "Kobus Barnard");
            BUFF_CPY(documentor, "Kobus Barnard");
            have_documentor = FALSE;
            has_index_field = FALSE; 
        }
        /*
        // Next test is for first line after the begining of the block.
        // Make this invisible to c2man.
        */
        else if ((count == 1) && (slash_found || star_found || blank_found))
        {
            if (*line_pos)
            {
                BUFF_CPY(name, line_pos);
            }

            count = 2;
            /* Skip this line */
        }

        /*
        // Next test is for second line after the begining of the block.
        // Make this invisible to c2man.
        */
        else if (    (count == 2)
                     && (slash_found || star_found || blank_found)
                     && ( ! comment_beg_col)
                     && ( ! comment_end_col)
                )
        {
            if (*line_pos)
            {
                BUFF_CPY(name, line_pos);
            }

            /* Skip this line */
            count = 3;
        }

        /*
        // Make it so that lines starting with "|" are not formatted.
        */
        else if ((count == 3) && (*line_pos == '|'))
        {
            if (doing_html)
            {
                put_line(line_pos);

                if (comment_end_col)
                {
                    put_line(" */");
                }
            }
            else
            {
                char line_copy[ 1000 ];

                BUFF_CPY(line_copy, " ");
                /* Include the | otherwise setting up the spacing is a pain. */
                BUFF_CAT(line_copy, line_pos);

                strcpy(line_pos, ".br");

                if (comment_end_col)
                {
                    strcat(line_pos, " */");
                }
                else
                {
                    /* Could be part of a multi-preformatted comment. */
                    prev_line_not_formatted = TRUE;
                    BUFF_CPY(save_line_break_line, line);
                }

                put_line(line);
                put_line(line_copy);
            }
        }

        /*
        // Test is for "Index:". This is hidden from c2man. The line is used
        // instead to generate the index.
        */
        else if (    (count == 3)
                  && (    (IC_HEAD_CMP_EQ(line_pos, "index:"))
                       || (IC_HEAD_CMP_EQ(line_pos, "index :"))
                     )
                )
        {
            char phrase[ 1000 ];

            has_index_field = TRUE;

            if (index_fp == NULL)
            {
                NPETE(index_fp = kjb_fopen(index_file_name, "w"));
            }

            line_pos += 5;   /* len("index"); */

            trim_beg(&line_pos);

            if (*line_pos == ':')
            {
                line_pos++;
                trim_beg(&line_pos);
            }

            while (BUFF_GEN_GET_TOKEN_OK(&line_pos, phrase, ",."))
            {
                kjb_fprintf(index_fp, "%-35s %s\n", name, phrase);
            }
        }

#ifdef HOW_IT_WAS_10_07_03
        /* This requires some c2man changes that seem to have disappeared or
         * perhaps were only planned and never implemented.
        */

        /*
        // Test for "see also" section. This needs to be fudged a bit
        // in cooperation with changes to c2man itself
        */
        else if ((count == 3) && (IC_HEAD_CMP_EQ(line_pos, "see also:")))
        {
            *line_pos = '\0';

            kjb_puts(line);
            kjb_puts("SEEALSO:");
            if (comment_end_col) kjb_puts("*/");
            kjb_puts("\n");
        }
#else 
        else if ((count == 3) && (IC_HEAD_CMP_EQ(line_pos, "see also:")))
        {
            *line_pos = '\0';

            kjb_puts(line);
            kjb_puts("Related:");
            if (comment_end_col) kjb_puts("*/");
            kjb_puts("\n");
        }
#endif 

        else if (next_line_is_author)
        {
            next_line_is_author = FALSE;
            trim_beg(&line_pos);
            BUFF_CPY(author, line_pos);

            /*
             * Assume that the documentor is the author.
             */
            if (! have_documentor)
            {
                BUFF_CPY(documentor, line_pos);
            }
        }

        /*
        // Test for "author" to over-ride default.
        */
        else if (    (count == 3)
                     && (     (IC_HEAD_CMP_EQ(line_pos, "author:"))
                              ||  (IC_HEAD_CMP_EQ(line_pos, "authors:"))
                              ||  (IC_HEAD_CMP_EQ(line_pos, "author :"))
                              ||  (IC_HEAD_CMP_EQ(line_pos, "authors :"))
                        )
                )
        {
            line_pos += find_char(line_pos, ':');
            trim_beg(&line_pos);

            if (*line_pos != '\0')
            {
                BUFF_CPY(author, line_pos);
            }
            else
            {
                next_line_is_author = TRUE;
            }
        }

        else if (next_line_is_documentor)
        {
            next_line_is_documentor = FALSE;
            trim_beg(&line_pos);
            BUFF_CPY(documentor, line_pos);
        }

        /*
        // Test for "documentor" to over-ride default.
        */
        else if (    (count == 3)
                     && (         (IC_HEAD_CMP_EQ(line_pos, "documentor:"))
                              ||  (IC_HEAD_CMP_EQ(line_pos, "documentor :"))
                              ||  (IC_HEAD_CMP_EQ(line_pos, "documentors:"))
                              ||  (IC_HEAD_CMP_EQ(line_pos, "documentors :"))
                              ||  (IC_HEAD_CMP_EQ(line_pos, "documenter:"))
                              ||  (IC_HEAD_CMP_EQ(line_pos, "documenter :"))
                              ||  (IC_HEAD_CMP_EQ(line_pos, "documenters:"))
                              ||  (IC_HEAD_CMP_EQ(line_pos, "documenters :"))
                              ||  (IC_HEAD_CMP_EQ(line_pos, "documented:"))
                              ||  (IC_HEAD_CMP_EQ(line_pos, "documented :"))
                              ||  (IC_HEAD_CMP_EQ(line_pos, "documented by:"))
                              ||  (IC_HEAD_CMP_EQ(line_pos, "documented by :"))
                        )
                )
        {
            line_pos += find_char(line_pos, ':');
            trim_beg(&line_pos);

            if (*line_pos != '\0')
            {
                BUFF_CPY(documentor, line_pos);
            }
            else
            {
                next_line_is_documentor = TRUE;
            }

            have_documentor = TRUE;
        }

        /*
        // Force first line after block to be output. It will often be the
        // the trailing comment end; at worst it will be a blank.
        */
        else if (count == 4)
        {
            count = 5;
            put_line(line);
        }

        /*
        // The special end of function string is skipped.
         */
        else if (    (comment_beg_col) && (comment_end_col)
                     && (HEAD_CMP_EQ(line_pos, "/\\ /\\"))
                )
        {
            dbs("Skipping jagged");
            dbi(count);

            /*EMPTY*/
            ; /* Do nothing */
        }

        /*
        // The line of dashes used to separate stuff is also ignored.
        */
        else if (    (comment_beg_col) && (comment_end_col)
                     && (HEAD_CMP_EQ(line_pos, "-----"))
                )
        {
            dbs("Skipping hyphens");
            dbi(count);

            /*EMPTY*/
            ; /* Do nothing */
        }

        /*
        // If all the above fails, just echo the line.
        */
        else
        {
            put_line(line);
        }

        if (comment_end_col) in_comment = FALSE;
    }

    kjb_fclose(index_fp);

    return EXIT_SUCCESS;
}


