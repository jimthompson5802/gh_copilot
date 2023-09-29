       identification division.
         program-id.  read-file.
       data division.
            file section.
            fd  in-file.
            01  in-rec pic x(80).
       procedure division.
         begin.
                open input in-file
                read in-file
                at end
                    display "end of file"
                not at end
                    display in-rec
                end-read
                close in-file
                stop run.