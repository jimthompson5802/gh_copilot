data reformatted_data;
   infile 'your_file_path';
   input var1 var2 var3 var4 var5;

   /* Convert selected variables to strings */
   var1 = put(var1, $CHAR.);
   var2 = put(var2, $CHAR.);
   var3 = put(var3, $CHAR.);

   /* Keep var4 and var5 as numeric */

   /* Output the reformatted data */
   output;
run;
