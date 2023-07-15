%macro regression_macro(files=);
  %local dsid rc;

  /* Create an empty dataset to store the regression results */
  data regression_results;
    set &files.(obs=0);
  run;

  /* Loop through each file */
  %let dsid = %sysfunc(open(&files));
  %if &dsid > 0 %then %do;
    %let rc = %sysfunc(fetch(&dsid));
    %do %while (&rc = 0);
      /* Perform linear regression */
      proc reg data=&files.(keep=x1 x2 x3 x4 y) outest=reg_results_&sysindex.;
        model y = x1 x2 x3 x4 / noprint;
        output out=pred_results_&sysindex. predicted=y_pred;
      run;

      /* Append the regression results to the final dataset */
      data regression_results;
        merge regression_results reg_results_&sysindex.;
      run;

      %let rc = %sysfunc(fetch(&dsid));
    %end;
    %let rc = %sysfunc(close(&dsid));
  %end;

  /* View the regression results */
  proc print data=regression_results;
  run;
%mend;

/* Example usage */
%regression_macro(files=file1 file2 file3);
