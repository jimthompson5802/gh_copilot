macro generate_regression_models(input_files);
  %local i;

  %do i = 1 %to %sysfunc(countw(&input_files));
    %let input_file = %scan(&input_files, &i);

    data mydata;
      infile "&input_file" dlm=',' firstobs=2;
      input x y;
    run;

    proc reg data=mydata outest=outest&i;
      model y = x;
    run;

    %put Linear regression model for &input_file has been generated;
  %end;

  %put All regression models have been generated successfully;
%mend;

%generate_regression_models(input_files = "path/to/file1.csv path/to/file2.csv path/to/file3.csv");
