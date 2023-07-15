# Examples of SAS files to Python code conversion

Directories:
- sas_code: source SAS files
- converted_python: converted Python files

## Source for example SAS files
SAS User Guide [examples](https://documentation.sas.com/doc/en/pgmsascdc/9.4_3.5/procstat/titlepage.htm)

| SAS file | Description                                                                                                                                                             |
|---------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| example1.sas | https://documentation.sas.com/doc/en/pgmsascdc/9.4_3.5/procstat/procstat_corr_examples02.htm                                                                            |
| example2.sas| Handcrafted                                                                                                                                                             |
| macro_example1.sas | Handcrafted |
| sql_example1.sas | https://documentation.sas.com/doc/en/pgmsascdc/9.4_3.5/sqlproc/p015vwpsg8pas3n135iy1t43o1mc.htm.  Using prompt "convert sas program to python using sqlalchemy library" |


## Sample execution

Arguments:
- input SAS file
- output directory
- --prompt: LLM prompt, default is "convert this SAS program to Python"

Example execution:
```bash
python convert_sas_program.py  sas_files/example1.sas converted_python

python convert_sas_program.py  sas_files/example1.sas converted_python --prompt "convert sas program to python code using sklearn library"  

```

