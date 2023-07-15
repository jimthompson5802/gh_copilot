# Examples of SAS files to Python code conversion

Directories:
- sas_code: source SAS files
- converted_python: converted Python files

## Source for example SAS files
SAS User Guide [examples](https://documentation.sas.com/doc/en/pgmsascdc/9.4_3.5/procstat/titlepage.htm)

| SAS file | Description |
|---------|-------------|
| example1.sas | https://documentation.sas.com/doc/en/pgmsascdc/9.4_3.5/procstat/procstat_corr_examples02.htm |
| example2.sas| Handcrafted |


## Sample execution

Arguments:
- input SAS file
- output directory

Example execution:
```bash
python convert_sas_program.py  sas_files/example1.sas converted_python
```

