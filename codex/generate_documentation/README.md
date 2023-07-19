# Examples of generating documentation from code

source for example code: https://github.com/statsmodels/statsmodels/blob/main/statsmodels/multivariate/multivariate_ols.py
Prepared source file by removing all docstrings and comments.

Also generated documentation for this program, `generate_documentation.py`.

## Prompts:
| Prompt                                                                                                                                                              | output file                                                                 |
|---------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------|
| produce a general description of the code and describe what it does                                                                                                 | documentation_dir/documentation1.txt, documentation_dir/documentation1a.txt |
| generate docstring following the PEP guidelines for each method or function.  do not include source code in the output.                                             | documentation_dir/documentation2.txt                                        |
| generate docstring following the PEP guidelines for each method or function.                                                                                        | documentation_dir/documentation3.txt                                        |
| produce a general description of the code and describe what it does.  output in markdown format. For each function, method or class show as a markdown bullet item. | documentation_dir/documentation4.md                                         |                                         |                                       |
| "produce a general description of the code and describe what it does.  output in html format. Paragraphs should be delineated by with <p> and </p> tags.             | documentation_dir/documentation5.html|                                       |                                       |


## Sample execution

Arguments:
- input source file
- output directory
- --prompt: LLM prompt, default is "produce a general description of the code and describe what it does"

Example execution:
```bash
python generate_documentation.py  source_files/example1.sas generated_documentation

python generate_documentation.py  source_files/example1.sas generated_documentation --prompt "produce docstring for each method or function"  

```

## Observations:
* model gpt-3.5-turbo resulted in this error, to correct used `gpt-3.5-turbo-16k`, which supports 16k tokens.
```text
starting to document multivariate_ols.py...with prompt 'produce a general description of the code and describe what it does'
Traceback (most recent call last):
  File "/opt/project/codex/generate_documentation/generate_documentation.py", line 53, in <module>
    documentation_text = generate_documentation(prompt, source_file)
  File "/opt/project/codex/generate_documentation/generate_documentation.py", line 19, in generate_documentation
    response = openai.ChatCompletion.create(
  File "/usr/local/lib/python3.9/site-packages/openai/api_resources/chat_completion.py", line 25, in create
    return super().create(*args, **kwargs)
  File "/usr/local/lib/python3.9/site-packages/openai/api_resources/abstract/engine_api_resource.py", line 153, in create
    response, _, api_key = requestor.request(
  File "/usr/local/lib/python3.9/site-packages/openai/api_requestor.py", line 298, in request
    resp, got_stream = self._interpret_response(result, stream)
  File "/usr/local/lib/python3.9/site-packages/openai/api_requestor.py", line 700, in _interpret_response
    self._interpret_response_line(
  File "/usr/local/lib/python3.9/site-packages/openai/api_requestor.py", line 763, in _interpret_response_line
    raise self.handle_error_response(
openai.error.InvalidRequestError: This model's maximum context length is 4097 tokens. However, you requested 5556 tokens (3508 in the messages, 2048 in the completion). Please reduce the length of the messages or completion.
```
* if output exceeds max token length parameter, the output is truncated.  Resolved by increasing the parameter `max_tokens` from 2K to 8K in the `openai.ChatCompletion.create()` call.

* generated documentation `documentation1.txt` file, for the most part, reflects the processing in the module.  Depending on the level of detailed required, there are the missing details:
  * Missing detail for `_multivariate_ols_test`.  The generated documentation does not identify the call to `_multivariate_test(hypotheses, exog_names, endog_names, fn)`, which in turns calls  `multivariate_stats(eigv2, p, q, df_resid)`.  In other words, there is no indication of the nested function calls.
  * Missing documentation for the internal function `_multivariate_test`.
  * Missing detail on call to `patsy.DesignInfo()`

* Generated documentation file `documentation3.txt`.  Depending on requirement, the prompt to generate documentation should prevent copying of original source code.  There is the possibility of dropping code when generating the documentation file.  In this example, the in-line comments are dropped. ![](images/stripped_in-line_comments.png)

* Generated documentation for program `generate_documentation.py` (file `documentation1a.txt`), for the most part, reflects the processing in the module.  It misses describing a step and incorrectly describes the final step in the module.
  * Missing description of issuing a message at the start of the processing
  * Incorrectly describes the final step as, "the code prints a messaging the start and completion of the documetnation process".  The correct description is "the code prints a message for completing the documentation process".

* Generating markdown or html (`documentation4.md` and `documentation5.html`, respectively).  Generates the same text.  Able to take advantage of simple Markdown directives.  However, html output is only raw text.  More resarch is needed in this area.  May need to utilize few shot learning techniques to get right formatting of output.