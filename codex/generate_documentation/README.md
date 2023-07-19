# Examples of generating documentation from code

source for example code: https://github.com/statsmodels/statsmodels/blob/main/statsmodels/multivariate/multivariate_ols.py
Prepared source file by removing all docstrings and comments.

## Prompts:
| Prompt | output file                    |
|--------|--------------------------------|
|produce a general description of the code and describe what it does| documentation_dir/example1.txt |
|generate docstring following the PEP guidelines for each method or function.  do not include source code in the output.| documentation_dir/example2.txt |
|generate docstring following the PEP guidelines for each method or function.| documentation_dir/example3.txt |

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
* if output exceeds max token length parameter, the output is truncated.