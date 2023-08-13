# Testbed for LLM generated code reviews

## Observations:

* Due to randomness of LLM generated output, difficult to get consistent feedback. 

* For a few test runs, a hallucination was observed in the comments, e.g., `temperature` was never hard-coded to 0.0.
```text
4. The code includes a hard-coded value of `0.0` for the `temperature` parameter in the API request. It may be beneficial to make this configurable or experiment with different temperature values to get the desired output.
```

* The differences in output relate to the level of detail provided in the generated review comments.  With low temperature, e.g., 0.01, comments are more consistent from run-to-run.  OTOH, higher temperature, e.g., 1.0 (default) the comments provide more feedback but have higher variability from run-to-run.  In the example below, the `temperature=0.5` is better because it is more aligned to the intent of the `print` statement.
```text
temperature=0.1
- The function includes a `print` statement that outputs the token size. This print statement should be removed or commented out, as it is not necessary for the function to work correctly.

temperature=0.5  
6. The code uses `print` statements for debugging purposes. It would be better to use a logging library instead, so that the debugging information can be easily controlled and configured.
```

## ChatGPT recommended prompts

```text
I have a Python code snippet that I'd like you to review. Please provide feedback on any errors, potential bugs, or areas for improvement:
[YOUR CODE HERE]

I'm looking to optimize this Python code for better performance. Can you review the following code and suggest ways to make it faster and more efficient?
[YOUR CODE HERE]

I'm concerned about potential security vulnerabilities in my Python code. Could you review the following code and highlight any security issues or best practices I should be aware of?
[YOUR CODE HERE]

I want to ensure that my Python code adheres to PEP 8 style guidelines and general best practices. Please review the following code and provide feedback on style, readability, and best practices:
[YOUR CODE HERE]

I have a Python code snippet and I'm specifically concerned about error handling. Can you review this aspect and provide feedback on error handling?
[YOUR CODE HERE]

```