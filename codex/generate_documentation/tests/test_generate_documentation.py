import pytest
from unittest.mock import call, patch, Mock, mock_open as mock_builtin_open

import openai
from openai import OpenAIError

from generate_documentation import generate_documentation, main


def mock_openai_call(*args, **kwargs):
    # Mocked OpenAI API response
    class MockedChoice:
        class MockedMessage:
            content = 'Mocked documentation content.'
        message = MockedMessage()

    class MockedResponse: 
        choices = [MockedChoice()]

    return MockedResponse()

# This decorator will mock the call to openai.ChatCompletion.create
@patch('openai.ChatCompletion.create', side_effect=mock_openai_call)
def test_generate_documentation(mock_openai):
    prompt = "Generate Python documentation for a function"
    source_code = "def greet(name):\n    '''This function greets the given name.'''\n    print('Hello,', name)"
    
    expected_documentation = 'Mocked documentation content.'
    actual_documentation = generate_documentation(prompt, source_code)

    assert actual_documentation == expected_documentation


def mock_openai_failure(*args, **kwargs):
    raise OpenAIError("Simulated OpenAI API failure")

@patch('openai.ChatCompletion.create', side_effect=mock_openai_failure)
def test_generate_documentation_failure(mock_openai):
    prompt = "Generate Python documentation for a function"
    source_code = "def greet(name):\n    '''This function greets the given name.'''\n    print('Hello,', name)"

    # pytest's raises checks that the expected exception is raised
    with pytest.raises(OpenAIError, match="Simulated OpenAI API failure"):
        generate_documentation(prompt, source_code)


# Another variant for unit tests
# using this prompt: using pytest generate a unit test for the function generate_documentation
def mock_openai_response():
    class MockResponse:
        def __init__(self):
            self.choices = [Mock(message=Mock(content="Generated Documentation"))]

    return MockResponse()

def test_generate_documentation2():
    with patch("openai.ChatCompletion.create", return_value=mock_openai_response()):
        prompt = "Generate Python documentation for a function"
        source_code = """
def greet(name):
    '''This function greets the given name.'''
    print('Hello,', name)
"""
        documentation = generate_documentation(prompt, source_code)
        assert "Generated Documentation" in documentation

def test_generate_documentation_api_error():
    with patch("openai.ChatCompletion.create", side_effect=Exception("API Error")):
        with pytest.raises(Exception) as e_info:
            prompt = "Generate Python documentation for a function"
            source_code = """
def greet(name):
    '''This function greets the given name.'''
    print('Hello,', name)
"""
            generate_documentation(prompt, source_code)
        assert str(e_info.value) == "API Error"


def test_main_function():
    """
    Test the main functionality of the main() function for generating documentation.

    This test function simulates the main functionality of the `main()` function by
    mocking various dependencies. It ensures that the `main()` function correctly
    interacts with the mocked objects, performs file operations, and generates
    documentation.

    Steps:
    1. Create mock objects for argparse arguments, parser, open function, and generate_documentation function.
    2. Configure mock objects with appropriate behavior and return values.
    3. Use patches to replace built-in functions and modules with mock objects.
    4. Execute the main() function within a controlled context.
    5. Perform assertions to verify the interactions and calls made during the execution.

    Assertions:
    - Ensure the argparse parser is called once to parse command-line arguments.
    - Verify the 'open' function is called with specific arguments.
    - Confirm the 'read' and 'write' methods are called on the mock open object.
    - Validate that the 'generate_documentation' function is called with specific arguments.

    """
 
    mock_args = Mock()
    mock_args.source_file = "source.sas"
    mock_args.documentation_file = "documentation.txt"
    mock_args.prompt = "test prompt"

    mock_parser = Mock()
    mock_parser.parse_args.return_value = mock_args

    mock_open = mock_builtin_open()

    mock_source_file_content = "Mocked source file content"
    mock_open().__enter__().read.return_value = mock_source_file_content

    mock_generate_documentation = Mock(return_value="Mocked documentation")

    with patch("argparse.ArgumentParser", return_value=mock_parser), \
         patch("builtins.open", mock_open), \
         patch("generate_documentation.generate_documentation", mock_generate_documentation):

        main()

    # check parse_args call
    mock_parser.parse_args.assert_called_once()
    
    # check open calls
    mock_open.assert_has_calls(
        [
            call("source.sas", "r"),
            call("documentation.txt", "w")
        ],
        any_order=True,
    )
    
    # check read and write calls
    mock_open().__enter__().read.assert_called_once()
    mock_open().__enter__().write.assert_called_once_with("Mocked documentation")

    # check generate_documentation call
    mock_generate_documentation.assert_called_once_with("test prompt", mock_source_file_content)



def test_main_function_error():
    """
    Test the error handling behavior of the main function when generating documentation.

    This test function simulates an error scenario in the `main()` function by
    mocking various dependencies. It checks that the `main()` function handles
    exceptions properly and interacts with mocked objects as expected.

    Steps:
    1. Create mock objects for argparse arguments, parser, open function, and generate_documentation function.
    2. Configure mock objects with appropriate behavior and side effects.
    3. Use patches to replace built-in functions and modules with mock objects.
    4. Execute the main() function within a pytest context manager that expects an exception.
    5. Perform assertions to verify the interactions and calls made during the execution.

    Assertions:
    - Ensure the argparse parser is called once to parse command-line arguments.
    - Verify the 'open' function is called with specific arguments.
    - Confirm the 'read' method is called on the mock open object.
    - Validate that the 'generate_documentation' function is called with specific arguments.

    """

    mock_args = Mock()
    mock_args.source_file = "source.sas"
    mock_args.documentation_file = "documentation.txt"
    mock_args.prompt = "test prompt"

    mock_parser = Mock()
    mock_parser.parse_args.return_value = mock_args

    mock_open = mock_builtin_open()

    mock_source_file_content = "Mocked source file content"
    mock_open().__enter__().read.return_value = mock_source_file_content

    mock_generate_documentation = Mock(side_effect=Exception("Mocked error"))

    with patch("argparse.ArgumentParser", return_value=mock_parser), \
         patch("builtins.open", mock_open), \
         patch("generate_documentation.generate_documentation", mock_generate_documentation):

        with pytest.raises(Exception, match="Mocked error"):
            main()


    mock_parser.parse_args.assert_called_once()

    # check open calls
    mock_open.assert_has_calls(
        [
            call("source.sas", "r"),
        ],
        any_order=True,
    )
    # check read call
    mock_open().__enter__().read.assert_called_once()

    # check generate_documentation call
    mock_generate_documentation.assert_called_once_with("test prompt", mock_source_file_content)
