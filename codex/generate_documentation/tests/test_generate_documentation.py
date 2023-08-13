import pytest
from unittest.mock import patch, Mock

import openai
from openai import OpenAIError

from codex.generate_documentation.generate_documentation import generate_documentation


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

