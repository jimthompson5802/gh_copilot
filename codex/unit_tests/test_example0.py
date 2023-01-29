 test_add():
    assert add(2, 3) == 5
    assert add('space', 'ship') == 'spaceship'

# Unit test with pytest
def test_add_pytest():
    assert add(2, 3) == 5
    assert add('space', 'ship') == 'spaceship'

# Unit test with pytest and parameterized
@pytest.mark.parametrize("a, b, expected", [
    (2, 3, 5),
    ('space', 'ship', 'spaceship'),
])
def test_add_pytest_parametrized(a, b, expected):
    assert add(a, b) == expected

# Unit test with pytest and parameterized and fixture
@pytest.mark.parametrize("a, b, expected", [
    (2, 3, 5),
    ('space', 'ship', 'spaceship'),
])
def test_add_pytest_parametrized_fixture(a, b, expected, capsys):
    assert add(a, b) == expected
    out, err = capsys.readouterr()
    assert out == '5\n'
    assert err == ''

# Unit test with pytest and parameterized and fixture and scope
@pytest.mark.parametrize("a, b, expected", [
    (2, 3, 5),
    ('space', 'ship', 'spaceship'),
])
def test_add_pytest_parametrized_fixture_scope(a, b, expected, capsys):
    assert add(a, b) == expected
    out, err = capsys.readouterr()
    assert out == '5\n'
    assert err == ''

# Unit test with pytest and parameterized and fixture and scope and mark
@pytest.mark.parametrize("a, b, expected", [
    (2, 3, 5),
    ('space', 'ship', 'spaceship'),
])
@pytest.mark.add
def test_add_pytest_parametrized_fixture_scope_mark(a, b, expected, capsys):
    assert add(