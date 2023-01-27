 test_add():
    assert add(2, 3) == 5
    assert add('space', 'ship') == 'spaceship'

# Unit test with pytest
def test_add_pytest():
    assert add(2, 3) == 5
    assert add('space', 'ship') == 'spaceship'

# Unit test with pytest and parametrize
@pytest.mark.parametrize("a, b, expected", [
    (2, 3, 5),
    ('space', 'ship', 'spaceship'),
])
def test_add_pytest_parametrize(a, b, expected):
    assert add(a, b) == expected

# Unit test with pytest and parametrize and ids
@pytest.mark.parametrize("a, b, expected", [
    (2, 3, 5),
    ('space', 'ship', 'spaceship'),
], ids=['integers', 'strings'])
def test_add_pytest_parametrize_ids(a, b, expected):
