import pytest

from PALSYN.preprocessing.log_preprocessing import (
    extract_epsilon_from_string,
    find_noise_multiplier,
)


def test_extract_epsilon_from_string_parses_float() -> None:
    text = """
    Privacy Accounting Report
    Epsilon assuming Poisson sampling (*): 1.2345
    Delta: 1e-5
    """
    parsed = extract_epsilon_from_string(text)
    print(f"Parsed epsilon (float): {parsed}")
    assert parsed == pytest.approx(1.2345)


def test_extract_epsilon_from_string_handles_scientific_notation() -> None:
    text = "Epsilon assuming Poisson sampling (*): 3.2e-2 (DP-SGD)"
    parsed = extract_epsilon_from_string(text)
    print(f"Parsed epsilon (scientific): {parsed}")
    assert parsed == pytest.approx(3.2e-2)


def test_extract_epsilon_from_string_defaults_when_missing() -> None:
    with pytest.warns(RuntimeWarning):
        parsed = extract_epsilon_from_string("No epsilon reported here.")
        print(f"Parsed epsilon (missing -> default): {parsed}")
        assert parsed == 0.0


def test_find_noise_multiplier_converges_with_stub() -> None:
    def fake_privacy_statement(**kwargs):
        noise = kwargs["noise_multiplier"]
        epsilon = 1.0 / noise
        return f"Epsilon assuming Poisson sampling (*): {epsilon}"

    result = find_noise_multiplier(
        target_epsilon=0.25,
        num_examples=1000,
        batch_size=32,
        epochs=5,
        tol=1e-6,
        privacy_statement_fn=fake_privacy_statement,
    )
    print(f"Noise multiplier (converged): {result}")
    assert result == pytest.approx(4.0, rel=1e-3)


def test_find_noise_multiplier_warns_when_not_converged() -> None:
    def stubborn_privacy_statement(**kwargs):
        return "Epsilon assuming Poisson sampling (*): 10.0"

    with pytest.warns(RuntimeWarning):
        result = find_noise_multiplier(
            target_epsilon=1.0,
            num_examples=1000,
            batch_size=32,
            epochs=5,
            max_iter=5,
            privacy_statement_fn=stubborn_privacy_statement,
        )
    print(f"Noise multiplier (fallback upper bound): {result}")
    assert result == pytest.approx(100.0)


def test_find_noise_multiplier_validates_inputs() -> None:
    with pytest.raises(ValueError):
        find_noise_multiplier(
            target_epsilon=0.5,
            num_examples=0,
            batch_size=32,
            epochs=5,
        )
