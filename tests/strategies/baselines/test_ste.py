from unittest.mock import MagicMock

import numpy as np
from numpy.testing import assert_array_equal

from pyclad.strategies.baselines.ste import SingleTaskExpertStrategy
from tests.strategies.baselines.mock_model import MockModel


class NamedMockModel(MockModel):
    def __init__(self, name: str, predictions=None):
        self._name = name
        self.fit = MagicMock()
        self.predict = MagicMock(return_value=predictions)

    def name(self) -> str:
        return self._name


def test_creating_fresh_model_for_each_concept():
    base_model = NamedMockModel("base")
    expert_a = NamedMockModel("expert-a")
    expert_b = NamedMockModel("expert-b")
    model_fn = MagicMock(side_effect=[expert_a, expert_b])

    strategy = SingleTaskExpertStrategy(model=base_model, model_creation_fn=model_fn)
    first_data = np.array([[1, 2, 3], [4, 5, 6]])
    second_data = np.array([[7, 8, 9], [10, 11, 12]])

    strategy.learn(first_data)
    strategy.learn(second_data)

    assert model_fn.call_count == 2
    expert_a.fit.assert_called_once_with(first_data)
    expert_b.fit.assert_called_once_with(second_data)
    assert strategy.current_model() is expert_b
    assert strategy.info()["strategy"]["training_runs"] == 2
    assert strategy.info()["strategy"]["reset_before_each_concept"] is True


def test_returning_predictions_from_latest_expert():
    base_model = NamedMockModel("base")
    predictions_a = (np.array([0, 1]), np.array([0.1, 0.9]))
    predictions_b = (np.array([1, 0]), np.array([0.8, 0.2]))
    expert_a = NamedMockModel("expert-a", predictions=predictions_a)
    expert_b = NamedMockModel("expert-b", predictions=predictions_b)

    strategy = SingleTaskExpertStrategy(
        model=base_model,
        model_creation_fn=MagicMock(side_effect=[expert_a, expert_b]),
    )

    strategy.learn(np.array([[1, 1], [2, 2]]))
    assert_array_equal(strategy.predict(np.array([[3, 3], [4, 4]])), predictions_a)

    strategy.learn(np.array([[5, 5], [6, 6]]))
    assert_array_equal(strategy.predict(np.array([[7, 7], [8, 8]])), predictions_b)
