import unittest

from strbo import Dimension, SearchSpace, StrBOConfig, Trial, create_study
from strbo.gp import ExactStringGaussianProcess
from strbo.kernels import NGramStringKernel


class StrBOTests(unittest.TestCase):
    def test_ngram_kernel_prefers_similar_strings(self) -> None:
        kernel = NGramStringKernel(max_ngram=3)

        self.assertAlmostEqual(kernel("search_width=08", "search_width=08"), 1.0)
        self.assertGreater(
            kernel("search_width=08", "search_width=09"),
            kernel("search_width=08", "filter_sim=0.95"),
        )

    def test_gp_predicts_training_order(self) -> None:
        gp = ExactStringGaussianProcess(kernel=NGramStringKernel(max_ngram=2), noise=1e-6)
        gp.fit(["x:int:00", "x:int:05"], [0.0, 10.0])

        left = gp.predict("x:int:00")
        right = gp.predict("x:int:05")

        self.assertLess(left.mean, right.mean)
        self.assertGreaterEqual(left.variance, 0.0)
        self.assertGreaterEqual(right.variance, 0.0)

    def test_study_runs_suggest_trial_api(self) -> None:
        space = SearchSpace(
            (
                Dimension.integer("x", 0, 4),
                Dimension.categorical("flag", (True, False)),
            )
        )
        study = create_study(
            study_name="toy",
            space=space,
            direction="minimize",
            config=StrBOConfig(seed=7, n_initial=2, candidate_pool_size=32),
        )

        def objective(trial: Trial) -> float:
            x = trial.suggest_int("x", 0, 4)
            flag = trial.suggest_categorical("flag", [True, False])
            trial.set_user_attr("seen", True)
            return float((x - 2) ** 2 + (0.0 if flag else 0.5))

        study.optimize(objective, n_trials=5)

        self.assertEqual(len(study.trials), 5)
        self.assertTrue(all(trial.user_attrs["seen"] for trial in study.trials))
        self.assertEqual(study.best_trial.value, study.best_value)
        self.assertEqual(len({space.key(trial.params) for trial in study.trials}), 5)


if __name__ == "__main__":
    unittest.main()
