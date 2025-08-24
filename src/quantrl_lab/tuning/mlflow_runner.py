import json
import os
import tempfile
from typing import Any, Callable, Dict, List, Optional

import mlflow
import mlflow.pytorch
import optuna
import pandas as pd
from optuna.integration.mlflow import MLflowCallback
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.traceback import install as install_rich_traceback

from ..backtesting.runner import BacktestRunner

# Install rich traceback handler
install_rich_traceback()

# Set up rich console
console = Console()


class MLflowOptunaRunner:
    """A comprehensive hyperparameter tuning and experiment tracking
    system that integrates MLflow for experiment logging and Optuna for
    optimization."""

    def __init__(
        self,
        runner: BacktestRunner,
        mlflow_tracking_uri: Optional[str] = None,
        experiment_name: str = "quantrl_optimization",
        storage_url: Optional[str] = None,
    ):
        """
        Initialize the runner with MLflow and Optuna configuration.

        Args:
            runner: BacktestRunner instance
            mlflow_tracking_uri: MLflow tracking URI (defaults to local ./mlruns)
            experiment_name: Name for the MLflow experiment
            storage_url: Optuna storage URL (defaults to in-memory)
        """
        self.runner = runner
        self.experiment_name = experiment_name

        # End any existing MLflow runs to avoid conflicts
        try:
            mlflow.end_run()
        except Exception:
            pass  # No active run to end

        # Setup MLflow
        self._setup_mlflow(mlflow_tracking_uri)

        # Setup Optuna storage
        self.storage_url = storage_url or "sqlite:///optuna_studies.db"

        # Initialize MLflow callback for Optuna
        self.mlflow_callback = MLflowCallback(tracking_uri=mlflow.get_tracking_uri(), metric_name="test_avg_return_pct")

    def _setup_mlflow(self, tracking_uri: Optional[str] = None):
        """Setup MLflow tracking configuration."""
        if tracking_uri is None:
            mlflow_dir = os.path.join(os.getcwd(), "mlruns")
            os.makedirs(mlflow_dir, exist_ok=True)
            tracking_uri = f"file://{mlflow_dir}"

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(self.experiment_name)

        console.print(
            Panel(
                f"[bold blue]MLflow Tracking URI:[/bold blue] {mlflow.get_tracking_uri()}\n"
                f"[bold blue]MLflow Experiment:[/bold blue] [green]{self.experiment_name}[/green]",
                title="[bold yellow]QuantRL-Lab Experiment[/bold yellow]",
                border_style="blue",
            )
        )

    def create_objective_function(
        self,
        algo_class,
        env_config: Dict[str, Any],
        search_space: Dict[str, Any],
        fixed_params: Optional[Dict[str, Any]] = None,
        total_timesteps: int = 50000,
        num_eval_episodes: int = 5,
        optimization_metric: str = "test_avg_return_pct",
    ) -> Callable:
        """
        Create an objective function for Optuna optimization where each
        trial is logged as a nested MLflow run.

        Args:
            algo_class: RL algorithm class (PPO, SAC, A2C, etc.)
            env_config: Environment configuration
            search_space: Dictionary defining the hyperparameter search space
            fixed_params: Fixed parameters that won't be optimized
            total_timesteps: Number of training timesteps
            num_eval_episodes: Number of evaluation episodes
            optimization_metric: Metric to optimize (default: test_avg_return_pct)

        Returns:
            Objective function for Optuna
        """
        fixed_params = fixed_params or {}

        def objective(trial: optuna.Trial) -> float:
            # This `with` block ensures that each Optuna trial is captured as a
            # new, nested run within the parent "study" run.
            with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True):
                try:
                    # Log tags for easy filtering in the MLflow UI
                    mlflow.set_tag("trial_number", str(trial.number))
                    mlflow.set_tag("algorithm", algo_class.__name__)

                    # Sample hyperparameters from the search space
                    params = self._sample_hyperparameters(trial, search_space)
                    params.update(fixed_params)

                    # Log all sampled and fixed parameters for this trial
                    mlflow.log_params(params)
                    mlflow.log_param("total_timesteps", total_timesteps)

                    # Create the algorithm-specific configuration
                    # Assuming `create_custom_config` is a static method on BacktestRunner
                    # or you have a way to generate the config dict.
                    if hasattr(self.runner, "create_custom_config"):
                        config = self.runner.create_custom_config(algo_class, **params)
                    else:
                        config = params.copy()

                    console.print(
                        f"[bold cyan]Starting Trial {trial.number}[/bold cyan] with params: [yellow]{params}[/yellow]"
                    )

                    # Run the backtesting experiment for this set of hyperparameters
                    results = self.runner.run_single_experiment(
                        algo_class=algo_class,
                        env_config=env_config,
                        config=config,
                        total_timesteps=total_timesteps,
                        num_eval_episodes=num_eval_episodes,
                    )

                    # Log the resulting metrics and artifacts
                    self._log_experiment_metrics(results)
                    self._log_experiment_artifacts(results, trial.number)

                    # Extract the target value for Optuna to optimize
                    target_value = results.get(optimization_metric)
                    if target_value is None:
                        console.print(
                            f"[bold yellow]⚠️ Optimization metric '[/bold yellow][cyan]{optimization_metric}[/cyan]"
                            f"[bold yellow]' not found in results.[/bold yellow] "
                            "Defaulting to -1000.0. Please check your BacktestRunner output.",
                            style="yellow",
                        )
                        target_value = -1000.0

                    mlflow.log_metric("optimization_target", target_value)
                    mlflow.set_tag("status", "completed")

                    console.print(
                        f"[bold green]Trial {trial.number} finished.[/bold green] "
                        f"[blue]{optimization_metric}[/blue] = [cyan]{target_value:.4f}[/cyan] ✓"
                    )

                    return target_value

                except Exception as e:
                    # If a trial fails, log the error, mark it as failed, and let Optuna prune it.
                    console.print(
                        f"[bold red]❌ Trial {trial.number} failed with an exception:[/bold red] {str(e)}", style="red"
                    )
                    console.print_exception()
                    mlflow.set_tag("status", "failed")
                    mlflow.log_param("error_message", str(e))

                    # Raising TrialPruned tells Optuna to discard this trial
                    # and not consider it when determining the best hyperparameters.
                    raise optuna.exceptions.TrialPruned()

        return objective

    def _sample_hyperparameters(self, trial: optuna.Trial, search_space: Dict[str, Any]) -> Dict[str, Any]:
        """Sample hyperparameters from the defined search space."""
        params = {}

        for param_name, param_config in search_space.items():
            param_type = param_config["type"]

            if param_type == "float":
                params[param_name] = trial.suggest_float(
                    param_name, param_config["low"], param_config["high"], log=param_config.get("log", False)
                )
            elif param_type == "int":
                params[param_name] = trial.suggest_int(
                    param_name, param_config["low"], param_config["high"], log=param_config.get("log", False)
                )
            elif param_type == "categorical":
                params[param_name] = trial.suggest_categorical(param_name, param_config["choices"])
            elif param_type == "discrete_uniform":
                params[param_name] = trial.suggest_discrete_uniform(
                    param_name, param_config["low"], param_config["high"], param_config["q"]
                )
            else:
                console.print(
                    f"[bold yellow]⚠️ Unknown parameter type:[/bold yellow] {param_type} for [cyan]{param_name}[/cyan]",
                    style="yellow",
                )

        return params

    def optimize_hyperparameters(
        self,
        algo_class,
        env_config: Dict[str, Any],
        search_space: Dict[str, Any],
        study_name: str,
        n_trials: int = 100,
        fixed_params: Optional[Dict[str, Any]] = None,
        total_timesteps: int = 50000,
        num_eval_episodes: int = 5,
        optimization_metric: str = "test_avg_return_pct",
        direction: str = "maximize",
        timeout: Optional[float] = None,
        n_jobs: int = 1,
        sampler: Optional[optuna.samplers.BaseSampler] = None,
        pruner: Optional[optuna.pruners.BasePruner] = None,
    ) -> optuna.Study:
        """
        Run hyperparameter optimization using Optuna, with MLflow for
        tracking.

        This method creates a parent MLflow run for the entire study,
        and each Optuna trial is logged as a separate, nested run.
        """

        # Ensure no previous run is active to avoid conflicts
        if mlflow.active_run():
            mlflow.end_run()

        # Set default sampler and pruner if not provided
        sampler = sampler or optuna.samplers.TPESampler(seed=42)
        pruner = pruner or optuna.pruners.MedianPruner()

        # Create or load the Optuna study
        study = optuna.create_study(
            study_name=study_name,
            storage=self.storage_url,
            direction=direction,
            sampler=sampler,
            pruner=pruner,
            load_if_exists=True,
        )

        # Create the objective function that handles its own MLflow logging
        objective_func = self.create_objective_function(
            algo_class=algo_class,
            env_config=env_config,
            search_space=search_space,
            fixed_params=fixed_params,
            total_timesteps=total_timesteps,
            num_eval_episodes=num_eval_episodes,
            optimization_metric=optimization_metric,
        )

        # Start the main parent run for the entire study
        with mlflow.start_run(run_name=f"optuna_study_{study_name}") as parent_run:
            console.print(
                f"[bold blue]Parent run for Optuna study[/bold blue] '[green]{study_name}[/green]' "
                f"[bold blue]started with ID:[/bold blue] [cyan]{parent_run.info.run_id}[/cyan]"
            )

            # Log high-level study parameters to the parent run
            mlflow.log_params(
                {
                    "study_name": study_name,
                    "n_trials": n_trials,
                    "optimization_metric": optimization_metric,
                    "direction": direction,
                    "sampler": sampler.__class__.__name__,
                    "pruner": pruner.__class__.__name__,
                }
            )

            # Log the search space and any fixed parameters as artifacts
            mlflow.log_dict(search_space, "search_space.json")
            if fixed_params:
                mlflow.log_dict(fixed_params, "fixed_params.json")

            console.rule(
                f"[bold blue]Starting optimization for [/bold blue][yellow]{n_trials}[/yellow][bold blue] trials[/bold blue]"  # noqa: E501
            )
            console.print()

            try:
                # Run the optimization.
                # CRITICAL: We do NOT pass the MLflowCallback here, as our
                # objective function is already handling all logging.
                study.optimize(
                    objective_func,
                    n_trials=n_trials,
                    timeout=timeout,
                    n_jobs=n_jobs,
                    callbacks=[],  # No callbacks needed for MLflow logging
                )

                # After optimization is complete, log the final study results
                # to the parent run.
                self._log_study_results(study)
                console.rule("[bold green]Optimization finished successfully[/bold green]")
                console.print(f"[bold blue]Best trial value:[/bold blue] [cyan]{study.best_value:.4f}[/cyan]")
                console.print("[bold blue]Best params:[/bold blue]")
                console.print(study.best_params, style="yellow")

            except Exception as e:
                console.print(
                    f"[bold red]❌ Optimization loop failed with an exception:[/bold red] {str(e)}", style="red"
                )
                console.print_exception()
                mlflow.set_tag("status", "CRASHED")
                raise  # Re-raise the exception after logging

        return study

    def _log_experiment_metrics(self, results: Dict[str, Any]):
        """Log experiment metrics to MLflow."""
        metrics = {
            "train_avg_return_pct": results.get("train_avg_return_pct", 0.0),
            "test_avg_return_pct": results.get("test_avg_return_pct", 0.0),
            "train_avg_reward": results.get("train_avg_reward", 0.0),
            "test_avg_reward": results.get("test_avg_reward", 0.0),
            "train_reward_std": results.get("train_reward_std", 0.0),
            "test_reward_std": results.get("test_reward_std", 0.0),
        }

        # Log episode counts
        if "train_episodes" in results:
            metrics["num_train_episodes"] = len(results["train_episodes"])
        if "test_episodes" in results:
            metrics["num_test_episodes"] = len(results["test_episodes"])

        mlflow.log_metrics(metrics)

    def _log_experiment_artifacts(self, results: Dict[str, Any], trial_number: int):
        """Log experiment artifacts to MLflow."""

        try:
            # Log action statistics
            if "train_action_stats" in results:
                f = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
                try:
                    json.dump(results["train_action_stats"], f, indent=2, default=str)
                    f.flush()
                    mlflow.log_artifact(f.name, f"trial_{trial_number}_train_action_stats.json")
                finally:
                    f.close()
                    os.unlink(f.name)

            if "test_action_stats" in results:
                f = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
                try:
                    json.dump(results["test_action_stats"], f, indent=2, default=str)
                    f.flush()
                    mlflow.log_artifact(f.name, f"trial_{trial_number}_test_action_stats.json")
                finally:
                    f.close()
                    os.unlink(f.name)

            # Log episode data as CSV
            if "train_episodes" in results and results["train_episodes"]:
                train_df = pd.DataFrame(results["train_episodes"])
                f = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
                try:
                    train_df.to_csv(f.name, index=False)
                    f.flush()
                    mlflow.log_artifact(f.name, f"trial_{trial_number}_train_episodes.csv")
                finally:
                    f.close()
                    os.unlink(f.name)

            if "test_episodes" in results and results["test_episodes"]:
                test_df = pd.DataFrame(results["test_episodes"])
                f = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
                try:
                    test_df.to_csv(f.name, index=False)
                    f.flush()
                    mlflow.log_artifact(f.name, f"trial_{trial_number}_test_episodes.csv")
                finally:
                    f.close()
                    os.unlink(f.name)

        except Exception as e:
            console.print(
                f"[bold yellow]⚠️ Failed to log artifacts for trial [/bold yellow][cyan]{trial_number}[/cyan]"
                f"[bold yellow]:[/bold yellow] {str(e)}",
                style="yellow",
            )

    def _log_study_results(self, study: optuna.Study):
        """Log Optuna study results to MLflow."""

        try:
            # Log best trial information
            best_trial = study.best_trial
            mlflow.log_params({f"best_{k}": v for k, v in best_trial.params.items()})
            mlflow.log_metric("best_value", study.best_value)
            mlflow.log_metric("n_trials", len(study.trials))

            # Create and log study summary
            trials_df = study.trials_dataframe()
            with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
                trials_df.to_csv(f.name, index=False)
                f.flush()
                mlflow.log_artifact(f.name, "optuna_trials.csv")
                os.unlink(f.name)

            # Log study statistics
            study_summary = {
                "best_params": best_trial.params,
                "best_value": study.best_value,
                "n_trials": len(study.trials),
                "study_name": study.study_name,
            }
            mlflow.log_dict(study_summary, "study_summary.json")

        except Exception as e:
            console.print(f"[bold yellow]⚠️ Failed to log study results:[/bold yellow] {str(e)}", style="yellow")

    def run_experiment_with_logging(
        self,
        algo_class,
        env_config: Dict[str, Any],
        run_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        total_timesteps: int = 50000,
        num_eval_episodes: int = 5,
        tags: Optional[Dict[str, str]] = None,
        nested: bool = False,
    ) -> Dict[str, Any]:
        """
        Run a single experiment with comprehensive MLflow logging.

        Args:
            algo_class: RL algorithm class
            env_config: Environment configuration
            run_name: Custom name for the MLflow run
            config: Algorithm configuration
            total_timesteps: Number of training timesteps
            num_eval_episodes: Number of evaluation episodes
            tags: Custom tags for the MLflow run
            nested: Whether to create a nested run (default: False)

        Returns:
            Experiment results dictionary
        """

        # End any existing runs only if not nested
        if not nested and mlflow.active_run():
            mlflow.end_run()
            console.print("[bold blue]Ended active MLflow run to prevent conflicts[/bold blue]")

        with mlflow.start_run(run_name=run_name, nested=nested):
            # Set tags
            if tags:
                mlflow.set_tags(tags)

            # Log parameters
            params = {
                "algorithm": algo_class.__name__,
                "total_timesteps": total_timesteps,
                "num_eval_episodes": num_eval_episodes,
            }

            if config:
                params.update({f"config_{k}": v for k, v in config.items() if isinstance(v, (str, int, float, bool))})

            mlflow.log_params(params)

            # Run experiment
            results = self.runner.run_single_experiment(
                algo_class=algo_class,
                env_config=env_config,
                config=config,
                total_timesteps=total_timesteps,
                num_eval_episodes=num_eval_episodes,
            )

            # Log metrics and artifacts
            self._log_experiment_metrics(results)
            self._log_experiment_artifacts(results, 0)

            return_value = results.get("test_avg_return_pct", 0.0)
            return_color = "green" if return_value >= 0 else "red"
            console.print(
                Panel(
                    f"[bold green]Experiment completed successfully[/bold green]\n"
                    f"[bold blue]Test return:[/bold blue] [{return_color}]{return_value:.2f}%[/{return_color}]",
                    title="[bold yellow]✓ Experiment Results ✓[/bold yellow]",
                    border_style="green",
                )
            )

            return results

    def compare_algorithms_with_logging(
        self,
        algorithms: List,
        env_config: Dict[str, Any],
        configs: Optional[Dict[str, Dict[str, Any]]] = None,
        total_timesteps: int = 50000,
        num_eval_episodes: int = 5,
        run_name_prefix: str = "algorithm_comparison",
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compare multiple algorithms with MLflow logging.

        Args:
            algorithms: List of algorithm classes to compare
            env_config: Environment configuration
            configs: Optional configurations for each algorithm
            total_timesteps: Number of training timesteps
            num_eval_episodes: Number of evaluation episodes
            run_name_prefix: Prefix for MLflow run names

        Returns:
            Dictionary of results for each algorithm
        """
        configs = configs or {}
        results = {}

        # End any existing MLflow runs
        if mlflow.active_run():
            mlflow.end_run()
            console.print("[bold blue]Ended active MLflow run to prevent conflicts[/bold blue]")

        with mlflow.start_run(run_name=f"{run_name_prefix}_parent"):
            # Log comparison parameters
            mlflow.log_params(
                {
                    "algorithms": [algo.__name__ for algo in algorithms],
                    "total_timesteps": total_timesteps,
                    "num_eval_episodes": num_eval_episodes,
                    "n_algorithms": len(algorithms),
                }
            )

            for algo_class in algorithms:
                algo_name = algo_class.__name__
                algo_config = configs.get(algo_name, {})

                console.print(f"[bold blue]Running[/bold blue] [green]{algo_name}[/green][bold blue]...[/bold blue]")

                # Run algorithm with nested run
                algo_results = self.run_experiment_with_logging(
                    algo_class=algo_class,
                    env_config=env_config,
                    run_name=f"{run_name_prefix}_{algo_name}",
                    config=algo_config,
                    total_timesteps=total_timesteps,
                    num_eval_episodes=num_eval_episodes,
                    tags={"comparison_group": run_name_prefix, "algorithm": algo_name},
                    nested=True,
                )

                results[algo_name] = algo_results

            # Log comparison summary
            comparison_summary = {
                algo: {
                    "test_return_pct": results[algo].get("test_avg_return_pct", 0.0),
                    "test_reward": results[algo].get("test_avg_reward", 0.0),
                }
                for algo in results.keys()
            }

            mlflow.log_dict(comparison_summary, "algorithm_comparison_summary.json")

            # Log best algorithm
            best_algo = max(results.keys(), key=lambda x: results[x].get("test_avg_return_pct", -float("inf")))
            mlflow.log_params(
                {"best_algorithm": best_algo, "best_test_return": results[best_algo].get("test_avg_return_pct", 0.0)}
            )

            # Create a table for algorithm comparison results
            comparison_table = Table(title="Algorithm Comparison Results")
            comparison_table.add_column("Algorithm", style="cyan")
            comparison_table.add_column("Test Return (%)", style="green")
            comparison_table.add_column("Train Return (%)", style="blue")

            # Add rows for each algorithm
            for algo, result in results.items():
                test_return = result.get("test_avg_return_pct", 0.0)
                train_return = result.get("train_avg_return_pct", 0.0)
                test_color = "green" if test_return >= 0 else "red"
                train_color = "green" if train_return >= 0 else "red"

                comparison_table.add_row(
                    algo,
                    f"[{test_color}]{test_return:.2f}%[/{test_color}]",
                    f"[{train_color}]{train_return:.2f}%[/{train_color}]",
                )

            console.print(
                Panel(
                    console.render_str(comparison_table),
                    title="[bold yellow]Algorithm Comparison Completed ✓[/bold yellow]",
                    subtitle=f"[bold green]Best Algorithm: [/bold green][yellow]{best_algo}[/yellow]",
                    border_style="green",
                )
            )

        return results


# Utility functions for creating search spaces
def create_ppo_search_space() -> Dict[str, Any]:
    """Create a default search space for PPO hyperparameters."""
    return {
        "learning_rate": {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
        "n_steps": {"type": "categorical", "choices": [256, 512, 1024, 2048, 4096]},
        "batch_size": {"type": "categorical", "choices": [32, 64, 128, 256, 512]},
        "gamma": {"type": "float", "low": 0.9, "high": 0.9999},
        "gae_lambda": {"type": "float", "low": 0.8, "high": 1.0},
        "clip_range": {"type": "float", "low": 0.1, "high": 0.4},
        "ent_coef": {"type": "float", "low": 1e-8, "high": 1e-1, "log": True},
    }


def create_sac_search_space() -> Dict[str, Any]:
    """Create a default search space for SAC hyperparameters."""
    return {
        "learning_rate": {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
        "batch_size": {"type": "categorical", "choices": [64, 128, 256, 512]},
        "gamma": {"type": "float", "low": 0.9, "high": 0.9999},
        "tau": {"type": "float", "low": 0.001, "high": 0.1},
        "train_freq": {"type": "categorical", "choices": [1, 4, 8, 16]},
        "gradient_steps": {"type": "categorical", "choices": [1, 2, 4, 8]},
        "target_update_interval": {"type": "categorical", "choices": [1, 2, 4, 8]},
    }


def create_a2c_search_space() -> Dict[str, Any]:
    """Create a default search space for A2C hyperparameters."""
    return {
        "learning_rate": {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
        "n_steps": {"type": "categorical", "choices": [5, 16, 32, 64, 128]},
        "gamma": {"type": "float", "low": 0.9, "high": 0.9999},
        "gae_lambda": {"type": "float", "low": 0.8, "high": 1.0},
        "ent_coef": {"type": "float", "low": 1e-8, "high": 1e-1, "log": True},
        "vf_coef": {"type": "float", "low": 0.1, "high": 1.0},
    }
